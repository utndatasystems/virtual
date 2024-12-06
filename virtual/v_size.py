import duckdb
import os
import pandas as pd
import pyarrow.parquet as pq
from typing import List
import utils
from utils import ModelType, _get_info, _get_nullable, _get_column, get_data_hash
import math
import pathlib
from typing import Optional

PARQUET_COMPRESSION_TYPE = 'snappy'

# TODO: At compression time, we should also put the schema and functions to avoid the slow table rewrite at the end.
def arrow_compress_parquet(in_file, out_file):
  table = pq.read_table(in_file)
  pq.write_table(table, out_file, compression=PARQUET_COMPRESSION_TYPE)

def _model_needs_offset(iter, model_type):
  return True
  # return iter["metrics"][model_type]['equal'] != 1.0

# TODO: Do we really need this?
def _model_has_null_reference_columns(schema, model, model_type: ModelType):
  # Is the model a k-regression?
  if model_type.is_k_regression():
    # TODO: What if that model is never selected.
    # TODO: I mean, if that's the case, we should directly drop it in the explorer, right?
    for local_model in model['config']:
      for iter in local_model['coeffs']:
        if _get_column(schema, iter['col_name'])['null']['any']:
          return True
  else:
    for iter in model['coeffs']:
      if _get_column(schema, iter['col_name'])['null']['any']:
        return True
  return False

def _is_target_column_null(schema, target_column):
  return _get_column(schema, target_column['target_name'])['null']['any']

def get_model(iter, model_type: ModelType):
  # Have we already assigned a model type to this column?
  if 'model_type' in iter:
    # If so, just return the actual model.
    return iter[iter['model_type']], ModelType(iter['model_type'])
  else:
    # Otherwise, we must have a custom `model_type`.
    # In that case, check if it is present as a model.
    if 'models' in iter:
        # Nothing found?
        if model_type.name not in iter['models']:
          return None, model_type
        
        # Otherwise, return the model.
        return iter['models'][model_type.name], model_type
    
  # This should never happen.
  assert 0
  return None, None

def _create_base_table(target_columns, schema, model_type=None):
  columns = []
  for iter in schema:
    type = _get_info(schema, iter['name'])['type']
    nullable = _get_nullable(schema, iter['name'])
    columns.append(f'"{iter["name"]}" {type} {nullable}')

  def create_offset_column(target_name):
    name = f"{target_name}_offset"
    type = _get_info(schema, target_name)['type']
    return f"\"{name}\" {type} not null default 0"

  def create_outlier_column(target_name):
    name = f"{target_name}_outlier"
    type = _get_info(schema, target_name)['type']
    return f"\"{name}\" {type} null default null"

  def create_is_null_column(target_name):
    # NOTE: Only change this if you know what to do.
    # We use split('_') later and this could destroy the functionality of the script.
    name = f"{target_name}_null"
    return f"\"{name}\" bool not null default false"

  def create_switch_column(target_name):
    name = f"{target_name}_switch"
    # TODO: You need to restore to `not null`.
    # TODO: There was a bug in taxi, that's why we disabled it.
    return f"\"{name}\" smallint not null default 0"

  def create_formula_column(target_name, index):
    name = f"{target_name}_f{index}"

    # TODO: Actually, to avoid any problems, we should make this as large as it gets.
    type = _get_info(schema, target_name)['type']
    return f"\"{name}\" {type}"

  def create_least_column(target_name):
    name = f"{target_name}_least"
    type = _get_info(schema, target_name)['type']

    # TODO: This part should be signed, I think.
    # Or not, since we do abs.
    # But the operation inside that should be casted as double, right???
    return f"\"{name}\" {type}"

  # def create_temp_offset_column(target_name):
  #    name = f"{target_name}_temp_offset"
  #    type = _get_info(schema, target_name)['type']
  #    return f"\"{name}\" {type} default 0"

  # TODO: Maybe we can later remove these columns if they're not used.
  for iter in target_columns:
    # Fetch the model.
    model, model_type = get_model(iter, model_type)

    # No model? This can only happen if the `model_type` was specified before hand. 
    if model is None:
      assert model_type is not None
      continue

    # Do we need an `offset` column?
    # NOTE: Currently, we always create one. We remove it afterwards in an optimization step.
    if _model_needs_offset(iter, model_type):
      columns.append(create_offset_column(iter['target_name']))

    # Do we need an `outlier` column? This happens if one of the reference columns has NULLs.
    if _model_has_null_reference_columns(schema, model, model_type):
      columns.append(create_outlier_column(iter['target_name']))

    # Do we need an `null` column? This happens if the column has NULLs.
    if _is_target_column_null(schema, iter):
      columns.append(create_is_null_column(iter['target_name']))

    # Add the switch column.
    if model_type.is_k_regression():
        # TODO: Maybe don't do this if we actually don't find anything.
        # TODO: But this can already be done in the explorer.
        columns.append(create_switch_column(iter['target_name']))

    # This part only concerns temporary columns.
    if model_type.is_k_regression():
        # TODO: The `k` should be optimized in the explorer.
        k = model_type.extract_k()

        for index in range(k):
          columns.append(create_formula_column(iter['target_name'], index))
        
        # Add the `least` column.
        columns.append(create_least_column(iter['target_name']))

    # Add a temporary offset column which can be null.
    # if utils.is_k_regression(model_type):
    #    columns.append(create_temp_offset_column(iter['target_name']))

  columns = " , ".join(columns)

  return f"CREATE TABLE base_table ({columns});"

def _create_sample(data: pd.DataFrame | pathlib.Path, sample_path, sample_size):
  # If there is only a path, first read it.
  if isinstance(data, pathlib.Path):
    # TODO: We should ensure there is no header!
    df = pd.read_csv(data, dtype=str, nrows=sample_size, header=None)
  elif isinstance(data, pd.DataFrame):
    # Otherwise, use the dataframe we already have.
    if sample_size < len(data):
      df = data.iloc[:sample_size]
    else:
      df = data
  
  # Dump to `sample_path`.
  df.to_csv(sample_path, index=False, header=False)

def create_copy_csv(data: pd.DataFrame | pathlib.Path, schema, nrows=None):
  assert isinstance(data, (pd.DataFrame, pathlib.Path))

  # Take all columns we have to read.
  columns = ','.join([f'"{iter["name"]}"' for iter in schema])

  # Path?
  if isinstance(data, pathlib.Path):
    # Infer the delimiter.
    dialect = utils.infer_dialect(data)

    # Create the load SQL.
    # TODO: I now know why it works with the estimator and not with this one.
    # TODO: We always created the sample that uses `_create_sample` (pandas.df), which probably removed the NULLs.
    # TODO: We can do this better, really.
    load_sql = f"insert into base_table({columns}) select * from read_csv('{data}', header=true, delim='{dialect['delimiter']}', quote='{dialect['quotechar']}'" \
            + (f", sample_size={nrows}" if nrows is not None else "") \
            + ", nullstr=['NULL', 'null', '', 'None'])"
  elif isinstance(data, pd.DataFrame):
    # Then we have a dataframe.
    # TODO: Do we still need this one?
    # TODO: Should we actually say `TIMESTAMP` instead of `timestamp`?
    # TODO: Wait -- we might actually modify user's data?
    for i in schema:
      if i['type'] == 'timestamp':
        data[i['name']] = pd.to_datetime(data[i['name']])
    load_sql = f'insert into base_table({columns}) select * from data'
  else:
    assert 0

  # Add the LIMIT clause to bound until `nrows`.
  if nrows is not None:
    load_sql = f'{load_sql} limit {nrows}'

  # And return the SQL query.
  return load_sql

# Create a regression model.
def _create_regression(model, schema, target_name):
  # This will contain the regression components.
  regression = []

  # Get the information about the column.
  info = _get_info(schema, target_name)

  formula_has_only_integers = True
  all_coeffs_are_integers = True

  scales = []
  for iter in model['coeffs']:
    # Is the coefficient 0?
    # Then skip this part.
    if math.isclose(iter['coeff'], 0.0):
      continue

    # TODO: What if the coefficient is 0? We could skip this.
    # TODO: But then we also need to adapt this for the `is null` part.
    # TODO: This is wrong.
    if not utils.is_integer(_get_info(schema, iter['col_name'])['type']):
      formula_has_only_integers = False

    # Take the current scale.
    scales.append(_get_info(schema, iter['col_name'])['scale'])

    # Optimize out a 1.0 coefficient.
    if math.isclose(iter['coeff'], 1.0):
      regression.append(f"\"{iter['col_name']}\"")
    else:
      rounded_coeff = utils.custom_round(iter['coeff'])

      # Check if it is an integer.
      if not rounded_coeff.is_integer():
        formula_has_only_integers = False
        all_coeffs_are_integers = False
      else:
        rounded_coeff = int(rounded_coeff)
      regression.append(f"{rounded_coeff} * \"{iter['col_name']}\"")

  # Analyze the intercept.
  intercept = utils.custom_round(model['intercept'])

  # Not 0?
  if not math.isclose(intercept, 0.0):
    # TODO: Should we also round this?
    if not intercept.is_integer():
      all_coeffs_are_integers = False
      formula_has_only_integers = False
    else:
      # Otherwise make it a true integer.
      intercept = int(intercept)
    regression.append(str(intercept))
  elif len(regression) == 0:
    # NOTE: This is really a corner case. I am really glad we caught it.
    # This happens if everything is 0.
    regression.append(str(0))

  # And build the formula.
  formula = ' + '.join(regression)

  # Is the target column an integer?
  # Then force the offset to also be one.
  # TODO: Note, this can also overflow.
  if utils.is_integer(info['type']):
    actual_type = info['type']

    # Only round if there was a different type than integer in the formula.
    # Otherwise, let is as is.
    # TODO: Probably problems with overflow?
    if not formula_has_only_integers:
      formula = f'round({formula}, 0)::{actual_type}'
  elif utils.is_double(info['type']):
    scale = info['scale']

    # If we only have this scale.
    # TODO: Maybe even with lower scales would be better?
    # TODO: For instance, if we have integers.
    # EXAMPLE: [2, 2, 0, 1], and we have the scale = 2, than this would work.
    # So if scale >= max(scale).

    # Corner case: There is only the intercept in the regression.
    if not len(model['coeffs']):
      assert not len(scales)
      assert len(regression) == 1
      pass
    elif scale >= max(scales) and all_coeffs_are_integers:
      # NOTE: This only if all coefficients are integers.
      # Otherwise, we need to cast.
      pass
    else:
      # TODO: Maybe this is the right place to cast to the actual type.
      # TODO: But we keep DOUBLE for now.
      formula = f'round({formula}, {scale})'

  # And return.
  return formula

def _create_abs_offset(formula, col):
  return f"abs(\"{col}\" - {formula})"

def create_dump_base(con, schema, target_columns, out_file):
  columns = ",".join([f'"{iter["name"]}"' for iter in schema])
  return f"copy (select {columns} from base_table) to '{out_file}' (FORMAT PARQUET, COMPRESSION '{PARQUET_COMPRESSION_TYPE}');"

def create_dump_base_target_column(con, schema, target_column, out_file):
  return f"copy (select \"{target_column['target_name']}\" from base_table) to '{out_file}' (FORMAT PARQUET, COMPRESSION '{PARQUET_COMPRESSION_TYPE}');"

# NOTE: You need to adapt this part if you want change the layout.
def is_auxiliary_column(cn):
  token = cn.split('_')[-1].strip('"')
  return utils.is_formula(token) or token == 'least'

def is_special_column(cn):
  token = cn.split('_')[-1].strip('"')
  return token in ['switch', 'offset', 'outlier', 'null']

def clean_base_table(all_columns, target_columns):
  rest = all_columns.copy()
  for target_column in target_columns:
    for cn in all_columns:
      # TODO: We also need to take into account "...", right?
      # TODO: This is really a hack.
      # NOTE: Pay attention at this case: average_admissions_30_49_covid_confirmed_per_100k vs average_admissions_30_49_covid_confirmed.
      if utils.is_attached_column_of(target_column['target_name'], cn) and is_auxiliary_column(cn):
        rest.remove(cn)
  return rest

def create_dump_virtual(con, schema, target_columns, out_file):
  # Get the current header.
  all_columns = utils._get_curr_columns(con)

  # Clean the table.
  all_columns = clean_base_table(all_columns, target_columns) 

  # Remove the original target column names.
  # TODO: We should take care of spaces in the column names, right?
  for iter in target_columns:
    all_columns.remove(iter["target_name"])

  # And flush to parquet.
  all_columns = [f"\"{iter}\"" for iter in all_columns]
  all_columns = ",".join(all_columns)
  return f"copy (select {all_columns} from base_table) to '{out_file}' (FORMAT PARQUET, COMPRESSION '{PARQUET_COMPRESSION_TYPE}');"

def create_dump_virtual_target_column(con, schema, target_column, out_file):
  # Get the current header.
  all_columns = utils._get_curr_columns(con)

  # Clean the table.
  all_columns = clean_base_table(all_columns, [target_column]) 

  # Remove the target column.
  all_columns.remove(target_column['target_name'])

  # Take the attached column.
  keep = []
  for cn in all_columns:
    # TODO: Do we need to use quotes?
    if utils.is_attached_column_of(target_column['target_name'], cn):
        keep.append(cn)
  
  # If there is nothing to keep, just store a dummy parquet file.
  # TODO: Note - if you are to read this, it becomes complicated.
  select_clause = '1 as dummy_column'
  if keep:
    keep = [f"\"{iter}\"" for iter in keep]
    select_clause = ",".join(keep)

  # And flush to parquet.
  return f"copy (select {select_clause} from base_table) to '{out_file}' (FORMAT PARQUET, COMPRESSION '{PARQUET_COMPRESSION_TYPE}');"

def build_least_column(model, schema, target_iter):
  least_builder = []
  updates = []
  for index, local_model in enumerate(model['config']):
    # NOTE: This formula has already taken into consideration the type of the target column.
    local_formula = _create_regression(local_model, schema, target_iter['target_name'])

    # Build the `least` column.
    # TODO: The problem is that the difference could be negative.
    # TODO: Or maybe we do an if.
    # TODO: Done, but this could still lead to an overflow.
    local_offset = _create_abs_offset(local_formula, target_iter['target_name'])
    least_builder.append(local_offset)

    updates.append(f"\"{target_iter['target_name']}_f{index}\" = {local_formula}")

  # Build.
  least_builder_str = ', '.join(least_builder)
  least_builder_str = f"\"{target_iter['target_name']}_least\" = least({least_builder_str})"

  updates.append(least_builder_str)

  # And return.
  return f"update base_table set {', '.join(updates)};"

def build_switch_column(model, schema, target_iter):
  switch_builder = []
  for index, local_model in enumerate(model['config']):
    local_formula = _create_regression(local_model, schema, target_iter['target_name'])
    local_offset = _create_abs_offset(local_formula, target_iter['target_name'])
    switch_builder.append(f"when {local_offset} - \"{target_iter['target_name']}_least\" < 1e-6 then {index}")
  
  # Build. Please keep the space before the first ` end`, otherwise we have problems.
  switch_builder_str = f"case when \"{target_iter['target_name']}_least\" is null then 0 else (case " + '\n'.join(switch_builder) + " end) end"
  
  # And return.
  return f"update base_table set \"{target_iter['target_name']}_switch\" = {switch_builder_str};"

def build_offset_column(model, schema, target_iter):
  # TODO: Maybe we can put the switch to -1, and then update it to 0.
  offset_builder = []
  for index, local_model in enumerate(model['config']):
    local_formula = _create_regression(local_model, schema, target_iter['target_name'])
    
    # TODO: This offset can be signed, so it's pretty bad for the type.
    offset_builder.append(f"when \"{target_iter['target_name']}_switch\" = {index} then \"{target_iter['target_name']}\" - {local_formula}")
  
  # Build.
  offset_builder_str = f"case when \"{target_iter['target_name']}_least\" is null then 0 else (case " + '\n'.join(offset_builder) + " end) end"
  
  # And return.
  return f"update base_table set \"{target_iter['target_name']}_offset\" = {offset_builder_str};"

def debug_base_table(con, msg, verbose=False):
  if verbose:
    print(f'\n ---- @@@ {msg} @@@ ---- \n')
    tmp = con.execute(f'select * from base_table;').fetchdf()
    print(tmp[tmp.columns.sort_values()].head(100).to_markdown())
  pass

def create_virtual_column_layout(con, target_iter, schema, model_type: Optional[ModelType]=None):
  def _create_sql_any_reference_column_is_null(model):
    # TODO: But this is _not_ the point, since we just want to check if it is null.
    # TODO: Not that this is nullable.
    # TODO: Even in that case - why should be the switch column NULL when all are nulls?
    reference_columns = [it for it in model["coeffs"] if _get_column(schema, it["col_name"])["null"]]
    reference_columns = [f"\"{it['col_name']}\" is null" for it in reference_columns]
    reference_columns = ' or '.join(reference_columns)

    # If we really do not have any nullable columns, just return `false` (the boolean SQL type).
    if not reference_columns:
      return 'false'
    
    # Otherwise, return the default.
    return reference_columns

  # In case the column has _only_ NULLs, skip.
  if _get_column(schema, target_iter['target_name'])['null']['all']:
    return []

  # Fetch the model.
  model, model_type = get_model(target_iter, model_type)
  if model is None:
    return []

  # The result.
  local_updates = []

  # Add temporary columns in case of a k-regression.
  if model_type.is_k_regression():      
    try:
      # Build the least column.
      con.execute(build_least_column(model, schema, target_iter))

      debug_base_table(con, 'after least')
      
      # Build the switch column.
      con.execute(build_switch_column(model, schema, target_iter))

      debug_base_table(con, 'after switch')
      
      # Build the offset column.
      con.execute(build_offset_column(model, schema, target_iter))

      debug_base_table(con, 'after offset')
    except Exception as e:
      # TODO: Just a hack to avoid other issues.
      # TODO: This fails for the divvy dataset, where the intercept is too large.
      # TODO: Yet, sparse solves this much better.
      return []

  # Add offset column.
  if not model_type.is_k_regression():
    # TODO: There is a problem here, since the offset doesn't take into account the outlier.
    if _model_needs_offset(target_iter, model_type):
      # Create the regression.
      formula = _create_regression(model, schema, target_iter['target_name'])

      # NOTE | TODO: The offset should also take the scale of the original column?
      update = f"round(\"{target_iter['target_name']}\" - ({formula}), {utils._get_info(schema, target_iter['target_name'])['scale']})"
      update = f"coalesce({update}, 0)"
      update = f"\"{target_iter['target_name']}_offset\" = {update}"
      local_updates.append(update)
  
  # Check for null reference columns.
  if _model_has_null_reference_columns(schema, model, model_type):
    # Simple regression?
    expr = None
    if not model_type.is_k_regression():
      ref_is_null = _create_sql_any_reference_column_is_null(model)
      expr = f"case when {ref_is_null} then \"{target_iter['target_name']}\" else null end"
    else:
      # At this point, we need to only take _our_ reference columns, i.e., those corresponding to the switch.         
      # Build the expressions for the `is null` expressions.
      ref_is_null = []
      for it in model['config']:
        ref_is_null.append(_create_sql_any_reference_column_is_null(it))

      # Build the outlier column.
      outlier_builder = []
      for index in range(len(model['config'])):
        col = target_iter['target_name']

        # Inspect the switch.            
        outlier_builder.append(f"when \"{col}_switch\" = {index} then (case when {ref_is_null[index]} then null else \"{col}\" end)")
      expr = f'case ' + '\n'.join(outlier_builder) + ' end'

    # Set the outlier column.
    assert expr is not None
    local_updates.append(f"\"{target_iter['target_name']}_outlier\" = {expr}")

  # Check if the column can be null.
  if _is_target_column_null(schema, target_iter):
    update = f"\"{target_iter['target_name']}_null\" = \"{target_iter['target_name']}\" is null"
    local_updates.append(update)

  # And return.
  return local_updates

# TODO: This is the one with the issue.
def create_virtual_table_layout(con, target_columns, schema, model_type=None):
  # Virtualize each possible column.
  for iter in target_columns:
    curr_updates = create_virtual_column_layout(con, iter, schema, model_type)

    # No updates? Then skip.
    if not curr_updates:
      continue

    # Try to update it.
    try:
      update_sql = f"update base_table set {','.join(curr_updates)};"
      con.execute(update_sql)

      debug_base_table(con, f"{iter['target_name']} @@@@@ after the updates!!!!!!!!!!")
    except Exception as e:
      # Otherwise, skip it.
      continue
  return

def reoptimize_virtual_table_layout(con, target_columns):
  freqs = utils._get_distinct_counts(con)

  to_remove = []
  for target_column in target_columns:
    for cn in freqs:
      # Is it actually distinct?
      if freqs[cn][0] != 1:
        continue

      assert cn.endswith('_count')
      cn = utils.rreplace(cn, '_count', '')

      # NOTE: This is really important!
      if cn == target_column['target_name']:
        continue

      # Then remove it if it is an auxiliary column.
      if utils.is_attached_column_of(target_column['target_name'], cn) and is_special_column(cn):
        to_remove.append(cn)

  if len(to_remove):
    for cn in to_remove:
      con.execute(f"alter table base_table drop column \"{cn}\";")
  pass

def _size_impl(con, schema, target_column, model_type=None, data_hash=None):
  assert data_hash is not None

  def get_size(dump_callback, target_column):
    out_file = "size"

    sql = dump_callback(con, schema, target_column, f'.{out_file}_duckdb-{data_hash}.parquet')
    if len(sql) == 0:
      return 0

    # Execute.
    con.execute(sql)

    # Compress via arrow.
    arrow_compress_parquet(f'.{out_file}_duckdb-{data_hash}.parquet', f'.{out_file}_arrow-{data_hash}.parquet')

    # Calculate the final size.
    size_to_return = os.path.getsize(f'.{out_file}_arrow-{data_hash}.parquet')

    # Remove both parquet files, since they are just temporary.
    os.remove(f'.{out_file}_duckdb-{data_hash}.parquet')
    os.remove(f'.{out_file}_arrow-{data_hash}.parquet')

    # And return the size.
    return size_to_return

  # Fetch the model.
  model, model_type = get_model(target_column, model_type)
  if model is None:
    return

  # The raw size of the base column.
  base_target_column_size = get_size(create_dump_base_target_column, target_column)

  # Its virtual size.
  virtual_target_column_size = get_size(create_dump_virtual_target_column, target_column)
  
  # Add the sizes into the dictionary.
  if 'sizes' not in target_column:
    target_column['sizes'] = dict()
  
  # NOTE: This is _not_ an `elif`!
  if model_type.name not in target_column['sizes']:
    target_column['sizes'][model_type.name] = dict()

  # And add them.
  target_column['sizes'][model_type.name]['base'] = base_target_column_size
  target_column['sizes'][model_type.name]['virtual'] = virtual_target_column_size
  return

# We use this function to estimate the {arquet sizes for the columns.
# We say "estimate", since we only analyze a sample size of 10K tuples.
def compute_sizes_of_target_columns(functions, data: pd.DataFrame | pathlib.Path, schema, model_types: List[ModelType], sample_size=10_000):
  assert isinstance(data, (pd.DataFrame, pathlib.Path))
  # TODO: Maybe we can sample only once?

  # Construct an unique hash.
  data_hash = get_data_hash(data)

  for model_type in model_types:
    # Make a new connection to be sure that we do not use the same tables as the previous model.
    con = duckdb.connect(database=':memory:', read_only=False)

    # Create the base table.
    # NOTE: This also creates the auxiliary columns (only the schema thereof).
    con.execute(_create_base_table(functions, schema, model_type))
    sample_path = f'.sample-{data_hash}.csv'
    
    debug_base_table(con, 'after create basetable')

    # Load the sample.
    _create_sample(data, sample_path, sample_size)

    # Copy the CSV into the SQL table.
    con.execute(create_copy_csv(pathlib.Path(sample_path), schema, nrows=sample_size))

    # Remove the sample.
    os.remove(sample_path)

    # Finalize the layout. This works for both simple and k-regression.
    create_virtual_table_layout(con, functions, schema, model_type)

    debug_base_table(con, 'After `create_virtual_table_layout`.')

    # Compute the base and virtual sizes.
    for target_column in functions:
      _size_impl(con, schema, target_column, model_type=model_type, data_hash=data_hash)

    # And close the connection.
    con.close()

  return functions