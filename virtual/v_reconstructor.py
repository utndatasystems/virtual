import time
import duckdb
import numpy as np
import utils
from utils import ModelType
import expr_parser, expr_optimizer
from v_size import _create_regression

def collect_required_virtual_columns(target_iter, model_type, all_virtual_columns):
  cols = []
  if not model_type.is_k_regression():
    cols = utils.collect_lr_column_names(target_iter[model_type.name])
  else:
    for local_model in target_iter[model_type.name]['config']:
      cols.extend(utils.collect_lr_column_names(local_model))
  return cols

def generate_formula(schema, parquet_header, target_iter, all_virtual_columns, enforce_replacement=False):
  # Fetch the model type.
  assert 'model_type' in target_iter
  model_type = ModelType(target_iter['model_type'])

  # Determine which virtual columns we require to build extra.
  required = collect_required_virtual_columns(target_iter, model_type, all_virtual_columns)
  assert target_iter['target_name'] not in required

  # Create the mapping from the name to their expression.
  mapping = dict()
  for other_iter in all_virtual_columns:
    if other_iter['target_name'] in required:
      _, expr = generate_formula(schema, parquet_header, other_iter, all_virtual_columns, enforce_replacement=enforce_replacement)
      mapping[other_iter['target_name']] = expr

  # Declare the columns we will be using.
  # TODO: Add switch.
  is_null_col = f"{target_iter['target_name']}_null"
  outlier_col = f"{target_iter['target_name']}_outlier"
  offset_col = f"{target_iter['target_name']}_offset"
  # switch_col = f"\"{target_iter['target_name']}_switch\""

  # Simple regression.
  if not model_type.is_k_regression():
    formula = _create_regression(target_iter[model_type.name], schema, target_iter['target_name'])
  else:
    local_fs = []
    for index, local_model in enumerate(target_iter[model_type.name]['config']):
      local_fs.append(_create_regression(local_model, schema, target_iter['target_name']))

    builder = []
    for index, local_model in enumerate(target_iter[model_type.name]['config']):
      # TODO: This offset can be signed, so it's pretty bad for the type.
      if index != len(target_iter[model_type.name]['config']) - 1:
        builder.append(f"when \"{target_iter['target_name']}_switch\" = {index} then {local_fs[index]}")
      else:
        # Optimize the last `when` since we can directly put `else`.
        builder.append(f"else {local_fs[index]}")
    formula = f'case ' + '\n'.join(builder) + ' end'

  # Do we have an offset column?
  # TODO: If the offset column is a constant, we have to optimize it out.
  # TODO: Or maybe just remove it before hand within the optimizer.
  # TODO: Since we anyway do this check if we have an offset column.
  if offset_col in parquet_header:
    assert formula and formula is not None
    formula = formula + f" + \"{offset_col}\""

  # Do we have an outlier column?
  if outlier_col in parquet_header:
    formula = f"coalesce (\"{outlier_col}\", {formula})"
  
  # Do we have an `is_null` column?
  if is_null_col in parquet_header:
    formula = f"case when \"{is_null_col}\" then null else {formula} end"

  # If we really need to enforce the replacement of the required virtual columns.
  if enforce_replacement:
    for elem in mapping:
      formula = formula.replace(f"\"{elem}\"", f"({mapping[elem]})")

  # Optimize the formula.
  new_formula = None
  if not model_type.is_k_regression():
    parser = expr_parser.ExprParser(formula)
    formula_tree = parser.parse()
    optimized_tree = expr_optimizer.optimize(formula_tree)
    new_formula = expr_parser.print_expr(optimized_tree)
  else:
    # TODO: Implement the optimization for k-regression.
    new_formula = formula
  assert new_formula is not None

  # Strip out the '(' and ')'.
  if new_formula[0] == '(' and new_formula[-1] == ')':
    new_formula = new_formula[1:-1]

  # Update the formula with the optimized one.
  formula = new_formula

  # And return it, also with a suffix with the actual column name.
  return f"{formula} as \"{target_iter['target_name']}\"", formula

def _measure_impl(bench_col, parquet_path):
  # Open the connection.
  local_con = duckdb.connect(database=':memory:', read_only=False)

  # Build the sql query.
  assert bench_col is not None
  sql_query = f"select sum({bench_col}) as sum_ from read_parquet('{parquet_path}');" #  + (f" where {filter}" if filter else "") + ";"

  lats = []
  ret = None

  # NOTE: We do a warm-up query to avoid measuring the disk read.
  for _ in range(11):
    start = time.time_ns()
    try:
      # NOTE: We also fetch the result to be sure.
      ret = local_con.execute(sql_query).fetchdf()["sum_"][0]
    except Exception as e:
      print(e)
      return None

    end = time.time_ns()
    lats.append((end - start) / 1_000_000)

  # Close the connection.
  local_con.close()

  # And return.
  assert ret is not None
  return np.mean(np.asarray(lats[1:]))

def measure_reconstruction_latency_per_column(parquet_type, parquet_path, schema, all_target_columns, target_iter):
  bench_col = None
  if parquet_type == 'base':
    bench_col = f"\"{target_iter['target_name']}\""
  elif parquet_type == 'virtual':
    # Open a temporary connection.
    # This is used to read the header of the parquet file.
    local_con = duckdb.connect(database=':memory:', read_only=False)
    parquet_header = utils._get_parquet_header(local_con, parquet_path)
    local_con.close()

    # Generate the formula.
    _, raw_formula = generate_formula(schema, parquet_header, target_iter, all_target_columns, enforce_replacement=True)
    bench_col = raw_formula
  else:
    # Type `parquet_type` not supported.
    assert 0
    
  assert bench_col is not None
  return _measure_impl(bench_col, parquet_path)