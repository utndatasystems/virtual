import pathlib
import typing
import sys
import os

# Set the path for the modules to load.
base_path = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(base_path))

# The system components.
import v_driller
import v_optimizer
import v_compressor
import v_reconstructor

# Other utils.
from typing import Optional
import schema_inference
import pandas as pd
import duckdb
import utils
import time
import re

# TODO: We should add `nrows`. Maybe also the sample_size.
def train(data: pd.DataFrame | pathlib.Path | utils.URLPath | str, nrows=None, sample_size=10_000, model_types: Optional[list[str]]=['sparse-lr', 'custom'], schema=None, prefix=None):
  """
    Computes the functions hidden in `data`.

    Args:
      data (pd.DataFrame | pathlib.Path | str): The data source.
      nrows (int): The number of rows to take.
      sample_size (int): The number of rows to train on.
      model_types (list[str]): The models that the user wants to explore. If `None`, then we try all our models (for better file sizes).

      [prefix: The directory which to save the results to; helpful for debugging purposes.]

    Example (1):
      df = pd.read_csv('file.csv')
      functions = virtual.train(df)
  """

  # Convert into an actual `Path`.
  if isinstance(data, str):
    if data.startswith('hf://'):
      data = pd.read_csv(data)
    elif data.startswith('s3://'):
      data = utils.URLPath(data)
    else:
      data = pathlib.Path(data)
  assert isinstance(data, (pd.DataFrame, pathlib.Path, utils.URLPath))

  # Check that we're dealing with a file.
  if isinstance(data, pathlib.Path):
    if not str(data).startswith('s3:'):
      assert data.is_file()

  print(f'Drilling functions..')

  # Virtualize the table. Note that for this particular step we don't require a schema.
  functions = v_driller.virtualize_table(data, nrows=nrows, sample_size=sample_size, model_types=model_types)

  # Dump `functions`.
  utils.dump_json_data(data, functions, 'driller', prefix)

  # And return them.
  return functions

# TODO: Check whether all cases for `nrows` work out.
def to_format(data: pd.DataFrame | pathlib.Path | utils.URLPath | str, format_path, functions=None, schema=None, nrows=None, model_types: Optional[list[str]]=['sparse-lr', 'custom'], prefix=None):
  """
    Converts `data` into a file of the specified format.

    Args:
      data (pd.DataFrame | pathlib.Path | str): The data source.
      format_path (str): The file which to save to; will be created if it doesn't exist.
      functions (dict): The functions to be used during compression. If `None`, then `virtual` automatically computes them.
      schema (dict): The schema of the data. If `None`, then `virtual` automatically infers it.
      nrows (int): The #rows to save to Parquet. If `None`, the _entire_ file will be processed.
      model_types (list[str]): The models that the user wants to explore. If `None`, then all are tried out.

      [prefix: The directory which to save the results to; helpful for debugging purposes.]

    Example (1):
      df = pd.read_csv('file.csv')
      virtual.to_format(df, 'file.parquet')

    Example (2):
      virtual.to_format('file.csv', 'file.parquet')
  """

  # Check the format path.
  assert '.' in format_path, 'The `format_path` is invalid.'
  format_type = os.path.splitext(format_path)[-1]
  format_type = format_type[1:]
  assert format_type, 'The `format_path` is invalid!'
  if format_type not in ['csv', 'parquet', 'btrblocks']:
    assert False, f'Format {format_type[1:]} not yet supported.'

  # Convert into an actual `Path`.
  if isinstance(data, str):
    if data.startswith('hf://'):
      # TODO: We need to check whether the data is actually `.csv` or `.parquet`.
      data = pd.read_csv(data)
    elif data.startswith('s3://'):
      if format_type == 'parquet':
        # Set as URL.
        data = utils.URLPath(data)
      else:
        assert 0, f'Reading this format from S3 is not yet supported.'
    elif data.startswith('https://'):
      if format_type == 'parquet':
        # Set as URL.
        data = utils.URLPath(data)
      else:
        assert 0, f'Reading this format from HTTPS is not yet supported.'  
    else:
      data = pathlib.Path(data)
  assert isinstance(data, (pd.DataFrame, pathlib.Path, utils.URLPath))

  # Check that we're dealing with a file.
  if isinstance(data, pathlib.Path):
    assert data.is_file()
  if isinstance(data, utils.URLPath):
    assert data.suffix in ['.parquet', '.csv']

  # No schema? Then generate it.
  if schema is None:
    schema = schema_inference.schema_utils.generate_schema(data, nrows)
  elif isinstance(schema, str):
    assert os.path.isfile(schema), f'The schema path you provided is unfortunately not valid.'
    schema = utils._read_json(schema)

  # Dump `schema`.
  utils.dump_json_data(data, schema, 'schema', prefix)

  # If no functions provided, drill them.
  if functions is None:
    functions = train(data, prefix=prefix, model_types=model_types, schema=schema)

  # Dump `functions`.
  utils.dump_json_data(data, functions, 'driller', prefix)

  # If not custom model types specified, take'em all.
  if model_types is None:
    model_types = utils.gather_all_models(functions)
  else:
    model_types = utils.select_models(model_types, functions)
  model_types = list(map(utils.ModelType, model_types))

  print(f'Let\'s see how many benefit virtualization..')

  # Estimate the column sizes.
  estimated_sizes = v_optimizer.compute_target_sizes(data, functions, schema, model_types)

  # Dump `estimated_sizes`.
  utils.dump_json_data(data, estimated_sizes, 'estimates', prefix)

  # Optimize the layout of the virtual table based on the previous estimates.
  functions = v_optimizer.optimize(estimated_sizes, model_types)

  # Dump `layout`.
  utils.dump_json_data(data, functions, 'layout', prefix)

  print(f"It seems that {len(functions['greedy']['chosen'])} function(s) can indeed be used for virtualization.")

  # Compress the file via the optimized layout.
  virtual_path = v_compressor.compress(data, schema, functions, optimize_layout=True, nrows=nrows)

  # Insert the functions into the metadata. They're later used when we want to decompress / query the file.
  if format_type == 'parquet':
    utils.add_metadata_pq(virtual_path, functions, 'functions')

    # Insert the schema into the metadata. TODO: Add when it's used.
    utils.add_metadata_pq(virtual_path, schema, 'schema')

    # Rename with the wished path, `format_path`.
    utils.rename(virtual_path, format_path)
  elif format_type == 'csv':
    # TODO: @Ping.
    utils.to_csv(virtual_path, format_path)
    folder_path = os.path.dirname(os.path.abspath(format_path))
    base_name = os.path.splitext(os.path.basename(format_path))[0]
    utils._write_json(os.path.join(folder_path, f'{base_name}-functions.json'), functions)
    utils._write_json(os.path.join(folder_path, f'{base_name}-schema.json'), schema)
  elif format_type == 'btrblocks':
    # TODO: @Ping: `btrblocks`.
    utils.to_btrblocks(virtual_path, format_path)
    utils._write_json(os.path.join(format_path, 'functions.json'), functions)
    utils._write_json(os.path.join(format_path, 'schema.json'), schema)

  print(f'Done.')
  return

def from_format(format_path, functions=None, schema=None):
  # TODO: @Ping.
  assert '.' in format_path, "The `format_path` is invalid!"
  format_type = os.path.splitext(format_path)[-1]
  assert format_type[1:], "The `format_path` is invalid!"
  if format_type[1:] not in ['csv', 'parquet']:
    assert False, f"Format {format_type[1:]} not yet supported!"

  # Get the data from the format and also check the metadata (functions and schema).
  if format_type[1:] == 'parquet':
    if schema is None:
      schema = utils.get_metadata_pq(format_path, ['schema'])['schema']
    # df = pd.read_parquet(format_path)
    # con = utils.get_duckdb_conn()
    # header = utils._get_parquet_header(format_path)
    pass
  elif format_type[1:] == 'csv':
    folder_path = os.path.dirname(os.path.abspath(format_path))
    base_name = os.path.splitext(os.path.basename(format_path))[0]
    if schema is None:
      schema = utils._read_json(os.path.join(folder_path, f'{base_name}-schema.json'))
    # df = pd.read_csv(format_path)
    pass
  elif format_type[1:] == 'btrblocks':
    if schema is None:
      schema = utils._read_json(os.path.join(format_path, 'schema.json'))
    pass
  else:
    assert 0, f'Format {format_type[1:]} not (yet) supported.'

  # Create the SELECT clause.
  select_clause = ','.join([f'"{row['name']}"' for row in schema])

  if format_type[1:] == 'parquet':
    sql_query = f'select {select_clause} from read_parquet("{format_path}");'
  elif format_type[1:] == 'csv':
    sql_query = f'select {select_clause} from read_csv("{format_path}");'

  # Query.
  df = query(sql_query, functions=functions, schema=schema, engine='duckdb')
  
  # And return.
  return df

def query(sql_query, functions=None, schema=None, engine='duckdb', run=True, fancy=True, return_execution_time=False):
  """
    Query the Parquet file via SQL.

    Args:
      sql_query (str): The SQL string. It should contain a `read_parquet(...)` call, so that we call the engine.
      functions (dict): The functions to be used during query. If `None`, then `virtual` recovers them from the Parquet's metadata.
      schema (dict): The schema of the data. If `None`, then `virtual` recovers it from Parquet's metadata.
      engine (str): The query engine to use. Options: ['duckdb', 'pandas'].
      run (bool): If we should really run the query. Otherwise, we simply return the rewritten query as result.
      fancy (bool): If we should also keep the original column names of the result.
      return_execution_time (float): Also returns the execution time of the query (if `run` is set).

    Example (1):
      virtual.query(
        'select avg(price) from read_parquet("file.parquet") where year >= 2024',
        engine = 'duckdb'
      )
  """

  # Extract the parquet path. Supports both type of quotes.
  match = re.search(r'(read_parquet|read_csv)\((["\'])(.*?)\2\)', sql_query)

  # Fallback for queries where there is no read from a file.
  if match is None:
    # No need to run?
    if not run:
      return sql_query

    start_time = time.time_ns()
    ans = duckdb.sql(sql_query).fetchdf()
    stop_time = time.time_ns()

    # Should we also return the execution time?
    if return_execution_time:
      return ans, stop_time - start_time
    return ans

  # Infer the format type and the file path.
  format_type = match.group(1).replace('read_', '') if match else None
  file_path = match.group(3) if match else None

  # Check if we support this format.
  if format_type not in ['parquet', 'csv']:
    assert False, f'Format {format_type} not yet supported!'

  # Get the metadata, along with the header.
  # Note: We rely on the fact the variable names remain `schema` and `functions`.
  if format_type == 'parquet':
    format_metadata = get_metadata_from_file(format_type, file_path, ['header'] + [key for key in ['schema', 'functions'] if locals().get(key) is None])

  # Read the schema.
  if schema is None:
    if format_type == 'parquet':
      assert format_metadata is not None and format_metadata['schema'] is not None, 'Your virtualized Parquet file doesn\'t have the necessary metadata.'
      schema = format_metadata['schema']
    elif format_type == 'csv':
      folder_path = os.path.dirname(os.path.abspath(file_path))
      base_name = os.path.splitext(os.path.basename(file_path))[0]
      schema = utils._read_json(os.path.join(folder_path, f'{base_name}-schema.json'))
    else:
      assert 0, f'File format {format_type} not supported yet.'
  elif isinstance(schema, str):
    schema = utils._read_json(schema)
  assert schema is not None

  # Read the functions.
  if functions is None:
    if format_type == 'parquet':
      assert format_metadata is not None and format_metadata['functions'], 'Your virtualized Parquet file doesn\'t have the necessary metadata.'
      functions = format_metadata['functions']
    elif format_type == 'csv':
      folder_path = os.path.dirname(os.path.abspath(file_path))
      base_name = os.path.splitext(os.path.basename(file_path))[0]
      functions = utils._read_json(os.path.join(folder_path, f'{base_name}-functions.json'))
    else:
      assert 0, f'File format {format_type} not supported yet.'
  elif isinstance(functions, str):
    functions = utils._read_json(functions)
  assert functions is not None

  # Check the query engine is valid.
  assert engine in ['duckdb', 'pandas']

  # Workaround to support `pandas`.
  if engine == 'pandas':
    if format_type == 'parquet':
      df = pd.read_parquet(file_path)
      sql_query.replace(f"read_parquet('{file_path}')", "df")
    elif format_type == 'csv':
      # TODO: @Ping.
      df = pd.read_csv(file_path)
      sql_query.replace(f"read_csv('{file_path}')", "df")
      pass

  # Fetch the parquet header.
  # TODO: Maybe combine with the metadata reading to be faster.
  if format_type == 'parquet':
    assert format_metadata is not None, 'Your virtualized Parquet file doesn\'t have the necessary metadata.'
    header = format_metadata['header']
  if format_type == 'csv':
    header = utils.get_csv_header(file_path)

  # TODO: Update the naming convention in the layout file to avoid this `greedy` and `chosen`.
  matches = []

  def add_match(m, raw_formula):
    # TODO: What about unicodes? The length might be different for case insensitive letters?
    matches.append((m.start(), m.start() + len(m.group()), m.group(), f'({raw_formula})'))

  for iter in functions['greedy']['chosen']:
    # Generate the formula.
    _, raw_formula = v_reconstructor.generate_formula(schema, header, iter, functions['greedy']['chosen'], enforce_replacement=True)

    pattern = re.compile(rf"\"?\b{re.escape(iter['target-name'])}\b\"?")

    # NOTE: If we have double quotes, then we *must* enforce case sensitiveness.
    for m in pattern.finditer(sql_query, re.IGNORECASE):
      # Double-quote?
      if m.group()[0].startswith('"'):
        # Should be the same, but without the double quotes (if any).
        # TODO: Do we also replace with the double-quoted column names?
        if m.group().strip('"') == iter['target-name'].strip('"'):
          add_match(m, raw_formula)
      else:
        add_match(m, raw_formula)

  # Now comes the most clever part:
  # Since we don't want to purely replace the attributes naively (since we could replace multiple times),
  # we sort the starting positions in *decreasing* order and start replacing backwards,
  # enforcing that we search until the last token replaced. Brrr.
        
  sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)
  for start, end, before, after in sorted_matches:
    sql_query = sql_query[:start] + after + sql_query[end:]

  # Set the rewritten query.
  rewritten_sql_query: typing.Final[str] = sql_query

  # No need to run? Then simply return the query.
  if not run:
    return rewritten_sql_query

  # Execute.
  start_time = time.time_ns()
  ans = duckdb.sql(rewritten_sql_query).fetchdf()
  stop_time = time.time_ns()

  # Rename the columns to maintain consistency with the default output by pandas.
  if fancy:
    # Compile the patterns.
    replacements = [(re.compile(re.escape(after)), before) for _, _, before, after in sorted_matches]

    # Update the names.
    col_dict = { col: re.sub(rep[0], rep[1], col) for col in ans.columns for rep in replacements }

    # And rename the columns.
    ans = ans.rename(columns=col_dict)

  # And return.
  if return_execution_time:
    return ans, stop_time - start_time
  return ans

def functions_to_json(functions, json_path):
  utils._write_json(json_path, functions)

def get_functions_from_json(json_path):
  return utils._read_json(json_path)

def get_metadata_from_file(format_type, file_path, metadata_types):
  if format_type == 'parquet':
    return utils.get_metadata_pq(file_path, metadata_types)
  elif format_type == 'csv':
    # TODO: @Ping.
    pass
  elif format_type == 'btrblocks':
    # TODO: @Ping.
    pass
  assert 0, f'File format {format_type} not supported yet.'