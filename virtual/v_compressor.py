from v_size import \
  _size_impl, \
  _create_base_table, \
  create_copy_csv, \
  create_virtual_table_layout, \
  create_dump_base, \
  create_dump_virtual, \
  debug_base_table, \
  arrow_compress_parquet, \
  reoptimize_virtual_table_layout
import duckdb
import utils
import os
import pandas as pd
import schema_inference
from schema_inference import schema_utils
from v_reconstructor import measure_reconstruction_latency_per_column
import pathlib

def _apply_arrow(tmp_path, final_path):
  arrow_compress_parquet(tmp_path, final_path)

  # Delete the duckdb parquet file.
  utils._delete_file(tmp_path)

  # And return the final path.
  return final_path

def _compress_impl(con, fn, data: pd.DataFrame | pathlib.Path, schema, target_columns, type, apply_arrow=True, data_hash=None):
  assert isinstance(data, (pd.DataFrame, pathlib.Path))
  assert data_hash is not None

  # Fix the paths.
  tmp_path = f'.{type}_duckdb-{data_hash}.parquet'
  final_path = f'.{type}_arrow-{data_hash}.parquet'

  # Flush from duckdb.
  con.execute(fn(con, schema, target_columns, tmp_path))

  # No Arrow to apply?
  if not apply_arrow:
    return {
      'tmp_path' : tmp_path,
      'final_path' : final_path
    }

  # Compress via arrow for better compression ratios.
  arrow_compress_parquet(tmp_path, final_path)

  # Delete the duckdb parquet file.
  utils._delete_file(tmp_path)

  # And return.
  return final_path

# This is used to avoid duplicated code in `compress` and `benchmark`.
def _load_data(data: pd.DataFrame | pathlib.Path, schema, layout, nrows):
  # Open the connection.
  con = duckdb.connect(database=':memory:', read_only=False)

  # Select the target columns the layout has chosen.
  target_columns = layout['greedy']['chosen']

  # Create the base table.
  con.execute(_create_base_table(target_columns, schema))

  # Load the csv.
  con.execute(create_copy_csv(data, schema, nrows=nrows))

  # Return the connection.
  return con, target_columns

def compress(data: pd.DataFrame | pathlib.Path, schema, layout, optimize_layout=True, nrows=None, enforce_base=False):
  # Get the data hash.
  data_hash = utils.get_data_hash(data)
  
  # Read the schema.
  if isinstance(schema, pathlib.Path):
    schema = utils._read_json(schema)
  elif schema is None:
    schema = schema_utils.generate_schema(data)

  # Load the data.
  con, target_columns = _load_data(data, schema, layout, nrows)

  # Create the offset and outlier columns.
  create_virtual_table_layout(con, target_columns, schema)

  # Re-optimize the constant auxiliary columns.
  if optimize_layout:
    reoptimize_virtual_table_layout(con, target_columns)

  debug_base_table(con, 'after virtual table layout')

  # And compress.
  base_path = None
  if enforce_base:
    base_path = _compress_impl(con, create_dump_base, data, schema, target_columns, 'base', apply_arrow=True, data_hash=data_hash)
  virtual_path = _compress_impl(con, create_dump_virtual, data, schema, target_columns, 'virtual', apply_arrow=True, data_hash=data_hash)

  # Close the connection.
  con.close()

  # Return the paths, depending on what `enforce_base` says.
  if enforce_base:
    assert base_path is not None
    return [base_path, virtual_path]
  return virtual_path

def _latency_impl(parquet_type, parquet_path, schema, all_target_columns, target_iter):
  assert 'model_type' in target_iter
  model_type = utils.ModelType(target_iter['model_type'])

  # Add the sizes into the dictionary.
  if 'latencies' not in target_iter:
    target_iter['latencies'] = dict()
  
  # NOTE: This is _not_ an `elif`!
  if model_type.name not in target_iter['latencies']:
    target_iter['latencies'][model_type.name] = dict()

  # Run.
  latency = measure_reconstruction_latency_per_column(parquet_type, parquet_path, schema, all_target_columns, target_iter)

  # And add them.
  target_iter['latencies'][model_type.name][parquet_type] = latency
  pass

def benchmark(bench_type, data: pd.DataFrame | pathlib.Path, schema, layout, optimize_layout=True, nrows=None):
  assert bench_type in ['size', 'latency']

  # Get the data hash.
  data_hash = utils.get_data_hash(data)

  # Read the schema.
  if isinstance(schema, pathlib.Path):
    schema = utils._read_json(schema)
  elif schema is None:
    schema = schema_utils.generate_schema(data)

  # Load the data.
  con, target_columns = _load_data(data, schema, layout, nrows)

  # If we want to measure the latency, duplicate the dataset until we have 1M rows.
  if bench_type == 'latency':
    # Double the size until you reach 1M.
    curr_size = utils._get_size(con)
    while 2 * curr_size <= 1_000_000:
      con.execute(f'insert into base_table select * from base_table;')
      curr_size = utils._get_size(con)

    # Insert until 1M rows if we didn't reach the limit yet.
    if curr_size < 1_000_000:
      con.execute(f'insert into base_table select * from base_table limit {1_000_000 - curr_size};')

    print(f'Loaded with repetition: {utils._get_size(con)}')

  # Finalize the layout. This works for both simple and k-regression.
  # NOTE: The `model_type` is also `None` in this case.
  create_virtual_table_layout(con, target_columns, schema)

  # Re-optimize the constant auxiliary columns.
  if optimize_layout:
    reoptimize_virtual_table_layout(con, target_columns)

  # Compute the base and virtual sizes.
  if bench_type == 'size':
    for target_column in target_columns:
      _size_impl(con, schema, target_column, data_hash=data_hash)

  path_mapping = {
    'base' : _compress_impl(con, create_dump_base, data, schema, target_columns, 'base', apply_arrow=False, data_hash=data_hash),
    'virtual' : _compress_impl(con, create_dump_virtual, data, schema, target_columns, 'virtual', apply_arrow=False, data_hash=data_hash)
  }

  # Close the connection. NOTE: We do this trick just to avoid having the table active in the connection.
  # From now on, we won't need the connection anymore.
  con.close()

  # Now apply arrow.
  for key in path_mapping:
    _apply_arrow(path_mapping[key]['tmp_path'], path_mapping[key]['final_path'])
    path_mapping[key] = path_mapping[key]['final_path']

  # Calculate the sizes.
  base_size = os.path.getsize(path_mapping['base'])
  virtual_size = os.path.getsize(path_mapping['virtual'])

  # Compute the base and virtual sizes.
  for target_column in target_columns:
    # Measure the latencies.
    for parquet_type in ['base', 'virtual']:
      _latency_impl(parquet_type, path_mapping[parquet_type], schema, target_columns, target_column)
        
  # Delete the paths.
  for parquet_type in path_mapping:
    utils._delete_file(path_mapping[parquet_type])
 
  # Close the connection.
  # NOTE: Should have been already closed.
  # con.close()

  # And return the sizes.
  return {
    'base_size' : base_size,
    'virtual_size' : virtual_size,
    'cols' : target_columns
  }