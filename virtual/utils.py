import pandas as pd
import numpy as np
import duckdb
import time
import json
import os
import re

# Our utils
import schema_inference
import btrblocks_utils

# Other utils
import pyarrow.parquet as pq
from typing import List
import pathlib
import hashlib
import csv

def to_csv(parquet_path, format_path):
  df = pd.read_parquet(parquet_path)
  df.to_csv(format_path, index=False)
  _delete_file(parquet_path)
  return format_path

def to_btrblocks(parquet_path, format_path, schema=None):
  tmp_path = format_path + '.csv'
  df = pd.read_parquet(parquet_path)
  schema = schema_inference.schema_utils.generate_schema(df)
  df.to_csv(tmp_path, index=False)
  btrblocks_utils.convert_csv_to_btrblocks(tmp_path, format_path, schema)
  os.remove(tmp_path)
  _delete_file(parquet_path)
  return format_path

def custom_round(x):
  # TODO: Maybe -- if the weight is almost an integer, convert it?
  return round(x, 4)

def build_query(select_stmt, input, header, delim=',', quotechar='"', nullstr=[], sample_size=None):
  return f"""
    select {select_stmt}
    from read_csv(
      '{input}',
      header={str(header).lower()},
      delim='{delim}',
      quote='{quotechar}',
      nullstr={nullstr}""" \
    + (f", sample_size={sample_size}" if sample_size is not None else ", sample_size=-1") \
    + ")"

def get_csv_null_options():
  return "['NULL', 'null', '', 'None']"

# Get a hash for this data.
def get_data_hash(data: pd.DataFrame | pathlib.Path):
  assert isinstance(data, (pd.DataFrame, pathlib.Path))

  # Take a fingerprint of the data.
  if isinstance(data, pd.DataFrame):
    # The column names.
    column_string = ','.join(data.columns)
  elif isinstance(data, pathlib.Path):
    # The path itself.
    column_string = str(data.resolve())
  else:
    raise TypeError("Input must be a pandas `DataFrame` or a `pathlib.Path`.")

  # And return the hash.
  return hashlib.md5(column_string.encode('utf-8')).hexdigest()

def infer_dialect(file_path, num_lines=5):
  with open(file_path, 'r', encoding='utf-8') as file:
    lines = [file.readline() for _ in range(num_lines)]

    # Detect delimiters using csv.Sniffer
    sniffer = csv.Sniffer()
    sample = ''.join(lines)
    try:
      dialect = sniffer.sniff(sample)
      return {
        'delimiter' : dialect.delimiter,
        'quotechar' : dialect.quotechar
      }
    except csv.Error:
      # Fallback.
      return get_default_csv_dialect()
    
def get_default_csv_dialect():
  return {
    'delimiter' : ',',
    'quotechar' : '"'
  }

def get_duckdb_conn():
  return duckdb.connect(database=':memory:', read_only=False)

def rename(src, dst):
  os.rename(src, dst)

# TODO: Maybe also slow.
def add_metadata_pq(parquet_path, info, info_name, new_parquet_path=None):
  table = pq.read_table(parquet_path)
  metadata = table.schema.metadata

  if metadata is None:
    pass
  elif info_name.encode('utf-8') in metadata:
    print('The functions are already present in the metadata.')
    return -1
  functions_str = json.dumps(info)
  custom_metadata = {info_name : functions_str}
  merged_metadata = { **custom_metadata, **(table.schema.metadata or {}) }

  fixed_table = table.replace_schema_metadata(merged_metadata)
  if new_parquet_path is not None:
    pq.write_table(fixed_table, new_parquet_path)
  else:
    pq.write_table(fixed_table, parquet_path)
  return 1

def get_metadata_pq(parquet_path, metadata_types):
  # Read the metadata.
  parquet_file = pq.ParquetFile(parquet_path)
  metadata = parquet_file.schema_arrow.metadata

  # No metadata?
  if metadata is None:
    print('Nothing in parquet metadata.')
    return None

  # Take for each `metadata_type`.
  ret = {}
  for metadata_type in metadata_types:
    ret[metadata_type] = None
    if metadata_type == 'header':
      assert parquet_file.schema_arrow.names is not None
      ret[metadata_type] = parquet_file.schema_arrow.names
    else:
      if metadata_type.encode('utf-8') not in metadata:
        continue
      ret[metadata_type] = json.loads(metadata[metadata_type.encode('utf-8')].decode('utf-8'))
  return ret

def rreplace(text, old, new):
  return new.join(text.rsplit(old, 1))

# Precompile the regex pattern to ensure it ends exactly with "-regression"
k_regression_pattern = re.compile(r'^\d+-regression$')

# Regex for formula, e.g. f0, f1.
formula_pattern = re.compile(r'^f\d+$')

def is_formula(token):
  return bool(formula_pattern.match(token))

class ModelType:
  def __init__(self, name):
    self.name = name

  def is_k_regression(self):
    return bool(k_regression_pattern.match(self.name))

  def extract_k(self):
    assert self.is_k_regression()
    assert len(self.name.split('-')) == 2
    try:
      val = int(self.name.split('-')[0])
      return val
    except Exception as e:
      # This should never happen.
      print(e)
      return None

  def __repr__(self):
    return f"ModelType(name={self.name})"

# Check if the type is an integer.
def is_integer(type):
  return 'int' in type.lower()

# Check if the type is a floting point.
# TODO: Extend with decimal for the future.
def is_double(type):
  if 'decimal' in type.lower():
    assert 0
  return any(x in type.lower() for x in ['double', 'float'])

def is_virtualizable(type):
  return is_integer(type) or is_double(type)

def is_attached_column_of(target_name, column_name):
  # TODO: This was a real hack.
  return column_name.startswith(f'{target_name}_')# or (column_name[1:].startswith(f'{target_name}_')

def collect_lr_column_names(model):
  cols = []
  for iter in model['coeffs']:
    cols.append(iter['col_name'])
  return cols

def _get_column(schema, column_name):
  for iter in schema:
    if iter['name'] == column_name:
      return iter
  return None

def _get_nullable(schema, column_name):
  column = _get_column(schema, column_name)
  return 'null' if column['null']['any'] else 'not null'

def _get_info(schema, column_name):
  column = _get_column(schema, column_name)

  if column['type'] == 'DOUBLE':
    return {'type' : 'DOUBLE', 'scale' : column['scale']}
  return {'type' : column['type'], 'scale' : 0}

def _read_json(json_path):
  # Check if the file exists.
  assert os.path.isfile(json_path)

  # And read the data.
  f = open(json_path, 'r', encoding='utf-8')
  data = json.load(f)
  f.close()

  # Return it.
  return data

def _write_json(json_path, json_content):
  f = open(json_path, 'w', encoding='utf-8')
  json.dump(json_content, f, ensure_ascii=False)
  f.close()

def _delete_file(filepath):
  assert os.path.exists(filepath)
  os.remove(filepath)

def csv_path_to_schema_path(prefix, csv_path):
  basename = os.path.basename(csv_path).replace('.csv', '')
  return f'{prefix}/{basename}.json'

def _get_curr_columns(con):
  return list(set(con.execute("select * from base_table limit 1;").fetchdf().columns))

def _get_size(con):
  return list(con.execute('select count(*) as size from base_table;').fetchdf()["size"])[0]

def _get_parquet_header(parquet_path):
  return pq.ParquetFile(parquet_path).schema_arrow.names

def _get_distinct_counts(con):
  curr_columns = _get_curr_columns(con)
  assert curr_columns
  return con.execute(f'select ' + ','.join([f"(count(distinct \"{x}\") + count(distinct case when \"{x}\" is null then 1 end)) as \"{x}_count\"" for x in curr_columns]) + ' from base_table').fetchdf().to_dict()

def dump_data(csv_filename, data, filename, prefix, key=None):
  csv_basename = os.path.basename(csv_filename)
  csv_basename_noext = os.path.splitext(csv_basename)[0]
  json_filename = os.path.join(prefix, f'{filename}_{csv_basename_noext}.json')
  
  with open(json_filename, 'w', encoding='utf-8') as f:
    if key is not None:
      json.dump(sorted(data, key=key), f, ensure_ascii=False, indent=2)
    else:
      json.dump(data, f, ensure_ascii=False, indent=2)
  return

def dump_data_without_filename(data, filename, prefix, key=None):
  json_filename = os.path.join(prefix, f"{filename}.json")
  
  with open(json_filename, 'w', encoding='utf-8') as f:
    if key is not None:
      json.dump(sorted(data, key=key), f, ensure_ascii=False, indent=2)
    else:
      json.dump(data, f, ensure_ascii=False, indent=2)
  return

def dump_json_data(data, json_data, component_name, prefix=None):
  # Only dump if the user wished so.
  if prefix is None:
    return
  
  # Dump (even if the result is empty).
  # First, ensure that the folder exists.
  prefix_folder = pathlib.Path(prefix)
  prefix_folder.mkdir(parents=True, exist_ok=True)
  
  # And dump.
  if isinstance(data, pathlib.Path):
    dump_data(data, json_data, component_name, prefix)
  elif isinstance(data, pd.DataFrame):
    dump_data_without_filename(json_data, component_name, prefix)
  else:
    assert 0

def gather_all_models(functions):
  # Collect all model names present in the json file.

  all_keys = set()
  for col in functions:
    all_keys |= set(col['models'].keys())
  return all_keys

def select_models(model_types: List[str], functions):
  # TODO: Now we need to change, since we always want a functions type.

  # Collect all model names present in the json file.
  all_keys = gather_all_models(functions)

  # Select only those model names that are also present in the json file.
  validated = []
  for model_type in model_types:
    # Already present?
    if model_type in all_keys:
      validated.append(model_type)
    elif model_type == 'k-regression':
      # Otherwise, we check for `k-regression`, since these have a number in front.
      for possible_match in all_keys:
        # Is it a k-regression?
        if ModelType(possible_match).is_k_regression():
          validated.append(possible_match)

  # Map to `ModelType`.
  return validated

def natural_rounding(coeff):
  return round(coeff, 4)

# Maybe we need to implement this:
# https://stackoverflow.com/questions/3951505/how-do-i-round-a-set-of-numbers-while-ensuring-the-total-adds-to-1
def rounding(combs):
  import copy
  new_combs = copy.deepcopy(combs)

  # Round a linear regression model.
  def round_model(model):
    for col in model['coeffs']:
      curr_coeff = col['coeff']
      new_coeff = natural_rounding(curr_coeff)
      col['coeff'] = new_coeff
    model['intercept'] = natural_rounding(model['intercept'])

  # Round all numbers present in the json file.
  for combination in new_combs:
    for model_type in combination['models']:
      if model_type in ['lr', 'sparse-lr']:
        round_model(combination['models'][model_type])
      elif ModelType(model_type).is_k_regression():
        for local_model in combination['models'][model_type]['config']:
          round_model(local_model)
      else:
        assert 0
  return new_combs

def compute_metrics(table, combs):
  def get_metrics(y1, y2):
    assert y1.shape == y2.shape

    # TODO: This should be the precision from decimal, if any!
    diff = abs(y1 - y2)
    diff = np.array([x if x is not pd.NA else np.nan for x in diff], dtype=float)
    diff = diff[~np.isnan(diff)]
    equal = (diff < 1e-6).mean()
    average = diff.mean()
    maximum = float(diff.max())
    return {'equal' : equal, 'avg_diff' : average, 'max_diff': maximum}

  def estimate_y(model):
    intercept = model['intercept']
    coeffs = np.array([entry['coeff'] for entry in model['coeffs']])
    col_indices = [entry['col_index'] for entry in model['coeffs']]

    # Estimate the y-values.
    y_est = intercept + np.dot(table[:, col_indices], coeffs)

    # And return it.
    return y_est

  def find_closest_estimate(ests, target):
    # Convert 'ests' to a NumPy array if it's not already
    ests = np.array(ests)  

    # Compute the absolute difference between each estimate and the target
    diffs = np.abs(ests - target)  
    
    # Find the index of the minimum difference along axis 0 (for each column)
    argmin = np.argmin(diffs, axis=0)  
    
    # Use fancy indexing to select the closest estimates based on the minimum indices
    return ests[argmin, np.arange(argmin.size)]

  metrics = []
  for comb in combs:
    comb['metrics'] = dict()
    for model_type in comb['models']:
      if model_type in ['lr', 'sparse-lr']:
        y_est = estimate_y(comb['models'][model_type])

        # Get the metrics.
        local_metrics = get_metrics(y_est, table[:, comb['target_index']])
        comb['metrics'][model_type] = local_metrics
      elif ModelType(model_type).is_k_regression():
        local_ests = []
        for local_model in comb['models'][model_type]['config']:
          local_ests.append(estimate_y(local_model))

        # Get the estimate closest to table[:, comb['target_index']].
        closest_est = find_closest_estimate(local_ests, table[:, comb['target_index']])
        local_metrics = get_metrics(closest_est, table[:, comb['target_index']])
        comb['metrics'][model_type] = local_metrics
      else:
        assert 0
    metrics.append(comb)
  return metrics

def compute_avg_stats(parquet_file, config_path, model_type='sparse-lr'):
  # Open the connection.
  con = duckdb.connect(database=':memory:', read_only=False)

  # Read the config.
  config = _read_json(config_path)

  avgs = {}
  for col_config in config['greedy']['chosen']:
    compare_value = list(con.execute(f"select avg(\"{col_config['target_name']}\") as res from read_parquet('{parquet_file}');").fetchdf()["res"])[0]
    avgs[col_config['target_name']] = compare_value
  return avgs

class TimeTracker:
  def __init__(self):
    self.prev_time = time.time_ns()
    self.register = dict()
    pass

  def add(self, name):
    curr_time = time.time_ns()
    time_diff = curr_time - self.prev_time
    self.prev_time = curr_time
    self.register[name] = time_diff / 1_000_000
  
  def log(self):
    sorted_register = sorted(self.register.items(), key=lambda x: -x[1])
    for k, v in sorted_register:
      print(f'{k}: {v} ms')