from virtual.interface import train, to_format, query, from_format
import pandas as pd
import numpy as np
from .utils import extract_functions, compare_fns
import os
import duckdb

def sum_data():
  a = np.random.randint(0, 10, size=1000)
  b = np.random.randint(0, 10, size=1000)
  c = a + b
  return pd.DataFrame({
    'a': a,
    'b': b,
    'c': c
  })

def get_wide_df():
  a = np.random.randint(0, 10, size=1000)
  b = np.random.randint(0, 10, size=1000)
  c = np.random.randint(0, 10, size=1000)
  z = np.random.randint(0, 10, size=1000)
  x = np.random.randint(0, 10, size=1000)

  d = a + b + c
  e = d + x
  return pd.DataFrame({
    'a': a,
    'b': b,
    'c': c,
    'd' : d,
    'e' : e,
    'x' : x,
    'z' : z
  })

def get_model_types():
  return ['sparse-lr']

def test_train_sum1():
  # TODO: Fix this one.
  model_types = get_model_types()
  data = sum_data()
  ret = train(data, model_types=model_types)
  fns = extract_functions(ret, model_type='sparse-lr')
  gt = ['a = c + -b', 'c = b + a', 'b = -a + c']
  assert compare_fns(fns, gt)

def test_train_sum2():
  # TODO: Fix this one.
  model_types = get_model_types()
  data = get_wide_df()
  ret = train(data, model_types=model_types)
  fns = extract_functions(ret, model_type='sparse-lr')
  gt = ['d = a + b + c', 'e = d + x']
  assert compare_fns(fns, gt)

def test_query_sum():
  model_types = get_model_types()
  data = sum_data()
  ret = train(data, model_types=model_types)

  # Save.
  to_format(data, 'test_query_sum.parquet', functions=ret)

  # Check the queries.
  for col in ['a', 'b', 'c']:
    ret = query(f"select sum({col}) as test from read_parquet('test_query_sum.parquet');")
    act = duckdb.query(f'select sum({col}) as test from data').fetchdf()
    assert ret['test'].dtype == act['test'].dtype
    assert list(ret['test'])[0] == list(act['test'])[0]

  # Remove the file
  os.remove('test_query_sum.parquet')

def test_from_format():
  model_types = get_model_types()
  data = sum_data()
  ret = train(data, model_types=model_types)

  # Save.
  to_format(data, 'test_query_sum.parquet', functions=ret)

  df = from_format('test_query_sum.parquet')
  assert df == data

  # Remove the file
  os.remove('test_query_sum.parquet')

def test_for_hf_url():
  # Test for csv url
  csv_url = 'hf://datasets/fka/awesome-chatgpt-prompts/prompts.csv'
  to_format(csv_url, 'test_hf.parquet')
  df = from_format('test_hf.parquet')

  # Remove the file
  os.remove('test_hf.parquet')

  # Test for parquet url
  parquet_url = 'hf://datasets/simplescaling/s1K/data/train-00000-of-00001.parquet'
  to_format(csv_url, 'test_hf.parquet')
  df = from_format('test_hf.parquet')

  # Remove the file
  os.remove('test_hf.parquet')

