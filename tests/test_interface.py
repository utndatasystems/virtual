from virtual.interface import train, to_format, query
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

def get_model_types():
  return ['sparse-lr']

def test_train_sum():
  model_types = get_model_types()
  data = sum_data()
  ret = train(data, model_types=model_types)
  fns = extract_functions(ret, model_type='sparse-lr')
  gt = ['a = c + -b', 'c = b + a', 'b = -a + c']
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
