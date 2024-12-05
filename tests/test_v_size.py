from virtual.v_size import create_virtual_column_layout, _create_regression
import pytest
from virtual.utils import ModelType, _read_json
from .utils import extract_functions

@pytest.fixture
def model_types():
  return list(map(ModelType, ['sparse-lr']))

@pytest.fixture
def sample_data1():
  return _read_json('tests/sample_data1.json')

@pytest.fixture
def schema1():
  return _read_json('tests/schema1.json')

def test_create_regression(sample_data1, schema1, model_types):
  fns = extract_functions(sample_data1, model_types[0])
  for index, iter in enumerate(sample_data1):
    ret = _create_regression(
      iter['models']['sparse-lr'],
      schema=schema1,
      target_name=iter['target_name']
    )
    other_cols = fns[index].split('=')[1].strip()
    assert ret == f'"{other_cols}"'

def test_create_virtual_column_layout(sample_data1, schema1, model_types):
  for _, iter in enumerate(sample_data1):
    ret = create_virtual_column_layout(
      con=None,
      target_iter=iter,
      schema=schema1,
      model_type=model_types[0]
    )
    col_name = iter['target_name']

    # Create the regression.
    tmp = _create_regression(
      iter['models']['sparse-lr'],
      schema=schema1,
      target_name=col_name
    )
    assert ret == [f'"{col_name}_offset" = coalesce(round("{col_name}" - ({tmp}), 0), 0)']