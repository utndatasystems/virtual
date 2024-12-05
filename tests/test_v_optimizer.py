from virtual.v_optimizer import optimize
import pytest
from virtual.utils import ModelType, _read_json

@pytest.fixture
def model_types():
  return list(map(ModelType, ['sparse-lr']))

@pytest.fixture
def sample_data1():
  # Generic example.
  return _read_json('tests/sample_data1.json')

@pytest.fixture
def sample_data2():
  # Counter-example for the greedy algorithm.
  return _read_json('tests/sample_data2.json')

def test_optimize1(sample_data1, model_types):
  ret = optimize(sample_data1, model_types)
 
  # We only take 2 functions.
  assert ret['greedy']['obj-value'] == 2

def test_optimize2(sample_data2, model_types):
  ret = optimize(sample_data2, model_types)
 
  # TODO: This could be better by using the optimal algorithm.
  # We only take 2 functions.
  assert ret['greedy']['obj-value'] == 1.5
