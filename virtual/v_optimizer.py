from utils import ModelType
from typing import List
import pandas as pd
import pathlib
import v_size
import sys

def compute_target_sizes(data: pd.DataFrame | pathlib.Path, functions, schema, model_types: List[ModelType], sample_size=10_000):
  assert schema is not None
  assert isinstance(data, (pd.DataFrame, pathlib.Path))

  # Compute the gains for the target columns.
  target_sizes = v_size.compute_sizes_of_target_columns(functions, data, schema, model_types, sample_size)

  # Error?
  if target_sizes is None:
    assert 0

  # And return.
  return target_sizes

def collect_refs(model, model_type: ModelType):
  ref_idxs, ref_names = [], []
  if model_type.is_k_regression():
    for local_model in model['config']:
      for coeff in local_model['coeffs']:
        if coeff['col_index'] not in ref_idxs:
          assert coeff['col_name'] not in ref_names
          ref_idxs.append(coeff['col_index'])
          ref_names.append(coeff['col_name'])
  else:  
    for coeff in model['coeffs']:
      ref_idxs.append(coeff['col_index'])
      ref_names.append(coeff['col_name'])
  return ref_idxs, ref_names

def conflict(e1, e2):
  if e1['right'] == e2['right']:
    return True
  
  # Cyclic: But now only if the left side of `e2` has only one element.
  if e1['right'] in e2['left'] and e2['right'] in e1['left']:
    # TODO: Wait, this was wrong.
    return True
  return False

def run_greedy(hyperedges):
  # Sort in decreasing order.
  sorted_hyperedges = sorted(hyperedges, key=lambda x: -x['gain'])

  chosen = []

  def is_transitive(he1, he2):
    if he1['right'] in he2['left']:
      return True
    return False

  def combine(he1, he2):
    assert is_transitive(he1, he2)
    return {
      'left' : he1['left'],
      'right' : he2['right'],
      'is_fake' : True
    }

  def merge(he):
    chosen.append(he)
    for i in range(len(chosen)):
      side = 0
      test = 0
      if is_transitive(he, chosen[i]):
        test += 1
        side = -1
      if is_transitive(chosen[i], he):
        test += 1
        side = +1

      assert test <= 1
      if side == -1:
        chosen.append(combine(he, chosen[i]))
      elif side == +1:
        chosen.append(combine(chosen[i], he))

  total_gain = 0
  for i in range(len(sorted_hyperedges)):
    flag = True
    for chosen_he in chosen:
      if conflict(sorted_hyperedges[i], chosen_he):
        flag = False
    if flag:
      total_gain += sorted_hyperedges[i]['gain']
      merge(sorted_hyperedges[i])
      # chosen.append(sorted_hyperedges[i])

  final_chosen = []
  for i in range(len(chosen)):
    if 'is_fake' not in chosen[i]:
      final_chosen.append(chosen[i])
  return total_gain, final_chosen

def compute_space_gain(hyperedge, config, model_type):
  hyperedge['gain'] = config['sizes'][model_type.name]['base'] - config['sizes'][model_type.name]['virtual']

def optimize(configs, model_types: List[ModelType]):
  # Inspect the configs and represent them as hyperedges.
  # That is, the left side of the hyperedge is the set of reference columns.
  # The right side is the target column.
  hyperedges = []
  for config in configs:
    for model_type in model_types:
      if model_type.name not in config['models']:
        continue
      target_index = config['target_index']
      target_name = config['target_name']
      model = config['models'][model_type.name]

      curr_col_idxs, _ = collect_refs(model, model_type)

      hyperedges.append({
        'target_index' : target_index,
        'target_name' : target_name,
        'model_type' : model_type.name,
        model_type.name : model,
        'left' : curr_col_idxs,
        'right' : target_index,
      })

      compute_space_gain(hyperedges[-1], config, model_type)
      if hyperedges[-1]['gain'] <= 0:
        hyperedges = hyperedges[:-1]

  import time
  cp1 = time.time()
  # obj_value1, chosen1 = run_optimal(hyperedges)
  cp2 = time.time()
  obj_value2, chosen2 = run_greedy(hyperedges)
  cp3 = time.time()

  # print(f'Optimal: total_gain={obj_value1}, #corrs={len(chosen1)}/{len(hyperedges)}, time={cp2 - cp1}s')
  # print(f'Greedy: total_gain={obj_value2}, #corrs={len(chosen2)}/{len(hyperedges)}, time={cp3 - cp2}s')

  return {
    # 'optimal' : {
    #   'obj-value' : obj_value1,
    #   'chosen' : chosen1,
    #   'time' : cp2 - cp1
    # },
    'greedy' : {
      'obj-value' : obj_value2,
      'chosen' : chosen2,
      'time' : cp3 - cp2
    }
  }