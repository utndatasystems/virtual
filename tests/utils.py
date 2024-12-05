import math
from virtual.utils import ModelType, custom_round

def get_function(fn):
  intercept = fn['intercept']
  regression = []

  # Round the intercept.
  intercept = custom_round(float(intercept))
  if not math.isclose(intercept, 0.0):
    regression.append(str(intercept))

  for col_info in fn['coeffs']:
    col_name = col_info['col_name']
    col_coeff = col_info['coeff']
    if math.isclose(col_coeff, 0.0):
      continue
    if math.isclose(col_coeff, 1.0):
      regression.append(f'{col_name}')
    elif math.isclose(col_coeff, -1.0):
      regression.append(f'-{col_name}')
    else:
      regression.append(f'{custom_round(float(col_coeff))} * {col_name}')

  return ' + '.join(regression)

def extract_functions(fns, model_type):
  this_model = model_type
  if type(model_type) == ModelType:
    this_model = model_type.name

  ret = []
  for fn in fns:
    ret.append(
      f"{fn['target_name']} = {get_function(fn['models'][this_model])}"
    )
  return ret

def get_tokens(fn):
  tokens = fn.split(' ')
  ret = []
  for token in tokens:
    token = token.strip()
    if token != '+':
      ret.append(token)
  return sorted(ret)

def split_fn(fn):
  assert len(fn.split('=')) == 2
  return fn.split('=')[0].strip(), fn.split('=')[1].strip()

def compare_fn(fn1, fn2):
  l1, r1 = split_fn(fn1)
  l2, r2 = split_fn(fn2)
  if l1 != l2:
    return False
  return get_tokens(r1) == get_tokens(r2)

def compare_fns(fns1, fns2):
  if len(fns1) != len(fns2):
    return False
  taken = [False] * len(fns2)
  for fn1 in fns1:
    has_been_taken = False
    for index, fn2 in enumerate(fns2):
      if compare_fn(fn1, fn2):
        if taken[index]:
          return False
        has_been_taken = True
        taken[index] = True
    if not has_been_taken:
      return False 
  return True