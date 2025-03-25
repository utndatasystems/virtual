import pandas as pd
import numpy as np
import pathlib
import utils
from typing import Optional

# Data wrapper
from data_handler import DataWrapper

MAX_MSE_ALLOWED = 5

from sklearn.linear_model import LinearRegression
from models.sparse_model import SparseLR
from models.k_regression import K_Regression, k_regression_settings, compute_error
from sklearn.metrics import root_mean_squared_error

def virtualize_table(data: pd.DataFrame | pathlib.Path | utils.URLPath, nrows=None, sample_size=None, model_types: Optional[list[str]]=None):
  assert isinstance(data, (pd.DataFrame, pathlib.Path, utils.URLPath))
  
  # Instantiate the data wrapper.
  data_wrapper = DataWrapper(data, nrows)

  # Inspect the columns that we support. This also sets the valid column names and indices.
  data_wrapper.inspect_columns()

  # The functions found.
  results = []

  # Solve the numerical case.
  if len(data_wrapper.v_cols['num']['names']) > 1:
    results.extend(solve_num_cols(data_wrapper, sample_size=sample_size, model_types=model_types))

  # Solve the non-numerical cases.
  for category in ['date', 'timestamp', 'time']:
    if len(data_wrapper.v_cols[category]['names']) > 1:
      results.extend(solve_custom_cols(data_wrapper, category, sample_size=sample_size))

  print(f'We found {len(results)} function candidate(s) in your table.')
  return results

def solve_custom_cols(data_wrapper, category, sample_size=None):
  results = []
  for idx1, target_index in enumerate(data_wrapper.v_cols[category]['indices']):
    # Note: We use this index to reduce the number of custom models we evaluate.
    custom_idx = 0
    for idx2, ref_index in enumerate(data_wrapper.v_cols[category]['indices']):
      if ref_index == target_index:
        continue

      results.append({
        'category' : category,
        'target-index' : target_index,
        'target-name': data_wrapper.v_cols[category]['names'][idx1],
        'models' : {
          f'custom-{custom_idx}' : {
            'mse' : 0,
            'intercept' : 0,
            'coeffs' : [
              {
                'col-index': ref_index,
                'col-name': data_wrapper.v_cols[category]['names'][idx2],
                'coeff' : 1.0
              }
            ]
          }
        }
      })

      # Increase the custom index.
      custom_idx += 1
  return results

def run_model(model_type, X, y, col_name=None):
  model, y_pred = None, None
  if model_type == 'lr':
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
  elif model_type == 'sparse-lr':
    model = SparseLR(max_mse_allowed=MAX_MSE_ALLOWED).fit(X, y)
    y_pred = model.predict(X)
  elif model_type == 'k-regression':
    k_models = []
    for num_groups, num_iterations in k_regression_settings.items():
      model = K_Regression(num_groups=num_groups, iterations=num_iterations, max_mse_allowed=MAX_MSE_ALLOWED)

      # NOTE: This is done on the sample.
      # Similarly, the score is computed on the sample.      
      best_model_config = model.fit(X, y)

      # Recompute the membership.
      membership = model.expectation(X=X, y=y, models=best_model_config)

      global_err = 0
      valid_models = []
      for i in range(len(best_model_config)):
        indices = np.where(membership == i)[0]
        if len(indices) == 0:
          continue

        # Take this model since it has some membership.
        valid_models.append(best_model_config[i])

        # Compute the error.
        local_err = compute_error(best_model_config[i], X[indices], y[indices])
        global_err += local_err

      # Update the global error.
      global_err /= X.shape[0]

      # Register the model.
      k_models.append({
        'mse' : global_err,
        'models' : valid_models
      })
    return k_models

  assert model is not None
  mse = root_mean_squared_error(y_true=y, y_pred=y_pred)
  return mse, model.intercept_, model.coef_

def solve_num_cols(data_wrapper, sample_size=None, model_types=None):
  # Analyze the coefficients of the regression.
  def reduce_coeffs(coeffs, input_columns):
    selected = []
    for index, coeff in enumerate(coeffs):
      if abs(coeff) < 1e-6:
        continue
      selected.append({
        'col-index' : input_columns[index],
        'col-name' : data_wrapper.column_names[input_columns[index]],
        'coeff' : coeff,
      })
    return selected

  # Unwrap a configuration for a given `k`.
  def unwrap_k_regression_config(config):
    ret = []
    for model in config['models']:
      reduced_coeffs = reduce_coeffs(model.coef_, input_columns)

      # NOTE: Even if there are no selected coefficients, we still have to store the intercept.
      # This means that the column is composed of many constants.
      # See for instances column `improvement_surcharge` in Taxi: it only takes values in {-0.3, 0.3}.
      ret.append({
        'intercept' : model.intercept_,
        'coeffs' : reduced_coeffs
      })
    return ret

  # TODO: We can try multiple samples. This would also be helpful for k-regression.
  sample = data_wrapper.sample('num', sample_size=sample_size)

  # Convert to numpy.
  sample = sample.to_numpy()

  # Try each numeric column.
  results = []
  for target_index in data_wrapper.v_cols['num']['indices']:
    input_columns = data_wrapper.v_cols['num']['indices'].copy()
    input_columns.remove(target_index)

    # Define the regression data.
    X = sample[:, data_wrapper.get_rank('num', input_columns)]
    y = sample[:, data_wrapper.get_rank('num', target_index)]

    # Init the local results.
    local_results = None
    try:
      local_results = {
        'category' : 'num',
        'target-index': target_index,
        'target-name': data_wrapper.column_names[target_index],
        'target-stats': {
          'mean': y.mean(),
          'max': float(y.max()),
          'min': float(y.min())
        },
        'models': {}
      }
    except Exception as e:
      print(e)
      assert False, f"Please double check your file. There is a type/format inconsistency error on column {data_wrapper.column_names[target_index]}."
    assert local_results is not None

    # Try the standard models.
    # NOTE: We later try `k-regression` as well.
    for model_type in ['sparse-lr']:
      # Check if this model is allowed.
      # NOTE: If `model_types` is not None, we actually want to run this model!
      if model_types is not None:
        if model_type not in model_types:
          continue

      # Run the model.
      err, intercept, coeffs = run_model(model_type, X, y, data_wrapper.column_names[target_index])

      # Do we exceed the max. allowed mse?
      # TODO: We'll probably have problems with this one.
      if err > MAX_MSE_ALLOWED:
        continue

      # Reduce the coefficients.
      reduced_coeffs = reduce_coeffs(coeffs, input_columns)

      # NOTE: Even if there are no selected coefficients, we still have to store the intercept.
      # This means that the column is simply constant.
      local_results['models'][model_type] = {
        'mse' : err,
        'intercept' : intercept,
        'coeffs' : reduced_coeffs
      }

    # Try k-regression, only if it's actually allowed.
    if model_types is None or 'k-regression' in model_types:
      # Run the model.
      k_config = run_model('k-regression', X, y, data_wrapper.column_names[target_index])

      # Analyze the result found.
      # It could be that the actual _valid_ models are fewer than the group size we ran for.
      seen_ks = set()
      for iter in k_config:
        actual_k = len(iter['models'])

        # Skip, since we anyway do sparse LR.
        if actual_k == 1:
          continue

        # Do we exceed the max. allowed mse?
        if iter['mse'] > MAX_MSE_ALLOWED:
          continue

        # Already seen and better _or_ new?
        if (actual_k in seen_ks and iter['mse'] < local_results['models'][f'{actual_k}-regression']['mse']) or (actual_k not in seen_ks): 
          local_results['models'][f'{actual_k}-regression'] = {
            'mse' : iter['mse'],
            'config' : unwrap_k_regression_config(iter)
          }
        
        # Mark it as seen anyway.
        seen_ks.add(actual_k)
    
    # Only put once.
    if local_results['models']:
      results.append(local_results)

  # And return.
  return results