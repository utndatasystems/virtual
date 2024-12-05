import pandas as pd
import numpy as np
import pathlib
from typing import Optional

# Helper scripts
from data_handler import DataParser

MAX_MSE_ALLOWED = 5

from sklearn.linear_model import LinearRegression
from models.sparse_model import SparseLR
from models.k_regression import K_Regression, k_regression_settings, compute_error
from sklearn.metrics import root_mean_squared_error

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

def virtualize_table(data: pd.DataFrame | pathlib.Path, nrows=None, sample_size=None, allowed_model_types: Optional[list[str]]=None):
  assert isinstance(data, (pd.DataFrame, pathlib.Path))
  
  # Parse data.
  parser = DataParser(data, nrows)
  df = parser.parse()

  # Select the valid column indices.
  valid_column_indices = parser.compute_valid_column_indices(df)

  # Analzye the coefficients of the regression.
  def reduce_coeffs(coeffs, valid_column_indices):
    selected = []
    for index, coeff in enumerate(coeffs):
      if abs(coeff) < 1e-6:
        continue
      selected.append({
        'col_index' : valid_column_indices[index],
        'col_name' : parser.column_names[valid_column_indices[index]],
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

  results = []
  if len(valid_column_indices) <= 1:
    print('Only one numerical column.')
    return {}
  
  # TODO: We can try multiple samples.
  sample = parser.extract_sample(df, sample_size=sample_size)
  
  # Convert to numpy.
  df = df.to_numpy()
  sample = sample.to_numpy()

  debug_path = ''
  if isinstance(data, pathlib.Path):
    debug_path = str(data.name)

  # Try each column.
  for target_index in valid_column_indices:
    input_columns = valid_column_indices.copy()
    input_columns.remove(target_index)

    # TODO: Maybe try multiple samples.
    # TODO: This would also be helpful for k-regression.
    X = sample[:, input_columns]
    y = sample[:, target_index]

    # Init the local results.
    local_results = None
    try:
      local_results = {
        'target_index': target_index,
        'target_name': parser.column_names[target_index],
        'target_stats': {
          'mean': y.mean(),
          'max': float(y.max()),
          'min': float(y.min())
        },
        'models': {}
      }
    except Exception as e:
      assert False, f"Please double check your CSV file. There is a type/format inconsistency error on column {parser.column_names[target_index]}."
    assert local_results is not None

    # Try the standard models.
    for model_type in ['sparse-lr']:
      # Check if this model is allowed.
      # NOTE: If `allowed_model_types` is not None, we actually want to run this model!
      if allowed_model_types is not None:
        if model_type not in allowed_model_types:
          continue

      # Run the model.
      err, intercept, coeffs = run_model(model_type, X, y, parser.column_names[target_index])

      # Do we exceed the max. allowed mse?
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
    if allowed_model_types is None or 'k-regression' in allowed_model_types:
      # Run the model.
      k_config = run_model('k-regression', X, y, parser.column_names[target_index])

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

  print(f'We found {len(results)} function(s) in your table.')
  return results