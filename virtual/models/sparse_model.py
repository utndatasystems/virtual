from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np

# from sklearnex import patch_sklearn
# patch_sklearn()

class SparseLRInference:
  def __init__(self, intercept, coeffs):
    self.intercept_ = intercept
    self.coef_ = np.asarray(coeffs)
    pass

  def predict(self, x):
    return np.dot(x, self.coef_) + self.intercept_

class SparseLR:
  def __init__(self, max_mse_allowed):
    self.max_mse_allowed = max_mse_allowed
    pass     

  def fit(self, x, y):
    # The initial model.
    init_model = LinearRegression().fit(x, y)
    
    # The prediction.
    y_pred = init_model.predict(x)

    # Compute the original MSE.
    original_mse = root_mean_squared_error(y_true=y, y_pred=y_pred)

    if original_mse > self.max_mse_allowed:
      return init_model

    # Take the non-zero features only.
    selected = []
    for index, coeff in enumerate(init_model.coef_):
      if abs(coeff) < 1e-6:
        continue
      selected.append(index)

    # Is the column constant?
    if not selected:
      return SparseLRInference(init_model.intercept_, [0] * x.shape[1])

    # Fit the first LR model.
    test_model = LinearRegression().fit(x[:, selected], y)
    test_y_pred = test_model.predict(x[:, selected])
    test_mse = root_mean_squared_error(y_true=y, y_pred=test_y_pred)

    # Is it worse? Then take the original model.
    # TODO: Sometimes this actually happens.
    if test_mse > self.max_mse_allowed:
      return init_model
    
    # If there is one attribute, then we don't have anything to remove.
    if len(selected) == 1:
      return init_model

    all_mses = []
    for feature_index in selected:
      # Fit
      new_selected = [index for index in selected if index != feature_index]
      curr_model = LinearRegression().fit(x[:, new_selected], y)
      curr_y_pred = curr_model.predict(x[:, new_selected])
      curr_mse = root_mean_squared_error(y_true=y, y_pred=curr_y_pred)

      # Collect the MSEs.
      all_mses.append({
        'feature' : feature_index,
        'mse w/o' : curr_mse,
      })
    
    # Sort by the amplitude (in descending order!).
    all_mses = sorted(all_mses, key=lambda elem: +elem['mse w/o'], reverse=True)

    # Now gradually insert.
    temp_selected = []
    for index in range(len(all_mses)):
      temp_selected.append(all_mses[index]['feature'])
      curr_model = LinearRegression().fit(x[:, temp_selected], y)
      curr_y_pred = curr_model.predict(x[:, temp_selected])
      curr_mse = root_mean_squared_error(y_true=y, y_pred=curr_y_pred)
      all_mses[index]['mse'] = curr_mse

    # And decide when to stop.
    selected = []
    last = all_mses[-1]['mse']
    for index in range(len(all_mses)):
      # Add the feature.
      selected.append(all_mses[index]['feature'])

      # Already perfect MSE?
      if all_mses[index]['mse'] < 1e-4:
        break

      # Already dropped below a sanity threshold?
      if all_mses[index]['mse'] < last + 0.5:
        break

    # Fit again.
    best_model = LinearRegression().fit(x[:, selected], y)
    best_y_pred = best_model.predict(x[:, selected])
    # best_mse = root_mean_squared_error(y_true=y, y_pred=best_y_pred)

    best_coeffs = []
    index_in_selected = 0
    for i in range(x.shape[1]):
      if i in selected:
        best_coeffs.append(best_model.coef_[index_in_selected])
        index_in_selected += 1
      else:
        best_coeffs.append(0.0)
    return SparseLRInference(best_model.intercept_, best_coeffs)