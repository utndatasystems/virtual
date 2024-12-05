from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import numpy as np
# from celer import Lasso
import scipy

# from sklearnex import patch_sklearn
# patch_sklearn()

class SparseLRInference:
  def __init__(self, intercept, coeffs):
    self.intercept_ = intercept
    self.coef_ = np.asarray(coeffs)
    pass

  def predict(self, x):
    return np.dot(x, self.coef_) + self.intercept_

def apply_lasso(x, y, selected):
  new_model = Lasso().fit(x[:, selected], y)
  new_y_pred = new_model.predict(x[:, selected])
  new_mse = root_mean_squared_error(y_true=y, y_pred=new_y_pred)

  # if new_mse > 1e-6:
    # return None
  return new_model.intercept_, new_model.coef_

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

    # Is it work? Then take the original model.
    # TODO: Sometimes this actually happens.
    if test_mse > original_mse + 1e-6:
      return init_model
    
    # If there is one attribute, then we don't have anything to remove.
    if len(selected) == 1:
      return init_model

    temp_mse = original_mse
    fixed_window_mse = original_mse
    lazy_counter = 0

    # TODO: Maybe we can sort by the error from the previous iteration.
    while len(selected) > 1:
      # Set to false.
      has_removed_feature = False

      # Take each feature.
      for feature_index in selected:
        # Fit
        new_selected = [index for index in selected if index != feature_index]
        curr_model = LinearRegression().fit(x[:, new_selected], y)
        curr_y_pred = curr_model.predict(x[:, new_selected])
        curr_mse = root_mean_squared_error(y_true=y, y_pred=curr_y_pred)

        # Did we improve?
        if curr_mse < temp_mse + 1e-6:
          # Try to avoid sequences of errors that don't lead anywhere.
          lazy_counter += 1
          if lazy_counter == 2:
            # No improve since the last step?
            # Assumption: `fixed_window_mse` is much larger and `temp_mse` decreased compared to it.
            # If the decrease is not that much, try with another attribute.
            if fixed_window_mse - temp_mse < 1e-6:
              # Update the counter.
              lazy_counter -= 1
              continue

            # If the decrease is significantly, update the metadata, as follows:
            # Update the fixed-window mse.
            fixed_window_mse = temp_mse

            # Update the counter.
            lazy_counter = 0

          # Regular path: We remove this feature.
          # Remove this feature.
          selected.remove(feature_index)

          # Mark that we removed.
          has_removed_feature = True

          # Update the error.
          temp_mse = curr_mse

          # Out of the for loop.
          break

      # No feature removed?
      if not has_removed_feature:
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