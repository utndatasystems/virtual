from sklearn.linear_model import LinearRegression, Ridge
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from models.sparse_model import SparseLR
from sklearn.metrics import root_mean_squared_error

# from sklearnex import patch_sklearn
# patch_sklearn()

# (group_size, num_iterations).
k_regression_settings = {2: 10, 4: 10, 8: 10, 16: 10}

# Define which model to use
# MODEL_TYPE = 'linear'
MODEL_TYPE = 'sparse'
# MODEL_TYPE = 'ridge'

def BUILD_MODEL(max_mse_allowed=None):
    if MODEL_TYPE == 'ridge':
        return Ridge(alpha=10.0)
    elif MODEL_TYPE == 'linear':
        return LinearRegression()
    elif MODEL_TYPE == 'sparse':
        assert max_mse_allowed is not None
        return SparseLR(max_mse_allowed)
    else:
        assert 0

def compute_error(model, X, y):
    return root_mean_squared_error(y_true=y, y_pred=model.predict(X)) * X.shape[0]

class K_Regression:
    def __init__(self, num_groups, iterations, max_mse_allowed):
        self.num_groups = num_groups
        self.iterations = iterations
        self.max_mse_allowed = max_mse_allowed

    def initialize(self, X, y):
        permutation = np.random.permutation(X.shape[0])
        X_groups = np.array_split(X[permutation], self.num_groups)
        y_groups = np.array_split(y[permutation], self.num_groups)

        models = [BUILD_MODEL(self.max_mse_allowed).fit(X_groups[i], y_groups[i]) for i in range(self.num_groups)]
        errs = [compute_error(models[i], X_groups[i], y_groups[i]) for i in range(self.num_groups)]
        return models, sum(errs) / X.shape[0]

    def expectation(self, X, y, models):
        # TODO: But is this really the distance to the line?
        distance_matrix = np.asarray([np.abs(models[i].predict(X) - y) for i in range(len(models))])

        # Take the min.
        return np.argmin(distance_matrix, axis=0)

    def maximisation(self, X, y, membership, curr_num_groups):
        models = []

        # TODO: Compute the error and take the best shape.
        errs = []
        for i in range(curr_num_groups):
            # Take the indices.
            indices = np.where(membership == i)[0]

            # Skip if none.
            if indices.size == 0:
                continue

            # Set the data.
            y_i = y[indices]
            X_i = X[indices]

            # Build.
            reg = BUILD_MODEL(self.max_mse_allowed).fit(X_i, y_i)

            # Calculate the weighted error for this slice.
            errs.append(compute_error(reg, X_i, y_i))

            # And append the model.
            models.append(reg)
        
        # Return the weighted error.
        return models, sum(errs) / X.shape[0]

    def fit(self, X, y):
        # Init.
        models, best_err = self.initialize(X=X, y=y)

        best_models = models.copy()
        errs = []
        x = 0
        while True:
            # If there's absolutely no chance to decrease.
            if x == 3 and errs[-1] > 10.0:
                return best_models
            
            # Check the number of iterations.
            x += 1
            if x > self.iterations or self.iterations > 50:
                return best_models
            
            # Expectation.
            membership = self.expectation(X=X, y=y, models=models)

            # Maximization.
            models, curr_err = self.maximisation(X=X, y=y, membership=membership, curr_num_groups=len(models))

            # Append the error.
            errs.append(curr_err)

            # Just stop.
            if curr_err < 1e-4:
                return models

            if curr_err < best_err:
                best_err = curr_err
                best_models = models.copy()
            else:
                # It got worse -> stuck.
                # TODO: Maybe redo????
                return best_models

            # Should we give it a last chance?
            if x == self.iterations - 1 and errs[-1] < 1e-1:
                if errs[-2] - errs[-1] > 1e-3:
                    self.iterations += 5
        return best_models

    # def visualise(self, X, y, models, y_cutoff):
    #     membership = self.expectation(X=X, y=y, models=models)
    #     index = np.where(y < y_cutoff)[0]
    #     X_subset = X[index]
    #     y_subset = y[index]
    #     membership_subset = membership[index]

    #     clrs = sns.color_palette('tab10', n_colors=len(models))

    #     for i in range(self.num_groups):
    #         indices = np.where(membership_subset == i)[0]
    #         y_i = y_subset[indices]
    #         X_i = X_subset[indices].reshape(-1, 1)
    #         plt.scatter(X_i, y_i, color=clrs[i], s=5, alpha=0.3)
    #         plt.plot(X, models[i].predict(X), color=clrs[i])
    #     plt.show()

    def storage_cost(self, X, y, models, precision,path_parquet):
        membership = self.expectation(X=X, y=y, models=models)
        residuals = np.zeros_like(y, dtype=float)

        for i in range(self.num_groups):
            indices = np.where(membership == i)[0]
            if len(indices) == 0:
                continue
            y_i = y[indices]
            X_i = X[indices]
            pred = models[i].predict(X_i)

            res = np.round(pred, decimals=precision) - y_i

            # TODO: This is not so correct, since we first need to cast the prediction.
            if precision == 0:
                res = (pred - y_i).astype(int)

            residuals[indices] = res

        if (membership == 0).all():
            pa_table = pa.table({"offset": residuals.flatten()})
        else:
            pa_table = pa.table({"offset": residuals.flatten(), "reg_line": membership.flatten()})

        pq.write_table(pa_table, path_parquet)
        file_size = os.stat(path_parquet).st_size
        os.remove(path_parquet)
        return file_size, residuals, membership

def storage_cost(df, file_name):
    storage = {}
    # data=np.arange(0, df.shape[0])
    for series_name, series in df.items():

        pa_table = pa.table({"ref": series})
        pa.parquet.write_table(pa_table, file_name)
        file_size = os.stat(file_name).st_size
        storage[series_name] = file_size
        os.remove(file_name)
    return storage
