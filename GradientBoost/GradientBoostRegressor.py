import numpy as np

from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor


class GradientBoostRegressor:
    def __init__(self, n_learners=20, max_depth=3, loss='mse'):
        self.base_learner = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=8)
        self.n_learners = n_learners
        self.regrs = [deepcopy(self.base_learner) for _ in range(self.n_learners)]
        self.gammas = [[] for _ in range(self.n_learners)]
        self.leaves_output = [[] for _ in range(self.n_learners)]
        self.compute_loss = self.mse_loss if loss == 'mse' else loss

    def init_odd_pred(self, y):
        self.odd_pred = lambda x: np.ones(x.shape[0]) * y.mean()

    def mse_loss(self, y_true, y_pred):
        loss = ((y_true - y_pred) ** 2) / 2
        grad = y_true - y_pred
        return loss, grad

    def get_indices(self, regr, X):
        leaves = regr.apply(X)
        n_leaves = np.max(leaves)
        leaves_index = [np.where(leaves == leaf_num)[0] for leaf_num in range(1, 1 + n_leaves)]
        return leaves_index

    def find_optimal_coefs(self, X, y, y_pred, regr, t, leaves_index):
        for leaf_index in leaves_index:
            if leaf_index.size == 0:
                self.gammas[t].append(0)
                continue

            leaf_losses = []
            for i in np.arange(0.0, 10, 0.01):
                leaf_losses.append(
                    self.compute_loss(
                        y[leaf_index],
                        y_pred[leaf_index] + i * regr.predict(X[leaf_index])
                    )[0].mean()
                )
            leaf_gamma = np.arange(0.0, 10, 0.01)[np.argmin(leaf_losses)]
            self.gammas[t].append(leaf_gamma)

    def fit(self, X, y):
        self.init_odd_pred(y)
        y_pred = self.odd_pred(X)
        for t in range(self.n_learners):
            residuals = self.compute_loss(y, y_pred)[1]
            self.regrs[t].fit(X, residuals)
            leaves_index = self.get_indices(self.regrs[t], X)
            self.leaves_output[t] = [
                residuals[leaf_index].mean() if leaf_index.size != 0 else 0 for leaf_index in leaves_index
            ]
            self.find_optimal_coefs(X, y, y_pred, self.regrs[t], t, leaves_index)
            leaves = self.regrs[t].apply(X) - 1
            pred = [
                self.leaves_output[t][leaf_num] * self.gammas[t][leaf_num] for leaf_num in leaves
            ]
            y_pred += pred

    def predict(self, X):
        y_pred = self.odd_pred(X)
        for t in range(self.n_learners):
            leaves = self.regrs[t].apply(X) - 1
            pred = [
                self.leaves_output[t][leaf_num] * self.gammas[t][leaf_num] for leaf_num in leaves
            ]
            y_pred += pred
        return y_pred
