import numpy as np
import pandas as pd

class NeuralNetworkLogistic:
    def __init__(self, learning_rate, tolerance=1e-6, max_iter=10000, print_cost=False):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.print_cost = print_cost
        self.w = None
        self.b = None
        self.costs = []

    def fit(self, X, Y):
        n_features, m = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0.

        prev_cost = float("inf")
        for i in range(self.max_iter):
            z = np.dot(self.w.T, X) + self.b
            A = 1 / (1 + np.exp(-z))

            epsilon = 1e-15
            cost = -(np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))) / m

            dw = np.dot(X, (A - Y).T) / m
            db = np.sum(A - Y) / m

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if abs(prev_cost - cost) < self.tolerance:
                if self.print_cost:
                    print(f"Converged at iteration {i}, cost={cost:.6f}")
                break

            prev_cost = cost
            if i % 100 == 0:
                self.costs.append(cost)
                if self.print_cost:
                    print(f"Cost after {i}: {cost:.6f}")
        return self

    def predict_proba(self, X):
        z = np.dot(self.w.T, X) + self.b
        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    # --- Metrics ---
    def confusion_matrix(self, X, Y, threshold=0.5):
        Y_pred = self.predict(X, threshold).flatten()
        Y_true = Y.flatten()
        TP = np.sum((Y_true == 1) & (Y_pred == 1))
        TN = np.sum((Y_true == 0) & (Y_pred == 0))
        FP = np.sum((Y_true == 0) & (Y_pred == 1))
        FN = np.sum((Y_true == 1) & (Y_pred == 0))

        columns = pd.MultiIndex.from_product([["Predicted"], ["0", "1"]])
        index = pd.MultiIndex.from_product([["Actual"], ["0", "1"]])
        return pd.DataFrame([[TN, FP], [FN, TP]], index=index, columns=columns)

    def accuracy(self, X, Y, threshold=0.5):
        return np.mean(self.predict(X, threshold).flatten() == Y.flatten())

    def precision(self, X, Y, threshold=0.5):
        Y_pred = self.predict(X, threshold).flatten()
        Y_true = Y.flatten()
        TP = np.sum((Y_true == 1) & (Y_pred == 1))
        FP = np.sum((Y_true == 0) & (Y_pred == 1))
        return TP / (TP + FP + 1e-15)

    def recall(self, X, Y, threshold=0.5):
        Y_pred = self.predict(X, threshold).flatten()
        Y_true = Y.flatten()
        TP = np.sum((Y_true == 1) & (Y_pred == 1))
        FN = np.sum((Y_true == 1) & (Y_pred == 0))
        return TP / (TP + FN + 1e-15)

    def f1(self, X, Y, threshold=0.5):
        prec = self.precision(X, Y, threshold)
        rec = self.recall(X, Y, threshold)
        return 2 * (prec * rec) / (prec + rec + 1e-15)
