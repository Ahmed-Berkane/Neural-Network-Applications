# tests/test_model.py
import numpy as np

from models import NeuralNetworkLogistic
from sklearn.linear_model import LogisticRegression

def test_nn_shapes():
    # Dummy data: 5 features, 10 samples
    X = np.random.rand(5, 10)  # (n_features, m_samples)
    Y = np.random.randint(0, 2, size=(1, 10))  # (1, m_samples)
    
    nn = NeuralNetworkLogistic(learning_rate=0.01, max_iter=10, print_cost=False)
    nn.fit(X, Y)
    
    Y_pred = nn.predict(X)
    
    assert Y_pred.shape == Y.shape, "Prediction shape mismatch"
    assert ((Y_pred == 0) | (Y_pred == 1)).all(), "Predictions should be 0 or 1"

def test_sklearn_logistic():
    # Dummy data: 20 samples, 10 features
    X = np.random.rand(20, 10)  # (n_samples, n_features)
    y = np.random.randint(0, 2, size=20)
    
    clf = LogisticRegression(max_iter=100)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    assert y_pred.shape == y.shape, "Sklearn prediction shape mismatch"
    assert ((y_pred == 0) | (y_pred == 1)).all(), "Predictions should be 0 or 1"
