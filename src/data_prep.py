import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# For the custom Neural Network Logistic Regression (1, m format)
# ----------------------------------------------------------------
def preprocess_for_nn(X, y, test_size = 0.3):
    """
    Prepares data for custom NN logistic regression.
    X: numpy array (n_samples, height, width, 3)
    y: numpy array (n_samples,)
    Returns:
        X_train_flat (n_features, m_train)
        X_test_flat (n_features, m_test)
        y_train (1, m_train)
        y_test (1, m_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Flatten and normalize
    X_train_flat = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).T / 255.0

    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    return X_train_flat, X_test_flat, y_train, y_test

# ----------------------------------------------------------------
# For scikit-learn Logistic Regression (n_samples, n_features)
# ----------------------------------------------------------------
def preprocess_for_sklearn(X, y, test_size = 0.3):
    """
    Prepares data for scikit-learn logistic regression.
    X: numpy array (n_samples, height, width, 3)
    y: numpy array (n_samples,)
    Returns:
        X_train, X_test, y_train, y_test (flattened and normalized)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Flatten and normalize
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(n_train, -1) / 255.0
    X_test_flat = X_test.reshape(n_test, -1) / 255.0

    return X_train_flat, X_test_flat, y_train, y_test
