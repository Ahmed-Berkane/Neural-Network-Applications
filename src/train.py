
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from utils import load_and_save_data
from data_prep import preprocess_for_nn, preprocess_for_sklearn
from models import NeuralNetworkLogistic




# ----------------------------
# Step 1: Load dataset
# ----------------------------
DATA_PATH = "data/pizza_not_pizza"
X_raw, y_raw = load_and_save_data(DATA_PATH, image_size=(64, 64))

# ----------------------------
# Step 2: Preprocess for custom NN
# ----------------------------
# Returns X_train (n_features, m), y_train (1, m)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = preprocess_for_nn(X_raw, 
                                                                 y_raw, 
                                                                 test_size = 0.3)


# ----------------------------
# Step 3: Train Custom NN Logistic
# ----------------------------
nn_model = NeuralNetworkLogistic(learning_rate = 0.01)
nn_model.fit(X_train_nn, y_train_nn)

# Printing Train and Test metrics
print(f"NN Logistic Train Confusion Matrix:\n{nn_model.confusion_matrix(X_train_nn, y_train_nn)}")
print(f"NN Logistic Train Accuracy: {nn_model.accuracy(X_train_nn, y_train_nn)}")
print(f"NN Logistic Train precision: {nn_model.precision(X_train_nn, y_train_nn)}")
print(f"NN Logistic Train recall: {nn_model.recall(X_train_nn, y_train_nn)}")
print(f"NN Logistic Train f1: {nn_model.f1(X_train_nn, y_train_nn)}")

print(f"NN Test Confusion Matrix:\n{nn_model.confusion_matrix(X_test_nn, y_test_nn)}")
print(f"NN Logistic Test Accuracy: {nn_model.accuracy(X_test_nn, y_test_nn)}")
print(f"NN Logistic Test precision: {nn_model.precision(X_test_nn, y_test_nn)}")
print(f"NN Logistic Test recall: {nn_model.recall(X_test_nn, y_test_nn)}")
print(f"NN Logistic Test f1: {nn_model.f1(X_test_nn, y_test_nn)}")


# Save NN model
joblib.dump(nn_model, "models/nn_logistic.pkl")



# ----------------------------
# Step 4: Preprocess for sklearn Logistic Regression
# ----------------------------
# Returns X_train, X_test (n_samples, n_features)
X_train_skl, X_test_skl, y_train_skl, y_test_skl = preprocess_for_sklearn(X_raw, y_raw, test_size=0.3)

# ----------------------------
# Step 5: Train sklearn Logistic Regression
# ----------------------------
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_skl, y_train_skl)

# Predictions
y_test_pred = clf.predict(X_test_skl)
y_train_pred = clf.predict(X_train_skl)

# Printing Train and Test metrics
print(f"Sklearn Logistic Train Accuracy: {(y_train_pred == y_train_skl).mean()}")
print(f'Sklearn Logistic Train: {classification_report(y_train_skl, y_train_pred)}')

print(f"Sklearn Logistic Test Accuracy: {(y_test_pred == y_test_skl).mean()}")
print(f'Sklearn Logistic Test: {classification_report(y_test_skl, y_test_pred)}')

# Save sklearn model
joblib.dump(clf, "models/sklearn_logistic.pkl")