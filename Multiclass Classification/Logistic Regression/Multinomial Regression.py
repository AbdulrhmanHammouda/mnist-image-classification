import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. MULTINOMIAL LOGISTIC REGRESSION (SOFTMAX)
# ==========================================

def softmax(Z):
    """
    Numerically stable softmax:
    Subtract row-max before exponentiating to prevent overflow.
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def calculate_gradient(W, X, Y_onehot):
    """
    Gradient of the cross-entropy loss for softmax regression.

    Parameters:
        W       : weight matrix, shape (features+1, 10)
        X       : input with bias column, shape (N, features+1)
        Y_onehot: one-hot encoded labels, shape (N, 10)

    Returns:
        grad    : gradient matrix, shape (features+1, 10)
    """
    N = X.shape[0]
    Z = X @ W              # (N, 10)
    Y_hat = softmax(Z)     # (N, 10)
    error = Y_hat - Y_onehot  # (N, 10)
    grad = (1.0 / N) * (X.T @ error)  # (features+1, 10)
    return grad

def gradient_descent(X, Y_onehot, eta=1e-2, iterations=5000, tol=1e-7, lam=0.0):
    """
    Gradient descent for multinomial logistic regression.
    W(n+1) = W(n) - eta * (gradient + lam * W(n))
    """
    n_classes = Y_onehot.shape[1]
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # prepend bias column
    W = np.zeros((X_b.shape[1], n_classes))

    for i in range(iterations):
        grad = calculate_gradient(W, X_b, Y_onehot)
        W = W - eta * (grad + lam * W)  # Regularized update

        if np.linalg.norm(grad) < tol:
            print(f"    Converged at iteration {i+1}.")
            break

        # Print progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            # Compute regularized loss for monitoring
            Z = X_b @ W
            Y_hat = softmax(Z)
            ce_loss = -np.mean(np.sum(Y_onehot * np.log(Y_hat + 1e-12), axis=1))
            reg_loss = (lam / 2) * np.sum(W ** 2)
            loss = ce_loss + reg_loss
            print(f"    Iteration {i+1}/{iterations} | Loss: {loss:.4f}")

    return W

def predict(X, W):
    """
    Predict class labels (0-9) using argmax over softmax probabilities.
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    probs = softmax(X_b @ W)
    return np.argmax(probs, axis=1)


# ==========================================
# 2. DATA PREP (MNIST 10-CLASS)
# ==========================================

print("Fetching MNIST dataset... (this may take a minute)")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

print("Preparing and splitting data...")
y = y.astype(int)  # Convert string labels to integers 0-9

# One-hot encode the labels
Y_onehot_full = np.eye(10)[y]

# FIRST SPLIT: Isolate 10,000 samples for the final Test Set
X_temp, X_test, y_temp, y_test, Y_oh_temp, Y_oh_test = train_test_split(
    X, y, Y_onehot_full, test_size=10000, random_state=42
)

# SECOND SPLIT: Isolate 10,000 samples for the Validation Set
X_train, X_val, y_train, y_val, Y_oh_train, Y_oh_val = train_test_split(
    X_temp, y_temp, Y_oh_temp, test_size=10000, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
# Fit strictly on training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 3. HYPERPARAMETER TUNING (GRID SEARCH)
# ==========================================

# Define the hyperparameters to test
pca_components_to_test = [50, 100, 200]
learning_rates_to_test = [1e-2, 1e-1]
lambdas_to_test = [0.1, 1.0]
iterations_per_test = 5000

# Trackers for the best model
best_val_accuracy = 0.0
best_params = {}
best_W = None
best_pca = None

print("\nStarting automated hyperparameter tuning...")
print("==========================================")

for n_components in pca_components_to_test:
    for eta in learning_rates_to_test:
      for lam in lambdas_to_test:
        print(f"Testing -> PCA: {n_components} | eta: {eta} | lambda: {lam}")

        # Apply PCA (Must be re-applied for each n_components test)
        pca = PCA(n_components=n_components, random_state=42)
        X_train_features = pca.fit_transform(X_train_scaled)
        X_val_features = pca.transform(X_val_scaled)

        # Train model
        W_hat = gradient_descent(
            X_train_features,
            Y_oh_train,
            eta=eta,
            iterations=iterations_per_test,
            lam=lam
        )

        # Evaluate on Validation Set
        y_pred_val = predict(X_val_features, W_hat)
        val_acc = accuracy_score(y_val, y_pred_val)

        print(f"  -> Validation Accuracy: {val_acc * 100:.2f}%\n")

        # Check if this is the best model so far
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_params = {'n_components': n_components, 'eta': eta, 'lam': lam}
            best_W = W_hat
            best_pca = pca

print("==========================================")
print("TUNING COMPLETE!")
print(f"Best Validation Accuracy : {best_val_accuracy * 100:.2f}%")
print(f"Best Parameters Found    : PCA = {best_params['n_components']}, eta = {best_params['eta']}, lambda = {best_params['lam']}")
print("==========================================")


# ==========================================
# 4. FINAL EVALUATION (ON TEST SET)
# ==========================================

print("\nEvaluating winning model on the unseen Test Set...")

# Transform the test set using the best PCA transformer found during tuning
X_test_features = best_pca.transform(X_test_scaled)
X_train_features_best = best_pca.transform(X_train_scaled)  # For train accuracy comparison

# Predict using the best weights
y_pred_train = predict(X_train_features_best, best_W)
y_pred_test = predict(X_test_features, best_W)

# Calculate final metrics
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

precision_macro = precision_score(y_test, y_pred_test, average='macro')
recall_macro = recall_score(y_test, y_pred_test, average='macro')
f1_macro = f1_score(y_test, y_pred_test, average='macro')

precision_weighted = precision_score(y_test, y_pred_test, average='weighted')
recall_weighted = recall_score(y_test, y_pred_test, average='weighted')
f1_weighted = f1_score(y_test, y_pred_test, average='weighted')

cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(10))


# ==========================================
# 5. RESULTS
# ==========================================

print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")

print(f"\n--- Macro Averaged Metrics ---")
print(f"Precision (macro) : {precision_macro * 100:.2f}%")
print(f"Recall    (macro) : {recall_macro * 100:.2f}%")
print(f"F1-Score  (macro) : {f1_macro * 100:.2f}%")

print(f"\n--- Weighted Averaged Metrics ---")
print(f"Precision (weighted) : {precision_weighted * 100:.2f}%")
print(f"Recall    (weighted) : {recall_weighted * 100:.2f}%")
print(f"F1-Score  (weighted) : {f1_weighted * 100:.2f}%")

print("\n--- Per-Class Classification Report ---")
print(classification_report(y_test, y_pred_test, digits=4))

print("\nConfusion Matrix (Test Set)")
print("Rows = Actual, Columns = Predicted")
print("Classes: 0  1  2  3  4  5  6  7  8  9")
print(cm)
