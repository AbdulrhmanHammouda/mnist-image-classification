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
    confusion_matrix
)

# ==========================================
# 1. LOGISTIC REGRESSION (WEIGHTED FOR IMBALANCE)
# ==========================================

def sigmoid(z):
    """Sigmoid activation function"""
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))


def calculate_gradient_pdf(W, X, y):
    """
    Weighted gradient:
    Samples with label -1 (not 7) get weight 0.1
    Samples with label +1 (7) get weight 1.0
    """
    # x4* w4 + x3 * w3 + .... + w0= 0
    linear_comb = X @ W
    exponent = np.clip(y * linear_comb, -250, 250)
    denominator = 1.0 + np.exp(exponent)

    # Base multiplier from derivative
    multiplier = -y / denominator

    # ==========================================
    # CLASS IMBALANCE HANDLING
    # not 7 -> 0.5 influence
    # 7     -> 1.0 influence
    # ==========================================
    sample_weights = np.where(y == -1, 0.5, 1.0)

    # Apply weights per sample
    multiplier = multiplier * sample_weights

    # Sum all weighted sample contributions
    grad = X.T @ multiplier
    return grad


def gradient_descent_pdf(X, y, eta=1e-5, iterations=50, tol=1e-7):
    """
    Gradient descent update rule:
    W(n+1) = W(n) - eta * gradient
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    W = np.zeros(X_b.shape[1])

    for i in range(iterations):
        grad = calculate_gradient_pdf(W, X_b, y)
        W = W - eta * grad

        if np.linalg.norm(grad) < tol:
            break

        if (i + 1) % 1000 == 0:
            print(f"Iteration {i+1}/{iterations} completed.")

    return W


def predict_pdf(X, W, threshold=0.5):
    """
    Predict labels (+1 / -1)
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    probs = sigmoid(X_b @ W)
    return np.where(probs >= threshold, 1, -1)


# ==========================================
# 2. DATA PREP (MNIST "7" vs "Not 7")
# ==========================================

print("Fetching MNIST dataset... (this may take a minute)")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

print("Preparing data...")
y_bipolar = np.where(y == '7', 1, -1)



X_train, X_test, y_train, y_test = train_test_split(
    X, y_bipolar, test_size=10000, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 3. FEATURE EXTRACTION (PCA)
# ==========================================
# best parameters are 300
print("Applying PCA...")
pca = PCA(n_components=50, random_state=42)
X_train_features = pca.fit_transform(X_train_scaled)
X_test_features = pca.transform(X_test_scaled)

"""
X_train_features = X_train_scaled
X_test_features = X_test_scaled
"""

print(f"Original dimensions: {X_train_scaled.shape[1]}")
print(f"Reduced dimensions after PCA: {X_train_features.shape[1]}")


# ==========================================
# 4. TRAINING
# ==========================================

print("Training model...")
W_hat = gradient_descent_pdf(
    X_train_features,
    y_train,
    eta=1e-2,
    iterations=50000
)


# ==========================================
# 5. EVALUATION
# ==========================================

print("Evaluating...")

y_pred_train = predict_pdf(X_train_features, W_hat)
y_pred_test = predict_pdf(X_test_features, W_hat)

# Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# Precision, Recall, F1 (positive class = +1 => digit 7)
precision = precision_score(y_test, y_pred_test, pos_label=1)
recall = recall_score(y_test, y_pred_test, pos_label=1)
f1 = f1_score(y_test, y_pred_test, pos_label=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test, labels=[1, -1])

# ==========================================
# 6. RESULTS
# ==========================================

print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
print(f"Precision         : {precision * 100:.2f}%")
print(f"Recall            : {recall * 100:.2f}%")
print(f"F1-Score          : {f1 * 100:.2f}%")

print("\nConfusion Matrix")
print("Rows = Actual, Columns = Predicted")
print("Order of classes: [7 (+1), Not 7 (-1)]")
print(cm)