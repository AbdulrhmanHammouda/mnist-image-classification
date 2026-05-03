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
# 1. LOGISTIC REGRESSION (UNDERSAMPLED)
# ==========================================

def sigmoid(z):
    """Sigmoid activation function"""
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))


def calculate_gradient_pdf(W, X, y):
    """
    Standard (unweighted) gradient for logistic regression.
    Class imbalance is handled via undersampling instead.
    """
    # x4* w4 + x3 * w3 + .... + w0= 0
    linear_comb = X @ W
    exponent = np.clip(y * linear_comb, -250, 250)
    denominator = 1.0 + np.exp(exponent)

    # Base multiplier from derivative
    multiplier = -y / denominator

    # Sum all sample contributions (no weighting needed)
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

# ==========================================
# CLASS IMBALANCE HANDLING (UNDERSAMPLING)
# Reduce majority class (not 7) to match
# the number of minority class (7) samples
# ==========================================

print("Applying undersampling...")

# Separate indices by class
idx_positive = np.where(y_train == 1)[0]   # digit 7
idx_negative = np.where(y_train == -1)[0]  # not 7

n_positive = len(idx_positive)
n_negative = len(idx_negative)
print(f"Before undersampling: 7s = {n_positive}, Not 7s = {n_negative}")

# Randomly sample from the majority class to match the minority class count
rng = np.random.RandomState(42)
idx_negative_undersampled = rng.choice(idx_negative, size=n_positive, replace=False)

# Combine and shuffle
idx_balanced = np.concatenate([idx_positive, idx_negative_undersampled])
rng.shuffle(idx_balanced)

X_train = X_train[idx_balanced]
y_train = y_train[idx_balanced]

print(f"After undersampling:  7s = {np.sum(y_train == 1)}, Not 7s = {np.sum(y_train == -1)}")
print(f"Total training samples: {len(y_train)}")

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
