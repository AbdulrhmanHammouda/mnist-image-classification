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
    Samples with label -1 (not 7) get weight 0.5
    Samples with label +1 (7) get weight 1.0
    """
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

        # Reduced print frequency during tuning so it doesn't spam the console
        if (i + 1) % 5000 == 0:
            print(f"    Iteration {i+1}/{iterations} completed.")

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

print("Preparing and splitting data...")
y_bipolar = np.where(y == '7', 1, -1)

# FIRST SPLIT: Isolate 10,000 samples for the final Test Set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_bipolar, test_size=10000, random_state=42
)

# SECOND SPLIT: Isolate 10,000 samples for the Validation Set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=10000, random_state=42
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
pca_components_to_test = [50, 100, 300]
learning_rates_to_test = [1e-3, 1e-2]
iterations_to_test = [5000, 20000, 50000]

# Trackers for the best model
best_val_accuracy = 0.0
best_params = {}
best_W = None
best_pca = None

print("\nStarting automated hyperparameter tuning...")
print("==========================================")

for n_components in pca_components_to_test:
    for eta in learning_rates_to_test:
        for iterations in iterations_to_test:
            print(f"Testing combination -> PCA Components: {n_components} | Learning Rate: {eta} | Iterations: {iterations}")
            
            # Apply PCA (Must be re-applied for each n_components test)
            pca = PCA(n_components=n_components, random_state=42)
            X_train_features = pca.fit_transform(X_train_scaled)
            X_val_features = pca.transform(X_val_scaled)
            
            # Train model
            W_hat = gradient_descent_pdf(
                X_train_features,
                y_train,
                eta=eta,
                iterations=iterations
            )
            
            # Evaluate on Validation Set
            y_pred_val = predict_pdf(X_val_features, W_hat)
            val_acc = accuracy_score(y_val, y_pred_val)
            
            print(f"  -> Validation Accuracy: {val_acc * 100:.2f}%\n")
            
            # Check if this is the best model so far
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_params = {'n_components': n_components, 'eta': eta, 'iterations': iterations}
                best_W = W_hat
                best_pca = pca

print("==========================================")
print("TUNING COMPLETE!")
print(f"Best Validation Accuracy : {best_val_accuracy * 100:.2f}%")
print(f"Best Parameters Found    : PCA = {best_params['n_components']}, eta = {best_params['eta']}, iterations = {best_params['iterations']}")
print("==========================================")


# ==========================================
# 4. FINAL EVALUATION (ON TEST SET)
# ==========================================

print("\nEvaluating winning model on the unseen Test Set...")

# Transform the test set using the best PCA transformer found during tuning
X_test_features = best_pca.transform(X_test_scaled)
X_train_features_best = best_pca.transform(X_train_scaled) # For train accuracy comparison

# Predict using the best weights
y_pred_train = predict_pdf(X_train_features_best, best_W)
y_pred_test = predict_pdf(X_test_features, best_W)

# Calculate final metrics
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

precision = precision_score(y_test, y_pred_test, pos_label=1)
recall = recall_score(y_test, y_pred_test, pos_label=1)
f1 = f1_score(y_test, y_pred_test, pos_label=1)
cm = confusion_matrix(y_test, y_pred_test, labels=[1, -1])

# ==========================================
# 5. RESULTS
# ==========================================

print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
print(f"Precision         : {precision * 100:.2f}%")
print(f"Recall            : {recall * 100:.2f}%")
print(f"F1-Score          : {f1 * 100:.2f}%")

print("\nConfusion Matrix (Test Set)")
print("Rows = Actual, Columns = Predicted")
print("Order of classes: [7 (+1), Not 7 (-1)]")
print(cm)