import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
import os

# Create output directories
os.makedirs('Multiclass Classification/Multinomial/figures', exist_ok=True)

# ==========================================
# 1. CORE FUNCTIONS (FROM SCRATCH)
# ==========================================

def softmax(Z):
    """Numerically stable softmax: shift by row-max to prevent overflow."""
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

class MultinomialRegression:
    """Multinomial Logistic Regression using vanilla mini-batch gradient descent."""

    def __init__(self, eta=0.005, epochs=100, batch_size=512, lam=0.01):
        self.eta = eta          # Learning rate
        self.epochs = epochs    # Number of full passes over training data
        self.batch_size = batch_size
        self.lam = lam          # L2 regularization strength
        self.W = None
        self.history_train_loss = []
        self.history_val_loss = []

    def fit(self, X_train, Y_train_onehot, X_val=None, Y_val_onehot=None):
        n_samples, n_features = X_train.shape
        n_classes = Y_train_onehot.shape[1]

        # Initialize weights to zeros
        self.W = np.zeros((n_features + 1, n_classes))

        # Prepend bias column of 1s
        X_train_b = np.c_[np.ones((n_samples, 1)), X_train]
        if X_val is not None:
            X_val_b = np.c_[np.ones((X_val.shape[0], 1)), X_val]

        self.history_train_loss = []
        self.history_val_loss = []

        for epoch in range(self.epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train_b[indices]
            Y_shuffled = Y_train_onehot[indices]

            # Mini-batch loop
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                Y_batch = Y_shuffled[i:i + self.batch_size]
                batch_size = X_batch.shape[0]

                # Forward pass
                Z = X_batch @ self.W
                Y_hat = softmax(Z)
                error = Y_hat - Y_batch

                # Gradient + L2 regularization (exclude bias from penalty)
                grad = (1.0 / batch_size) * (X_batch.T @ error)
                W_reg = np.copy(self.W)
                W_reg[0, :] = 0  # Don't regularize the bias row
                grad += self.lam * W_reg

                # Vanilla gradient descent update
                self.W -= self.eta * grad

            # ---- End-of-epoch loss tracking ----
            Y_hat_full = softmax(X_train_b @ self.W)
            train_loss = -np.mean(np.sum(Y_train_onehot * np.log(Y_hat_full + 1e-15), axis=1))
            self.history_train_loss.append(train_loss)

            if X_val is not None and Y_val_onehot is not None:
                Y_hat_val = softmax(X_val_b @ self.W)
                val_loss = -np.mean(np.sum(Y_val_onehot * np.log(Y_hat_val + 1e-15), axis=1))
                self.history_val_loss.append(val_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        return self

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return softmax(X_b @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ==========================================
# 2. CUSTOM METRICS (FROM SCRATCH)
# ==========================================

def custom_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def custom_classification_report(y_true, y_pred, digits=4):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    classes = np.sort(classes)
    
    report = f"{'':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n\n"
    
    total_support = len(y_true)
    
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    
    for c in classes:
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))
        
        support = TP + FN
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support
        
        report += f"{c:<12} {precision:>10.{digits}f} {recall:>10.{digits}f} {f1:>10.{digits}f} {support:>10}\n"
    
    n_classes = len(classes)
    macro_p /= n_classes
    macro_r /= n_classes
    macro_f1 /= n_classes
    
    weighted_p /= total_support
    weighted_r /= total_support
    weighted_f1 /= total_support
    
    accuracy = custom_accuracy_score(y_true, y_pred)
    
    report += f"\n{'accuracy':<12} {'':>10} {'':>10} {accuracy:>10.{digits}f} {total_support:>10}\n"
    report += f"{'macro avg':<12} {macro_p:>10.{digits}f} {macro_r:>10.{digits}f} {macro_f1:>10.{digits}f} {total_support:>10}\n"
    report += f"{'weighted avg':<12} {weighted_p:>10.{digits}f} {weighted_r:>10.{digits}f} {weighted_f1:>10.{digits}f} {total_support:>10}\n"
    
    return report

def roc_auc_binary(y_true_binary, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true_binary = y_true_binary[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_binary.size - 1]
    
    tps = np.cumsum(y_true_binary)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan
        
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    
    auc = np.trapezoid(tpr, x=fpr)
    return auc

def custom_roc_auc_score(y_true_oh, y_pred_proba):
    n_classes = y_true_oh.shape[1]
    auc_scores = []
    
    for c in range(n_classes):
        auc = roc_auc_binary(y_true_oh[:, c], y_pred_proba[:, c])
        auc_scores.append(auc)
        
    return np.mean(auc_scores)


# ==========================================
# 3. DATA PREPARATION
# ==========================================

print("Fetching MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

print("Applying PCA and Polynomial Features...")
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_feat = poly.fit_transform(X_train_pca)
X_test_feat = poly.transform(X_test_pca)

Y_train_full_oh = np.eye(10)[y_train_full]
Y_test_oh = np.eye(10)[y_test]

# ==========================================
# 4. K-FOLD CROSS VALIDATION
# ==========================================
print("\n--- 3-Fold Cross-Validation ---")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []
fold = 1

for train_idx, val_idx in kf.split(X_train_feat):
    print(f"Fold {fold} Training...")
    X_f_train, X_f_val = X_train_feat[train_idx], X_train_feat[val_idx]
    Y_f_train, Y_f_val = Y_train_full_oh[train_idx], Y_train_full_oh[val_idx]
    y_f_val_labels = y_train_full[val_idx]

    model = MultinomialRegression(eta=0.005, epochs=30, batch_size=512, lam=0.01)
    model.fit(X_f_train, Y_f_train)

    preds = model.predict(X_f_val)
    acc = custom_accuracy_score(y_f_val_labels, preds)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    cv_scores.append(acc)
    fold += 1

print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ==========================================
# 5. FULL TRAINING & LEARNING CURVES
# ==========================================
print("\n--- Training Final Model ---")
X_train_sub, X_val_sub, Y_train_oh_sub, Y_val_oh_sub = train_test_split(
    X_train_feat, Y_train_full_oh, test_size=0.2, random_state=42
)

final_model = MultinomialRegression(eta=0.005, epochs=100, batch_size=512, lam=0.01)
final_model.fit(X_train_sub, Y_train_oh_sub, X_val_sub, Y_val_oh_sub)

plt.figure(figsize=(10, 6))
plt.plot(range(1, final_model.epochs + 1), final_model.history_train_loss, label='Training Loss', color='blue')
plt.plot(range(1, final_model.epochs + 1), final_model.history_val_loss, label='Validation Loss', color='orange')
plt.title('Learning Curves (Vanilla GD + Poly Features)')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True)
plt.savefig('Multiclass Classification/Multinomial/figures/multinomial_poly_learning_curves_custom_metrics.png')
plt.close()

# ==========================================
# 6. BIAS-VARIANCE TRADEOFF
# ==========================================
print("\n--- Bias-Variance Tradeoff (Varying L2) ---")
lams = [0.001, 0.01, 0.1, 1.0, 10.0]
train_errors = []
val_errors = []

for l in lams:
    bv_model = MultinomialRegression(eta=0.005, epochs=30, batch_size=512, lam=l)
    bv_model.fit(X_train_sub, Y_train_oh_sub, X_val_sub, Y_val_oh_sub)

    y_pred_t = bv_model.predict(X_train_sub)
    y_pred_v = bv_model.predict(X_val_sub)

    train_errors.append(1 - custom_accuracy_score(np.argmax(Y_train_oh_sub, axis=1), y_pred_t))
    val_errors.append(1 - custom_accuracy_score(np.argmax(Y_val_oh_sub, axis=1), y_pred_v))

plt.figure(figsize=(10, 6))
plt.plot(np.log10(lams), train_errors, label='Training Error', marker='o')
plt.plot(np.log10(lams), val_errors, label='Validation Error', marker='o')
plt.title(r'Bias-Variance Analysis vs L2 Regularization ($\lambda$)')
plt.xlabel(r'Log10($\lambda$)')
plt.ylabel('Error Rate')
plt.legend()
plt.grid(True)
plt.savefig('Multiclass Classification/Multinomial/figures/multinomial_poly_bias_variance_custom_metrics.png')
plt.close()

# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================
print("\n--- Final Test Set Evaluation ---")

# 1. Calculate and Print Training Accuracy
y_pred_train = final_model.predict(X_train_sub)
y_train_labels = np.argmax(Y_train_oh_sub, axis=1) # Convert one-hot back to 1D labels
train_acc = custom_accuracy_score(y_train_labels, y_pred_train)
print(f"Training Accuracy: {train_acc:.4f}")

# 2. Calculate and Print Test Accuracy
y_pred_test = final_model.predict(X_test_feat)
y_pred_proba_test = final_model.predict_proba(X_test_feat)
test_acc = custom_accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 3. Print the Reports
print("\nClassification Report:")
print(custom_classification_report(y_test, y_pred_test, digits=4))

roc_auc = custom_roc_auc_score(Y_test_oh, y_pred_proba_test)
print(f"Multiclass ROC-AUC : {roc_auc:.4f}")