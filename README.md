# 🧠 MNIST Image Classification Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange.svg)](https://scikit-learn.org/)

> A comprehensive machine learning project focusing on building, evaluating, and optimizing binary and multiclass classification models using the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

This repository implements a complete machine learning pipeline from data preprocessing to model evaluation. It was originally developed to explore traditional machine learning algorithms, handle class imbalances, perform hyperparameter tuning, and rigorously evaluate model performance.

---

## 🎯 Project Objectives

The project is divided into two progressive phases:

### Phase 1: Binary Classification (Baseline System)

Focuses on distinguishing between the digit **"7"** and **"not 7"**.

- **Data Processing:** Image flattening, Principal Component Analysis (PCA) for dimensionality reduction, and handling class imbalances.
- **Models Implemented:** Logistic Regression (with custom class weights and undersampling), Linear Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Bernoulli Naive Bayes.
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

### Phase 2: Multiclass Classification & Intelligent Enhancement

Scaling the pipeline to classify all **10 digit classes** (0-9).

- **Models Implemented:** Multinomial Logistic Regression, Multiclass Linear SVM, and Naive Bayes.
- **Enhancements:** Hyperparameter tuning with cross-validation, regularization analysis, and model optimization.

---

## 📂 Repository Structure

```text
MNIST-Image-Classification/
├── 📁 Binary Classification/
│   ├── 📁 KNN/                     # K-Nearest Neighbors implementation
│   ├── 📁 LinearSVM/               # Linear SVM analysis and experiments
│   ├── 📁 KernelSVM/               # Kernel SVM analysis and experiments
│   ├── 📁 Logistic Regression/     # Class imbalance handling, undersampling, and tuning
│   └── 📁 Naive Bayes/             # Bernoulli Naive Bayes experiments
├── 📁 Multiclass Classification/
│   ├── 📁 LinearSVM/               # Multiclass Linear SVM implementation and analysis
│   ├── 📁 KernelSVM/               # Multiclass Kernel SVM implementation and analysis
│   ├── 📁 Multionomial/     # Multinomial logistic regression
│   └── 📁 Naive Bayes/             # Multiclass Naive Bayes experiments
├── 📁 reports/                     # Technical reports, mathematical formulations, and results
├── 📄 README.md                    # Project documentation
└── 📄 requirements.txt             # Python dependencies
```

---

## ✨ Key Features & Implementations

### 1. Data Processing Pipeline

- **Dimensionality Reduction:** Utilized Principal Component Analysis (PCA) to reduce feature space while retaining high variance.
- **Feature Scaling:** Applied `StandardScaler` for zero-mean and unit-variance normalization.
- **Imbalance Handling:** Implemented weighted loss functions and undersampling techniques specifically for the heavily imbalanced binary classification task (~9:1 ratio).

### 2. Machine Learning Algorithms

- **Logistic Regression:** Custom implementations addressing class imbalance via weighted gradient descent and hyperparameter grid search.
- **Support Vector Machines (SVM):** Extensive experiments using Linear and Kernel SVMs, including decision boundary analysis and margin optimizations.
- **K-Nearest Neighbors (KNN):** Distance-based classification with optimal 'k' selection.
- **Naive Bayes:** Probabilistic classification utilizing the Bernoulli variant for binary tasks and Multinomial/Gaussian for multiclass.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/MNIST-Image-Classification.git
   cd MNIST-Image-Classification
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - **Windows (CMD/PowerShell):**
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Technology Stack

- **NumPy:** Core numerical computing and matrix operations.
- **Scikit-Learn:** Machine learning framework for algorithms, preprocessing, cross-validation, and metrics.
- **Matplotlib & Seaborn:** Data visualization, learning curves, and confusion matrices.
- **Jupyter Notebook:** Interactive data exploration and result analysis.

---

## 📈 Results & Evaluation

_(Populate this section with summary tables or key insights from your experiments)_

- **Binary Classification (7 vs Rest):** Achieved robust F1-scores by effectively handling the ~9:1 class imbalance.
- **Multiclass Classification (0-9):** Reached high accuracy across all 10 digits, minimizing confusion between visually similar digits (e.g., 4 and 9, 3 and 8).


