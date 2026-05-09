# KNN Multiclass Image Classification: A Detailed Code Guide

Welcome! If you are new to machine learning, Python, or these specific algorithms, this guide is for you. We will go through the `KNN_MultiClass.ipynb` notebook section by section, explaining what the code does, why we do it, and what the terminology means.

## Table of Contents
1. [Imports & Setup](#1-imports--setup)
2. [Data Loading & Exploration](#2-data-loading--exploration)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Feature Engineering (PCA, HOG, CNN)](#4-feature-engineering)
5. [KNN Implementation](#5-knn-implementation)
6. [Hyperparameter Tuning with Cross-Validation](#6-hyperparameter-tuning-with-cross-validation)
7. [Regularisation & Bias-Variance Analysis](#7-regularisation--bias-variance-analysis)
8. [Learning Curves](#8-learning-curves)
9. [Final Evaluation & Confusion Matrix](#9-final-evaluation--confusion-matrix)

---

## 1. Imports & Setup

At the very beginning of the notebook, we import the "libraries" (packages of pre-written code) that we need.
* **`numpy` (as `np`)**: The standard library for math and working with arrays/matrices of numbers.
* **`matplotlib.pyplot` (as `plt`)**: A library used to draw graphs, charts, and display images.
* **`sklearn` (Scikit-Learn)**: The main machine learning library in Python. We import many tools from here (like PCA, KNeighborsClassifier, etc.).
* **`tensorflow` / `keras`**: A deep learning library we use later specifically to load the MNIST dataset and run our images through a pre-trained CNN (MobileNetV2).

## 2. Data Loading & Exploration

```python
from tensorflow.keras.datasets import mnist
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
```
* **What this does**: Downloads the MNIST dataset. MNIST is a famous dataset of handwritten digits (0-9). 
* `X` represents our "features" (the images themselves). 
* `y` represents our "labels" (the actual digit the image represents, like a '5' or a '3').
* `_train` is the data we use to teach the model. `_test` is the data we use to grade the model at the very end.

## 3. Data Preprocessing

Before feeding images to an ML algorithm, we have to prepare the data.

### Normalisation & Flattening
```python
X_train_norm = X_train_raw.reshape(-1, 784) / 255.0
```
* **Reshape**: The raw images are 2D grids of pixels (28 rows by 28 columns). Most traditional ML algorithms expect a flat list of numbers. So, we flatten the 28x28 image into a single 1D list of 784 pixels (28 * 28 = 784).
* **Divide by 255.0**: Pixel colors are represented by numbers from 0 (black) to 255 (white). Dividing by 255 shrinks these numbers to be between 0.0 and 1.0. This is called "normalization" and helps ML models learn much faster and more accurately.

### Subsampling
```python
# Stratified subsample
```
* **What this does**: Training on all 60,000 MNIST images with KNN is very slow. We grab a smaller, random subset (e.g., 10,000 images). "Stratified" means we ensure we get an equal number of 0s, 1s, 2s, etc., so the model doesn't become biased toward one specific number.

## 4. Feature Engineering

"Feature Engineering" means transforming our raw pixels into a smarter, more condensed format so the model can understand the image better. We use three different techniques to compare them.

### PCA (Principal Component Analysis)
```python
pca = PCA(n_components=0.95, random_state=42)
X_tr_pca = pca.fit_transform(X_tr)
```
* **What this does**: PCA is a mathematical technique that reduces the size of the data. 784 pixels is a lot. Many pixels (like the black background at the edges) carry no useful information. PCA squishes the 784 pixels down to a much smaller number (like 150) while retaining 95% of the "variance" (the important structural information). This makes training faster and removes noise.

### HOG (Histogram of Oriented Gradients)
```python
f = hog(img_flat.reshape(28, 28), orientations=9...)
```
* **What this does**: HOG looks at the *edges* and *corners* in an image. Instead of looking at individual pixel brightness, it calculates which direction the lines in the image are going (e.g., vertical lines, horizontal loops). This is incredibly helpful for recognizing shapes like handwritten numbers.

### CNN (Pretrained MobileNetV2)
```python
X_tr_cnn = extractor.predict(imgs_to_rgb96(X_tr))
```
* **What this does**: We take a massive, deep Neural Network (MobileNetV2) that Google already trained on millions of images. We pass our MNIST images through it, and instead of asking it to predict the image, we extract the network's internal "understanding" of the image (a list of numbers). This gives our KNN model world-class, highly complex features to learn from.

## 5. KNN Implementation

KNN stands for **K-Nearest Neighbors**. It is one of the simplest ML algorithms:
1. You give the model a new, unknown image.
2. The model looks at all the images it memorized during training.
3. It finds the "K" closest images (e.g., if K=5, it finds the 5 most similar images).
4. The unknown image gets classified as whatever the majority of those 5 neighbors are.

## 6. Hyperparameter Tuning with Cross-Validation

Hyperparameters are the "settings" or "dials" of a machine learning model. For KNN, we need to decide:
1. **`n_neighbors` (K)**: How many neighbors should we look at? 1? 5? 15?
2. **`metric`**: How do we measure "distance" between images? (Euclidean straight-line distance, or Manhattan block-distance?)
3. **`weights`**: Should all 5 neighbors get an equal vote (`uniform`), or should closer neighbors get a stronger vote (`distance`)?

```python
gs = GridSearchCV(knn, param_grid, cv=cv...)
gs.fit(Xtr, y_tr)
```
* **GridSearchCV**: This automatically tries *every single combination* of the settings we defined in `param_grid`.
* **Cross-Validation (`cv`)**: To test if a setting is good, it splits the training data into 5 chunks (folds). It trains on 4 chunks and tests on the 1 remaining chunk, rotating until all chunks have been tested. This ensures we don't accidentally pick a setting that just got lucky.

## 7. Regularisation & Bias-Variance Analysis

In machine learning, we battle two enemies:
* **High Bias (Underfitting)**: The model is too simple and didn't learn anything useful.
* **High Variance (Overfitting)**: The model is too complex. It memorized the training data perfectly but fails completely on new, unseen data.

In KNN, **K** controls this trade-off:
* Small K (like 1) = Overfitting (looks at just 1 noisy neighbor).
* Large K (like 50) = Underfitting (blurs too many things together).

**Distance-Weighted Voting**: This is our "Regularization" (a technique to prevent overfitting). By making closer neighbors count more, the model is less sensitive to noise at the outer edges of the neighbor group.

## 8. Learning Curves

```python
learning_curve(...)
```
* **What this does**: A learning curve plots how well the model performs as we give it more and more training data.
* We plot two lines: Training Error (how well it does on data it has seen) and Validation Error (how well it does on new data).
* If the lines are very far apart, we are overfitting. If they are close together but the error is high, we are underfitting. Ideally, they meet together at a very low error rate.

## 9. Final Evaluation & Confusion Matrix

Once `GridSearchCV` finds the absolute best settings (the best features, the best K, the best weights), we apply those settings to the 10,000 `X_test` images that the model has *never seen before*.

```python
acc = accuracy_score(y_test, pred) * 100
```
* **Accuracy**: The percentage of test images it guessed correctly.

```python
cm = confusion_matrix(y_test, best_r['pred'])
disp = ConfusionMatrixDisplay(...)
```
* **Confusion Matrix**: A grid that shows exactly *where* the model gets confused. For example, it might show that the model is very good at predicting '1's, but it frequently confuses '4's with '9's. The rows represent the *true* labels, and the columns represent the *predicted* labels.
