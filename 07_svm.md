# Support Vector Machines (SVM) - Complete Guide

## 1. Fundamentals of Support Vector Machines

### 1.1 What is SVM?

**Definition:** Supervised learning algorithm that finds optimal hyperplane maximizing margin between classes in high-dimensional space.

**Core Idea:**
```
Find the widest street that separates two classes
```

**Key Characteristics:**
- Powerful for both classification and regression
- Works in high-dimensional spaces
- Memory efficient (uses subset of training data as support vectors)
- Versatile (kernel trick enables non-linear boundaries)

---

### 1.2 Binary Classification Problem

**Goal:** Find decision boundary separating two classes

**Linear Case:**
```
Hyperplane: w·x + b = 0

Where:
- w: Weight vector (normal to hyperplane)
- x: Feature vector
- b: Bias term
```

**Prediction:**
```
If w·x + b > 0: Predict Class 1
If w·x + b < 0: Predict Class -1
```

**Distance:** |w·x + b| / ||w|| (distance from point to hyperplane)

---

### 1.3 SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|--------|-----|-------------------|
| Decision Boundary | Linear (or non-linear with kernel) | Linear |
| Optimization | Margin maximization | Likelihood maximization |
| Loss Function | Hinge loss | Log loss |
| Output | Distance (sign) | Probability |
| Kernel Trick | Yes | No (manual features) |
| High-d Data | Excellent | Moderate |
| Interpretability | Low | High |
| Sparse Solution | Yes (support vectors) | No |

---

## 2. Maximum Margin & Support Vectors

### 2.1 Margin Concept

**Margin:** Distance between hyperplane and closest data points

**Goal:** Maximize margin (wider separation → Better generalization)

```
        Class 1
            o
            o
    ----o--------o---   <- Hyperplane
    --------o--------
          o
        Class 0
```

**Intuition:** Wide margin means robust decision (points far from boundary)

---

### 2.2 Support Vectors

**Definition:** Training samples lying exactly on margin boundary

**Characteristics:**
- Only these samples matter for boundary
- Other samples irrelevant
- Typically 5-30% of training data (sparse solution)

**Example:**
```
100 samples total
95 samples far from boundary: IGNORED
5 samples on/near margin: SUPPORT VECTORS (define boundary)
```

**Advantage:** Sparse representation (memory efficient)

**Mathematical:** Support vectors satisfy: |w·x_i + b| = 1 (for correctly classified)

---

### 2.3 Optimization Problem (Hard Margin SVM)

**Objective:** Maximize margin = 1 / ||w||

**Equivalently:** Minimize ||w||²

**Constraints:** All samples correctly classified
```
y_i(w·x_i + b) ≥ 1  for all i
```

Where y_i ∈ {-1, +1}

**Solution:** Quadratic programming problem (convex, unique global optimum)

---

## 3. Soft Margin SVM (Handling Non-Separable Data)

### 3.1 Problem: Not Linearly Separable

**Reality:** Most real-world data not perfectly separable

**Hard Margin Problem:** No solution if overlap

**Solution:** Allow some misclassifications (soft margin)

---

### 3.2 Slack Variables

**Idea:** Introduce slack ξ_i ≥ 0 for each sample

**Constraint:** y_i(w·x_i + b) ≥ 1 - ξ_i

**Interpretation:**
- ξ_i = 0: Correctly classified, outside margin
- 0 < ξ_i < 1: Correctly classified, inside margin
- ξ_i > 1: Misclassified

---

### 3.3 Soft Margin Objective

**Optimization Problem:**
```
Minimize: (1/2)||w||² + C × Σξ_i
```

Where:
- C: Regularization parameter (trade-off parameter)
- Σξ_i: Total slack (misclassification penalty)

**Trade-off:**
- Low C: Favor wide margin, tolerate errors
- High C: Minimize errors, narrow margin

---

### 3.4 C Parameter Effects

**C = Very High (C=1000):**
- Penalizes misclassification heavily
- Narrow margin, tight fit
- Overfitting risk
- All support vectors on/near boundary

**C = Moderate (C=1):**
- Balance between margin and error
- Typical choice

**C = Low (C=0.01):**
- Favor wide margin
- Allow more errors
- Simpler boundary
- Underfitting risk

---

## 4. Kernel Trick (Non-linear SVM)

### 4.1 Problem: Non-linear Separability

**Reality:** Many datasets not linearly separable

**Approach 1:** Manual polynomial features
- x₁, x₂ → x₁, x₂, x₁², x₁×x₂, x₂²
- Expensive, arbitrary

**Approach 2:** Kernel trick (SVM solution)
- Automatically transform to high-d space
- Compute in original space (efficient)
- No explicit feature creation

---

### 4.2 Kernel Trick Mechanism

**Idea:** Replace dot product with kernel function

**Original (Linear):**
```
w·x = Σ w_i × x_i
```

**With Kernel:**
```
K(x_i, x_j) = φ(x_i) · φ(x_j)
```

Where φ: Implicit feature transformation

**Advantage:** Compute K(x_i, x_j) efficiently without computing φ explicitly

**Example:**
```
Polynomial kernel: K(x, y) = (x·y + 1)^d
Equivalent to: φ(x) = all polynomials up to degree d
But compute directly without expanding!
```

---

### 4.3 Common Kernels

#### Linear Kernel
```
K(x, y) = x·y
```
- No transformation
- Fast, interpretable
- Use when already separable or high-d sparse data

#### Polynomial Kernel
```
K(x, y) = (x·y + c)^d
```

Where:
- d: Degree (1=linear, 2=quadratic, etc.)
- c: Offset (default 0)

**Use When:**
- Moderate non-linearity
- d=2,3 typical
- Higher d: Complex decision boundary

**Example:**
```
d=1: Linear
d=2: Quadratic (curves)
d=3: Cubic (more complex curves)
```

#### RBF (Radial Basis Function) Kernel
```
K(x, y) = exp(-γ × ||x-y||²)
```

Where:
- γ (gamma): Kernel coefficient
- Controls influence of each training sample

**Interpretation:**
- Similarity measure (Gaussian bump)
- Low γ: Smooth decision boundary (global influence)
- High γ: Complex boundary (local influence)

**Use When:**
- Strong non-linearity
- Most common choice
- Default in sklearn

#### Sigmoid Kernel
```
K(x, y) = tanh(α × x·y + c)
```

**Resembles:** Neural network hidden layer

**Use When:**
- Resembles neural network problems
- Rarely used (RBF usually better)

---

### 4.4 Kernel Selection

| Kernel | Data Type | Use When | Pros | Cons |
|--------|-----------|----------|------|------|
| Linear | Any | Linearly separable, high-d sparse | Fast, interpretable | Poor non-linear |
| Polynomial | Dense | Moderate non-linearity | Controlled complexity | Slow for high-d |
| RBF | Dense | Strong non-linearity | Flexible, works well | Slow, less interpretable |
| Sigmoid | Any | Neural network-like | Flexible | Rarely optimal |

**Strategy:** Start Linear → Polynomial (d=2,3) → RBF

---

## 5. SVM Hyperparameters

### 5.1 C (Regularization)

**Effect:**
- Low C: Underfitting (wide margin, more errors)
- High C: Overfitting (narrow margin, few errors)
- Typical: 0.1 to 100

**Tuning:**
```python
from sklearn.model_selection import GridSearchCV

C_values = [0.01, 0.1, 1, 10, 100]
param_grid = {'C': C_values}
# Cross-validate to find best
```

---

### 5.2 Gamma (Kernel Coefficient, RBF only)

**Effect:**
- Low γ: Smooth boundary (underfitting)
- High γ: Complex boundary (overfitting)
- Typical: 0.0001 to 1

**Relationship:** γ = 1 / (2σ²) in Gaussian interpretation

**Large γ:**
```
K(x, y) = exp(-high × ||x-y||²)
Only nearby points matter
Local boundary
```

**Small γ:**
```
K(x, y) = exp(-low × ||x-y||²)
All points matter
Global boundary
```

---

### 5.3 Kernel

**Linear:** Fastest, for separable data

**Polynomial:** Degree d (2-3 typical)

**RBF:** Most flexible (default)

---

### 5.4 Hyperparameter Tuning

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

svm = SVC()
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

---

## 6. SVM for Regression (SVR)

### 6.1 Support Vector Regression

**Idea:** Predict continuous values, maximize margin around predictions

**Loss Function (ε-insensitive):**
```
Loss = 0 if |y - ŷ| < ε
Loss = |y - ŷ| - ε if |y - ŷ| ≥ ε
```

**Interpretation:**
- Errors within ε: No penalty (insensitive band)
- Errors beyond ε: Linear penalty

**ε Parameter:**
- Small ε: Tight fit (overfitting risk)
- Large ε: Wide band (underfitting risk)

---

### 6.2 SVR vs Linear Regression

**Linear Regression:**
- Minimizes squared error
- Smooth surface
- Sensitive to outliers

**SVR:**
- Minimizes ε-insensitive loss
- Non-linear (with kernel)
- Robust to outliers (beyond ε ignored)

---

## 7. Multi-class SVM

### 7.1 Binary to Multi-class

**SVM naturally binary (w·x + b splits space into 2)**

**Multi-class Solutions:**

**1. One-vs-Rest (OvR):**
- Train k binary classifiers (class_i vs rest)
- For k classes, k classifiers
- Predict class with highest score
- Simple, scalable

**2. One-vs-One (OvO):**
- Train k(k-1)/2 classifiers (each pair)
- For k classes, k(k-1)/2 classifiers
- Vote on final class
- More computation, sometimes better

**Example (k=3 classes A, B, C):**

OvR: 3 classifiers
- A vs (B+C)
- B vs (A+C)
- C vs (A+B)

OvO: 3 classifiers
- A vs B
- A vs C
- B vs C

---

### 7.2 sklearn Default

**Multi_class parameter:**
```python
SVC(multi_class='ovr')      # One-vs-Rest
SVC(multi_class='ovo')      # One-vs-One
```

**Default:** OvR (faster for large k)

---

## 8. Advantages of SVM

### 8.1 Powerful Non-linear Classifier

**Kernel Trick:**
- Handles non-linear patterns
- High-d transformations implicit
- Efficient computation

---

### 8.2 Effective in High Dimensions

**Performance:**
- Works well even when p > n (features > samples)
- Typical for text (10k words, few docs)
- Sparse data efficient

---

### 8.3 Memory Efficient

**Sparse Solution:**
- Uses only support vectors (subset of data)
- Typical: 5-30% of training data
- Smaller model than k-NN (which stores all)

---

### 8.4 Theoretically Grounded

**Margin Theory:**
- Maximizing margin → Better generalization
- Validated by statistics
- Principled approach

---

### 8.5 Versatile

**Both Classification & Regression:**
- SVC for classification
- SVR for regression
- Any kernel for both

---

## 9. Disadvantages of SVM

### 9.1 Computational Cost

**Training:**
- Quadratic programming: O(n²) to O(n³)
- Slow for large datasets (n > 100k problematic)
- Scales poorly with n

**Prediction:**
- O(n_support_vectors × d) (if many support vectors)

**Solution:** SGDClassifier with hinge loss for large datasets

---

### 9.2 Parameter Tuning

**Many Parameters:**
- C (regularization)
- Kernel type
- Kernel parameters (d for poly, γ for RBF)

**Grid Search Expensive:** Large search space

**Solution:** Start simple (linear), add complexity if needed

---

### 9.3 Poor Probability Estimates

**Classification:**
- Decision function: w·x + b (distance, not probability)
- Not calibrated probabilities

**Workaround:**
```python
SVC(probability=True)  # Sigmoid calibration (slower)
```

---

### 9.4 Sensitive to Feature Scaling

**Problem:** Different scales → Distance dominated by large-scale

**Example:**
```
Age (0-100) vs Income (0-1000000)
Kernel distances: Income dominates
```

**Solution:** ALWAYS scale features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 9.5 Difficult to Interpret

**Black Box:**
- Support vectors support-space location hard to interpret
- Kernel non-linear transformation hidden
- Can't easily see "why" decision made

**Workaround:** SHAP values for explanation

---

### 9.6 Not Suitable for Very Large Datasets

**Practical Limit:** ~100k samples
- Memory: Stores kernel matrix (n×n)
- Time: Quadratic or cubic in n

**Alternatives for Large Scale:**
- SGDClassifier (online, stochastic)
- Mini-batch SGD
- Approximate kernels

---

### 9.7 Imbalanced Data

**Problem:** SVM biased toward majority class

**Solution:**
```python
SVC(class_weight='balanced')  # Auto-weight classes inversely proportional
```

---

## 10. SVM vs Other Algorithms

| Aspect | SVM | Logistic Regression | Decision Tree | Neural Network |
|--------|-----|-------------------|--------------|----------------|
| Non-linear | Yes (kernel) | No | Yes | Yes |
| High-d | Excellent | Good | Poor | Good |
| Training Speed | Slow | Fast | Medium | Very Slow |
| Prediction Speed | Medium | Very Fast | Fast | Medium |
| Interpretability | Low | High | High | Very Low |
| Tuning Difficulty | Hard | Easy | Easy | Very Hard |
| Probability | No (workaround) | Yes | No | Yes |
| Multiclass | OvR/OvO | Softmax | Native | Native |

---

## 11. Implementation in sklearn

### 11.1 Classification

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Scale data (CRITICAL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(svm, X_train, y_train, cv=5).mean()}")
print(f"\n{classification_report(y_test, y_pred)}")

# Support vectors
print(f"Number of support vectors: {len(svm.support_vectors_)}")
print(f"Percentage: {100 * len(svm.support_vectors_) / len(X_train):.1f}%")
```

### 11.2 Regression

```python
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Train
svr = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
svr.fit(X_train, y_train)

# Predict
y_pred = svr.predict(X_test)

# Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### 11.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

svm = SVC()
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
print(f"Test accuracy: {grid.best_estimator_.score(X_test, y_test)}")
```

### 11.4 Probability Estimates

```python
# Enable probability (slower, more accurate calibration)
svm = SVC(kernel='rbf', C=1.0, probability=True)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
y_pred_proba = svm.predict_proba(X_test)

print(f"Probability for first sample: {y_pred_proba[0]}")  # [P(class 0), P(class 1)]
```

### 11.5 Feature Scaling Pipeline

```python
from sklearn.pipeline import Pipeline

# Combine scaling and SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0))
])

# Train
pipeline.fit(X_train, y_train)

# Predict (scaling applied automatically)
y_pred = pipeline.predict(X_test)
```

---

## 12. Common Issues & Solutions

### 12.1 Poor Accuracy

**Causes:**
- Wrong kernel
- Poor hyperparameters (C, γ)
- Features not scaled
- Insufficient data

**Solutions:**
```python
# 1. Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Try different kernels
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel)
    svm.fit(X_train_scaled, y_train)
    print(f"{kernel}: {svm.score(X_test_scaled, y_test)}")

# 3. Tune hyperparameters
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
```

---

### 12.2 Slow Training

**Causes:**
- Large dataset (n > 100k)
- Complex kernel (polynomial degree high)
- Fine-tuning grid too large

**Solutions:**
```python
# 1. Use linear kernel first
svm = SVC(kernel='linear')

# 2. Subset data for tuning
svm.fit(X_train_scaled[:10000], y_train[:10000])

# 3. Use SGDClassifier for large data
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='hinge', n_jobs=-1)
sgd.fit(X_train_scaled, y_train)
```

---

### 12.3 Class Imbalance

**Problem:** SVM biased toward majority

**Solutions:**
```python
# 1. Balanced weights
svm = SVC(class_weight='balanced')

# 2. Manual weights
class_weights = {0: 1, 1: 99}  # Minority heavier
svm = SVC(class_weight=class_weights)

# 3. Resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
svm.fit(X_resampled, y_resampled)
```

---

## 13. When to Use SVM

### 13.1 Good For

- Medium-sized datasets (1k-100k)
- High-dimensional data
- Binary classification
- Clear margin separation
- Interpretability not critical

### 13.2 Not Good For

- Very large datasets (>100k)
- Real-time predictions (training slow)
- Need probability estimates
- Need interpretability
- Highly imbalanced data

---

## 14. Practice Problems

1. **Margin Maximization:** Why maximize margin? Connection to generalization?

2. **Soft Margin:** When needed? What does C control?

3. **Kernel Trick:** Why useful? Polynomial kernel formula?

4. **RBF Gamma:** Low vs High. Effects on boundary?

5. **Support Vectors:** What are they? Why sparse solution good?

6. **Multi-class:** OvR vs OvO. Pros/cons?

7. **Feature Scaling:** Why critical? What happens without?

8. **Imbalance:** Problem and class_weight solution?

---

## 15. Key Takeaways

1. **Core idea:** Maximize margin (distance to boundary)
2. **Support vectors:** Training samples defining boundary (sparse)
3. **Soft margin:** Allow errors via C parameter
4. **Kernel trick:** Non-linear via implicit feature transformation
5. **Common kernels:** Linear, Polynomial (d), RBF (γ)
6. **Parameters:** C (regularization), Kernel, Kernel params
7. **Advantages:** Non-linear, high-d, memory-efficient, theoretical
8. **Disadvantages:** Slow training, parameter tuning, scaling sensitive, poor probabilities
9. **Multi-class:** OvR or OvO
10. **When use:** Medium data, high-d, non-linear, accuracy priority

---

**Next Topic:** Random Forest (say "next" to continue)