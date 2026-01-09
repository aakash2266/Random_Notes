# k-Nearest Neighbors (k-NN) - Complete Guide

## 1. Fundamentals of k-Nearest Neighbors

### 1.1 What is k-Nearest Neighbors?

**Definition:** Non-parametric, instance-based learning algorithm that classifies samples based on k closest neighbors in feature space.

**Core Idea:**
```
"Tell me who your neighbors are, and I'll tell you who you are"
```

**Key Characteristics:**
- Lazy learner: No training phase (memorizes training data)
- Instance-based: Stores all training data
- Non-parametric: No assumptions about data distribution
- Local learning: Nearby points have similar labels

**Classification Rule:**
```
Predict class = Majority class among k nearest neighbors
```

**Regression Rule:**
```
Predict value = Average (or weighted average) of k nearest neighbors
```

---

### 1.2 k-NN vs Eager Learners

| Aspect | k-NN (Lazy) | Decision Tree (Eager) | Logistic Regression (Eager) |
|--------|-----------|----------------------|--------------------------|
| Training | Memorize data | Build model | Optimize parameters |
| Training Time | O(n) | O(n log n) | O(np) |
| Prediction | O(nd) | O(log n) | O(p) |
| Memory | O(n) | O(tree size) | O(p) |
| Interpretability | Low | High | High |
| Assumptions | None | Splits | Linearity |

Where n=samples, d=dimensions, p=features

---

## 2. Distance Metrics

### 2.1 Euclidean Distance (L2)

**Formula:**
```
d(x_i, x_j) = √(Σ(x_ik - x_jk)²)
```

**Interpretation:** Straight-line distance in space

**Example (2D):**
```
Point A = (1, 2)
Point B = (4, 6)
d = √((4-1)² + (6-2)²) = √(9 + 16) = √25 = 5
```

**Characteristics:**
- Default in most implementations
- Sensitive to scale (feature scaling important)
- Works well in low dimensions

---

### 2.2 Manhattan Distance (L1, Taxicab)

**Formula:**
```
d(x_i, x_j) = Σ|x_ik - x_jk|
```

**Interpretation:** Sum of absolute differences (grid movement)

**Example (2D):**
```
Point A = (1, 2)
Point B = (4, 6)
d = |4-1| + |6-2| = 3 + 4 = 7
```

**When Use:**
- Sparse high-dimensional data (text)
- When feature differences more important than joint distance

---

### 2.3 Minkowski Distance (General Form)

**Formula:**
```
d(x_i, x_j) = (Σ|x_ik - x_jk|^p)^(1/p)
```

**Cases:**
- p=1: Manhattan distance
- p=2: Euclidean distance
- p=∞: Chebyshev (max difference)

---

### 2.4 Cosine Distance (Angular)

**Formula:**
```
cosine_distance = 1 - (A·B) / (||A|| × ||B||)
```

Where A·B = dot product

**Interpretation:** Angle between vectors (not magnitude)

**Range:** [0, 2] (0=identical direction, 1=orthogonal)

**When Use:**
- Text classification (bag-of-words)
- High-dimensional sparse data
- Direction matters, magnitude doesn't

**Advantage:** Invariant to magnitude (normalized data)

---

### 2.5 Hamming Distance

**Formula:**
```
d(x_i, x_j) = # positions where values differ
```

**Example:**
```
String A = "kitten"
String B = "sitting"
Differences: positions 0(k vs s), 2(t vs t), 4(e vs i) = 3
```

**When Use:**
- Categorical data
- Binary strings
- String matching

---

### 2.6 Jaccard Distance

**Formula:**
```
Jaccard = 1 - |A ∩ B| / |A ∪ B|
```

**Interpretation:** Dissimilarity of sets

**When Use:**
- Set-based data
- Binary features (presence/absence)
- Recommendation systems

---

### 2.7 Distance Metric Selection

| Distance | Data Type | Dimensions | Sparse | Speed |
|----------|-----------|-----------|--------|-------|
| Euclidean | Numeric | Low-Medium | No | Fast |
| Manhattan | Numeric | Medium-High | Yes | Fast |
| Chebyshev | Numeric | Any | No | Very Fast |
| Cosine | Any | High | Yes | Medium |
| Hamming | Categorical | Any | Yes | Fast |
| Jaccard | Sets | Any | Yes | Medium |

---

## 3. Feature Scaling & Normalization

### 3.1 Why Scale Features?

**Problem:** Features different scales → Distance dominated by large-scale features

**Example:**
```
Age (18-80) vs Income (10k-500k)
Euclidean: d ≈ √((age_diff)² + (income_diff)²)
Income dominates (0-490k range) vs Age (0-62 range)
Age differences ignored!
```

---

### 3.2 Standardization (Z-score)

**Formula:**
```
x_standardized = (x - mean) / std
```

**Result:** Zero mean, unit variance

**Code:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

### 3.3 Normalization (Min-Max)

**Formula:**
```
x_normalized = (x - min) / (max - min)
```

**Result:** Range [0, 1]

**Code:**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

---

### 3.4 Important: Scale Before Prediction

**Always:**
```python
# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training scaler!

model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# WRONG
X_test_scaled_wrong = scaler.fit_transform(X_test)  # Data leakage!
```

---

## 4. Choosing k

### 4.1 Impact of k

**Small k (k=1):**
- Follows noise closely (overfitting)
- Decision boundary complex
- High training accuracy, low test accuracy
- Sensitive to outliers

**Large k (k=n):**
- Ignores local structure (underfitting)
- Decision boundary smooth
- High bias, low variance
- Ignores minority classes

**Optimal k:**
- Balance: Captures patterns, smooth boundary
- Typical: k=3 to 10 (rule of thumb: k ≈ √n)

---

### 4.2 k Selection Methods

**Cross-Validation:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

k_values = list(range(1, 31))
param_grid = {'n_neighbors': k_values}

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best k: {grid.best_params_['n_neighbors']}")
print(f"Best CV Score: {grid.best_score_}")
```

**Elbow Method:**
- Plot validation error vs k
- Choose k at "elbow" (improvement plateaus)

**Domain Knowledge:**
- Imbalanced: Smaller k helps minority
- Noisy: Larger k more robust
- Computational: Larger k slower

---

### 4.3 Odd vs Even k

**k Odd (for Binary Classification):**
- Avoids ties (k=3: either 2-1 or 1-2, never 1.5-1.5)
- Recommended for binary

**k Even:**
- Can tie (k=4: 2-2 tie possible)
- Tie-breaking: Random or distance-weighted

**Rule:** Use odd k for binary classification

---

## 5. Distance-Weighted k-NN

### 5.1 Standard k-NN (Uniform Weights)

**Rule:**
```
Predict class = Majority among k neighbors
```

**Problem:** All k neighbors weighted equally
- Neighbor at distance 0.1 vs 10.0 same weight

**Example:**
```
k=3, neighbors: [Class 1 at dist 0.1, Class 1 at dist 0.2, Class 2 at dist 100]
Prediction: Class 1 (2 votes)
But Class 2 neighbor very far!
```

---

### 5.2 Distance-Weighted k-NN

**Rule:**
```
Weight_i = 1 / distance_i  (or 1 / distance_i²)
Weighted_vote = Σ weight_i × class_i / Σ weight_i
```

**Effect:**
- Closer neighbors more influential
- Handles distance imbalance
- More informative

**Example (Regression):**
```
Neighbors: [value=5 at dist 0.1, value=3 at dist 10]
Uniform k-NN: (5 + 3) / 2 = 4
Distance-weighted: (5×10 + 3×0.1) / (10 + 0.1) ≈ 4.98 (closer neighbor dominates)
```

**Code:**
```python
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

**Variants:**
- `weights='uniform'`: Equal weights (default)
- `weights='distance'`: Inverse distance
- Custom function: `weights=lambda distances: 1 / (distances + 1)`

---

## 6. Computational Considerations

### 6.1 KD-Tree (k-dimensional tree)

**Idea:** Hierarchical spatial data structure

**Structure:**
- Recursively partition space
- Each node: Feature and threshold
- Branches: Left (≤ threshold), Right (> threshold)

**Advantage:** O(log n) nearest neighbor search (vs O(n) naive)

**Disadvantage:** Slow with high-dimensional data (curse of dimensionality)

**When Use:**
- Low to medium dimensions (< 20)
- Many queries
- Static data

**Code:**
```python
knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
```

---

### 6.2 Ball Tree

**Idea:** Tree of nested balls (hyperspheres)

**Advantage:** Better for high dimensions than KD-tree

**When Use:**
- Medium to high dimensions
- Many queries

**Code:**
```python
knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
```

---

### 6.3 Brute Force

**Idea:** Compute distances to all training points

**Advantage:**
- Works any metric
- No preprocessing needed

**Disadvantage:** O(n) per query (slow for large n)

**When Use:**
- Small datasets (n < 1000)
- Custom distance metrics
- Few predictions

**Code:**
```python
knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
```

---

### 6.4 Computational Complexity

| Algorithm | Build Time | Query Time | Memory |
|-----------|-----------|-----------|--------|
| Brute Force | O(n) | O(nd) | O(n) |
| KD-Tree | O(n log n) | O(d log n)* | O(n) |
| Ball Tree | O(n log n) | O(d log n)* | O(n) |

*High dimensions: Approaches O(n)

---

## 7. k-NN for Regression

### 7.1 Regression Predictions

**Unweighted Average:**
```
ŷ = (1/k) × Σ y_neighbors
```

**Distance-Weighted Average:**
```
ŷ = Σ(weight_i × y_i) / Σ weight_i
```

Where weight_i = 1 / distance_i

**Example:**
```
Neighbors: y = [10, 12, 8]
Unweighted: (10 + 12 + 8) / 3 = 10
Distance-weighted: If closer to 12 → Might predict 11.5
```

---

### 7.2 Regression Evaluation

**Metrics:**
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

**Code:**
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

---

## 8. Advantages of k-NN

### 8.1 Simplicity & Interpretability

**Easy to Understand:**
- Simple algorithm: Find k nearest, vote
- Easy to explain: "Neighbors have similar labels"

**Interpretable Predictions:**
- Can show which neighbors influenced prediction
- No hidden parameters

---

### 8.2 No Training Phase

**Fast Training:**
- Just memorize data: O(n)
- No parameter optimization needed

**Dynamic Updates:**
- Add new samples without retraining
- Instant updates (for production: risky)

---

### 8.3 Works with Any Distance Metric

**Flexibility:**
- Euclidean, Manhattan, Cosine, Hamming, etc.
- Can define custom metrics
- Adapts to data type

---

### 8.4 Non-parametric, No Assumptions

**Flexible:**
- No linearity assumption (like LR)
- No distribution assumption (like GD)
- Works with complex patterns

---

### 8.5 Multi-class & Multi-output

**Works Naturally:**
- Binary, multiclass, multilabel
- Regression & classification
- Single implementation

---

## 9. Disadvantages of k-NN

### 9.1 Curse of Dimensionality

**Problem:** High dimensions → All distances similar

**Why:** In 1000D space, all points roughly equidistant
- Concept: Distance concentrates around mean distance
- Effect: k-NN becomes uninformative (all points equally "nearest")

**Example:**
```
Randomly distributed points in d dimensions
Average distance to k-th nearest: Ω(n^(1/d))

d=2: Distance ∝ √n ≈ √1000 ≈ 31
d=1000: Distance ∝ n^(1/1000) ≈ 1 (nearly 0)
All distances converge!
```

**Solutions:**
- Feature selection (reduce d)
- PCA (dimensionality reduction)
- Feature engineering
- Skip k-NN for very high-d

---

### 9.2 Computational Cost

**Training:** Fast (O(n))

**Prediction:** Slow (O(nd) naive, O(d log n) with tree)

**Problem:** Every prediction computes k distances
- Not practical for millions of test samples
- KD-tree helps but still O(n) in worst case

**Large Scale:**
- Approximate nearest neighbors (Locality-Sensitive Hashing)
- Indexing structures
- Vector databases

---

### 9.3 Sensitive to Feature Scaling

**Problem:** Different scales → Distance dominated by large-scale features

**Solution:** ALWAYS scale features

**Mistakes:**
```python
# WRONG
model = KNeighborsClassifier()
model.fit(X_train, y_train)  # Unscaled!
predictions = model.predict(X_test)  # Results biased

# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
```

---

### 9.4 Sensitive to Irrelevant Features

**Problem:** Irrelevant features add noise to distances

**Example:**
```
Features: [age, income, random_noise]
Distance: √(age_diff² + income_diff² + noise_diff²)
Random noise increases all distances equally → Less discriminative
```

**Solutions:**
- Feature selection (remove irrelevant)
- Feature weighting (weight important features)
- Dimensionality reduction

---

### 9.5 Memory Intensive

**Problem:** Store entire training dataset

**Size:** O(n) memory for n samples

**Issue:** 
- Millions of samples → GB of memory
- Slow loading, slow queries

**Solutions:**
- Approximate methods (LSH)
- Prototype selection (keep representative samples)
- Condensed Nearest Neighbor

---

### 9.6 Imbalanced Classes Problem

**Problem:** k-NN majority voting doesn't account for class imbalance

**Example:**
```
k=3, Training data: 99% class A, 1% class B
99% chance all 3 neighbors are class A
→ Always predict A (even for class B samples)
```

**Solutions:**
- Distance-weighted voting (nearby B closer than A)
- k adjustment
- Class weights
- Resampling before training

---

### 9.7 Poor Performance with Noisy Data

**Problem:** Noise in neighbors affects prediction

**k=1:** Any noisy sample becomes neighbor → Bad prediction

**Large k:** Averages over noisy samples (but may over-smooth)

**Solutions:**
- Outlier removal
- Distance weighting (reduce noise influence)
- Larger k for robustness

---

## 10. k-NN Variants

### 10.1 Radius-Based Neighbors

**Idea:** All neighbors within radius r (not exactly k)

**Prediction:** Average/vote over all neighbors in radius

**Code:**
```python
from sklearn.neighbors import RadiusNeighborsClassifier

rnc = RadiusNeighborsClassifier(radius=1.0)
rnc.fit(X_train, y_train)
predictions = rnc.predict(X_test)
```

**Advantage:** Adaptive number of neighbors

**Disadvantage:** Some samples might have 0 neighbors (prediction fails)

---

### 10.2 Condensed Nearest Neighbor (CNN)

**Idea:** Select subset of training data (prototypes)

**Purpose:** Reduce memory, speed up prediction

**Algorithm:**
1. Start with one sample
2. Add samples misclassified by current set
3. Stop when all samples correctly classified

**Trade-off:** Less memory/computation vs potential accuracy loss

---

### 10.3 Edited Nearest Neighbor (ENN)

**Idea:** Remove noisy samples

**Algorithm:**
1. For each sample, check if misclassified by k neighbors
2. Remove if misclassified
3. Repeat until stable

**Effect:** Cleaner decision boundaries, handles noise

---

## 11. Comparison with Other Algorithms

| Aspect | k-NN | Decision Tree | Linear Model | SVM |
|--------|------|-------------|-------------|-----|
| Training Time | O(n) | O(n log n) | O(np) | O(n²p) |
| Prediction Time | O(nd) | O(log n) | O(p) | O(p) |
| Interpretability | Medium | High | High | Low |
| Non-linear | Yes | Yes | No | Yes (kernel) |
| Feature Scaling | Critical | Not needed | Needed | Needed |
| Multiclass | Native | Native | Softmax | OvR |
| Memory | O(n) | O(tree) | O(p) | O(m) |

---

## 12. Implementation in sklearn

### 12.1 Classification

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)
y_pred_proba = knn.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(knn, X_train, y_train, cv=5).mean()}")
print(f"\n{classification_report(y_test, y_pred)}")
```

### 12.2 Regression

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)

y_pred = knn_reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### 12.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'kd_tree', 'ball_tree'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Test Accuracy: {grid.best_estimator_.score(X_test, y_test)}")
```

### 12.4 Distance Metrics

```python
# Default Euclidean
knn = KNeighborsClassifier(metric='euclidean')

# Manhattan
knn = KNeighborsClassifier(metric='manhattan')

# Cosine
knn = KNeighborsClassifier(metric='cosine')

# Custom metric
def custom_metric(x, y):
    return np.sum(np.abs(x - y))  # Manhattan

knn = KNeighborsClassifier(metric=custom_metric)
```

---

## 13. Common Challenges & Solutions

### 13.1 High Dimensionality

**Problem:** Curse of dimensionality

**Solutions:**
```python
# 1. Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 2. PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# 3. Feature engineering
# Create meaningful features from domain knowledge
```

---

### 13.2 Imbalanced Classes

**Problem:** Majority class dominates

**Solutions:**
```python
# 1. Distance-weighted voting
knn = KNeighborsClassifier(weights='distance')

# 2. Adjust k
knn = KNeighborsClassifier(n_neighbors=1)  # Smaller k focuses on nearby

# 3. Resampling before training
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
knn.fit(X_balanced, y_balanced)
```

---

### 13.3 Slow Predictions

**Problem:** O(nd) per prediction

**Solutions:**
```python
# 1. KD-tree (if low dimensions)
knn = KNeighborsClassifier(algorithm='kd_tree')

# 2. Feature reduction (PCA, selection)
# Fewer features → Faster distances

# 3. Approximate methods
# Use libraries: FAISS, Annoy, Scann for large scale

# 4. Prototype selection
# Keep only representative samples
from sklearn.neighbors import NearestNeighbors
# Use Condensed Nearest Neighbor
```

---

## 14. Practice Problems

1. **Distance Metrics:** When use Euclidean vs Manhattan vs Cosine?

2. **Feature Scaling:** Why critical for k-NN? What happens without scaling?

3. **k Selection:** k=1 vs k=n. Bias-variance trade-off?

4. **Curse of Dimensionality:** Why high-d breaks k-NN? How mitigate?

5. **Distance Weighting:** Advantage over uniform weights?

6. **Computational Complexity:** Naive vs KD-tree. When each?

7. **Imbalance:** 1% class B, 99% class A. How adjust?

8. **Regression:** How predict continuous with k-NN? Evaluation metric?

---

## 15. Key Takeaways

1. **Core idea:** Majority vote among k nearest neighbors
2. **Distance metric:** Euclidean (default), Manhattan, Cosine (text)
3. **Feature scaling:** CRITICAL (always scale before)
4. **k selection:** Balance (k≈√n typical, tune via CV)
5. **Distance weighting:** Closer neighbors more influential (often better)
6. **Computational:** Fast training O(n), slow prediction O(nd)
7. **Curse of dimensionality:** Breaks in high-d (feature reduction helps)
8. **Advantages:** Simple, interpretable, non-parametric, flexible
9. **Disadvantages:** Memory-intensive, slow, sensitive to features, noisy
10. **Use when:** Low-d, need interpretability, small-medium data

---

**Next Topic:** Naive Bayes (say "next" to continue)