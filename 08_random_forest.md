# Random Forest - Complete Guide

## 1. Fundamentals of Random Forest

### 1.1 What is Random Forest?

**Definition:** Ensemble learning algorithm combining multiple decision trees with randomness to reduce variance and improve generalization.

**Core Idea:**
```
"Wisdom of crowds" - Multiple weak learners combine to strong learner
```

**Key Characteristics:**
- Ensemble of decision trees
- Randomness in: Data (bootstrap) & Features (random split)
- Parallel independent trees
- Aggregation: Voting (classification) or Averaging (regression)
- Reduces overfitting (relative to single tree)

---

### 1.2 From Single Tree to Forest

**Single Decision Tree:**
```
Overfitting risk: High (captures noise)
Variance: High (different data → different tree)
Bias: Low (can fit any pattern)
```

**Random Forest:**
```
Overfitting: Low (many trees average out noise)
Variance: Low (averaging reduces variance)
Bias: Similar (flexible models)
Generalization: Better
```

---

### 1.3 Ensemble Learning Principle

**Ensemble = Collection of Models**

**Why Works:**
```
- If errors uncorrelated: Averaging reduces variance
- If errors correlated: Ensemble no better than single model
→ Randomness ensures diverse errors
```

**Mathematical:**
```
Ensemble Error = Correlation × Variance + (1 - Correlation) × Variance
Lower correlation → Lower error
```

---

## 2. Bootstrap Aggregating (Bagging)

### 2.1 What is Bagging?

**Idea:** Train multiple models on random samples, aggregate predictions

**Process:**
```
1. Sample with replacement from training data (bootstrap sample)
2. Train model on sample
3. Repeat 1-2 for B iterations (e.g., B=100)
4. Aggregate: Vote (classification), Average (regression)
```

---

### 2.2 Bootstrap Samples

**Bootstrap Sample:**
- Sample n items with replacement from n items
- Some items repeat, some missing

**Key Property:**
```
Size = Original size
But different composition each iteration
```

**Example (n=5, B=3):**
```
Original: [A, B, C, D, E]

Sample 1: [A, A, C, D, E]   (A twice, B missing)
Sample 2: [B, C, C, D, D]   (D twice, A,E missing)
Sample 3: [A, B, C, E, E]   (E twice, D missing)
```

---

### 2.3 Out-of-Bag (OOB) Samples

**Observation:**
```
Each bootstrap sample: ~63.2% original data
Missing: ~36.8% original data
```

**Out-of-Bag (OOB) Samples:**
- Samples not in bootstrap
- Can test model without separate test set
- Reduces computational cost

**OOB Score:**
```
Average accuracy on OOB samples across all trees
Unbiased estimate of generalization error
```

---

### 2.4 Why Randomness Reduces Variance

**Intuition:**
```
One tree: High variance (overfits to peculiarities)
Many trees on different samples: Errors uncorrelated
Averaging: Variance reduces by factor of 1/B
```

**Formula:**
```
Var(Average of trees) = Var(single tree) / B
If trees independent (bootstrapped differently)
```

---

## 3. Feature Randomness

### 3.1 Random Feature Selection

**Idea:** Each split considers only random subset of features

**Process (at each node):**
```
1. Consider only m features (m < p total)
2. Choose best split among m
3. Typically: m = √p (classification), m = p/3 (regression)
```

**Effect:**
- Reduces correlation between trees
- Ensures diverse trees (not all using same top features)
- Further variance reduction

---

### 3.2 Why Random Features?

**Problem (without random features):**
```
All trees use same important features
→ Correlated trees
→ Averaging doesn't reduce variance much
```

**Solution (random features):**
```
Each tree uses different features (random)
→ Diverse trees
→ Uncorrelated errors
→ Averaging effectively reduces variance
```

**Example:**
```
Data: Age, Income, Education (Age most important)

Without random features:
Tree 1: Splits on Age (best), then Income
Tree 2: Splits on Age (best), then Education
...
All similar → Correlated errors

With random features (m=2, e.g.):
Tree 1: Choose from {Age, Income} → Age
Tree 2: Choose from {Income, Education} → Education
...
Different features → Diverse trees
```

---

### 3.3 max_features Parameter

**Typical Values:**
- Classification: m = √p (sqrt)
- Regression: m = p/3 (log2)
- p = total number of features

**Examples:**
```
p=100 features
Classification: m = 10
Regression: m = 33

p=1000 features (text)
Classification: m = 32
Regression: m = 333
```

**Effect:**
- Lower m: More randomness, more diverse trees, more bias
- Higher m: Less randomness, similar trees, less bias
- Tune via cross-validation

---

## 4. Random Forest Algorithm (Classification)

### 4.1 Training Algorithm

```
Input: Training data (X, y), B (number of trees)

For b = 1 to B:
  1. Generate bootstrap sample from (X, y)
  2. Grow decision tree on bootstrap sample
     For each node:
       a. Randomly select m features
       b. Find best split among m features
       c. Split node
     Until: Pure or max_depth or min_samples_leaf
  3. Store tree
  
Output: Ensemble of B trees
```

---

### 4.2 Prediction (Classification)

**For New Sample x:**
```
1. Pass x through all B trees
2. Each tree outputs class prediction (0 or 1)
3. Aggregate: Majority vote
4. Predict = Most common class

Example (Binary):
Tree 1: Predict 1
Tree 2: Predict 0
Tree 3: Predict 1
Tree 4: Predict 1
Tree 5: Predict 0
Majority: Class 1 (3 out of 5)
```

**Probability:**
```
P(class 1) = # trees predicting 1 / Total trees
P(class 0) = # trees predicting 0 / Total trees
```

---

### 4.3 Prediction (Regression)

**For New Sample x:**
```
1. Pass x through all B trees
2. Each tree outputs numeric prediction
3. Aggregate: Average
4. Predict = Mean of all trees

Example:
Tree 1: Predict 50
Tree 2: Predict 45
Tree 3: Predict 52
Tree 4: Predict 48
Tree 5: Predict 49
Final: (50+45+52+48+49) / 5 = 48.8
```

---

## 5. Random Forest Hyperparameters

### 5.1 Number of Trees (n_estimators)

**Effect:**
- More trees: Better performance (diminishing returns)
- More trees: More computation

**Typical Range:** 100-1000
- 100: Fast, reasonable performance
- 500: Good balance
- 1000+: Marginal improvement

**When Stop Adding:**
- OOB error plateaus
- Cross-validation score plateaus

```python
# Monitor improvement
from sklearn.ensemble import RandomForestClassifier

oob_scores = []
for n in range(10, 501, 10):
    rf = RandomForestClassifier(n_estimators=n, oob_score=True)
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    
# Plot to find elbow
```

---

### 5.2 Tree Depth (max_depth)

**Effect:**
- None (unlimited): Full trees, risk overfitting
- Limited (e.g., 10): Simpler trees, more bias
- Typical: None or 10-20

**Strategy:**
- Start unlimited, limit if overfitting
- Smaller for high-dimensional data

---

### 5.3 Minimum Samples per Leaf (min_samples_leaf)

**Effect:**
- Low (1): Deep trees, overfitting risk
- High (20): Shallow trees, simpler model
- Typical: 1-5

**Use:**
- Reduce if overfitting
- Increase for noise robustness

---

### 5.4 Feature Randomness (max_features)

**Default:**
- Classification: 'sqrt' (√p)
- Regression: 'log2' (log₂p)

**Options:**
```python
max_features='sqrt'      # √p
max_features='log2'      # log₂p
max_features=0.5         # 50% of features
max_features=10          # 10 features
```

**Effect:**
- Lower: More randomness, more diverse trees
- Higher: More similar to single tree

---

### 5.5 Bootstrap (bootstrap parameter)

**Effect:**
```python
bootstrap=True   # Use bootstrap (default, bagging)
bootstrap=False  # Use entire dataset
```

**Why False?**
- More data per tree
- Less variance reduction benefit
- Rarely better than bootstrap

**Keep:** bootstrap=True (default)

---

## 6. Feature Importance in Random Forest

### 6.1 Gini-Based Importance

**Idea:** Sum Gini decrease from each feature across all trees

**Formula:**
```
Importance(f) = Σ(# trees using f) × Gini_decrease_f / # total splits
```

**Interpretation:**
- Feature used in many important splits → High importance
- Feature rarely used → Low importance
- Normalized to sum to 1.0

**Advantages:**
- Fast (computed during training)
- Built-in to sklearn

**Disadvantages:**
- Biased toward high-cardinality features
- Biased toward top-of-tree features (affect more samples)
- Can miss non-linear interactions

---

### 6.2 Permutation Importance

**Idea:** Shuffle feature, measure accuracy drop

**Process:**
```
1. Train model (full forest)
2. For each feature f:
   a. Shuffle feature f in test set
   b. Measure accuracy (drop indicates importance)
3. Importance = Accuracy drop
```

**Advantages:**
- Model-agnostic (works any model)
- No bias toward cardinality
- Captures interactions

**Disadvantages:**
- Slow (shuffle + predict for each feature)
- Can be negative (feature hurts model)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10)
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)
```

---

### 6.3 SHAP Values

**Idea:** Game-theoretic attribution (Shapley values)

**Advantages:**
- Theoretically sound
- Per-sample explanations
- Handles interactions

**Disadvantages:**
- Slow (exponential in features)
- Complex to understand

---

## 7. Advantages of Random Forest

### 7.1 Handles Non-linearity Automatically

**Advantage:**
- Captures non-linear patterns
- No manual feature engineering
- Complex interactions automatically

---

### 7.2 Feature Interactions

**Automatic:**
- Trees naturally discover interactions
- No need to specify x₁×x₂

---

### 7.3 Mixed Feature Types

**Native Support:**
- Numeric and categorical
- No encoding needed (sklearn handles automatically)
- Missing values (can handle with imputation)

---

### 7.4 Low Interpretability, High Performance

**Trade-off:**
- Less interpretable than single tree
- More interpretable than neural network
- Feature importance provides some insight

---

### 7.5 Robust to Outliers

**Why:**
- Trees use splits, not distances
- Outliers just another data point in subset
- Averaging reduces outlier impact

---

### 7.6 Parallel Training

**Advantage:**
- Independent trees (train in parallel)
- n_jobs=-1 uses all CPU cores
- Fast training for large datasets

---

### 7.7 Out-of-Bag Validation

**Advantage:**
- OOB score without separate test set
- Reduces computational cost
- Unbiased generalization estimate

---

## 8. Disadvantages of Random Forest

### 8.1 Black Box

**Problem:**
- Ensemble of trees hard to interpret
- Can't easily follow decision path
- "Why?" harder to answer

**Workaround:** Feature importance, SHAP values

---

### 8.2 Memory Intensive

**Problem:**
- Store B trees (100-1000)
- Each tree O(n) to store (for large n)
- Total: O(B×n)

**Example:**
```
1000 trees, 1M samples, 100 features
≈ 1GB+ memory (rough)
```

**Solution:**
- Reduce n_estimators
- Use simpler trees (limit depth)
- Sparse data more efficient

---

### 8.3 Slower Prediction

**Problem:**
- Must pass sample through all B trees
- Prediction time: O(B × tree_depth)

**Comparison:**
```
Single tree: Fast (O(log n) with optimal tree)
Random Forest: Slower (O(B × depth))
Typically: 100x slower prediction than single tree
```

**When Matters:** Real-time systems, mobile devices

---

### 8.4 Bias toward High-Cardinality Features

**Problem:**
- Features with many unique values used more
- Random feature selection: Categorical with 100 values likely selected

**Solution:**
- Feature engineering (group categories)
- Feature selection before training
- Permutation importance instead of Gini

---

### 8.5 Struggles with Imbalanced Data

**Problem:**
- Majority class dominates
- Random tree split favor majority
- Recall for minority poor

**Solutions:**
```python
# 1. Class weights
RandomForestClassifier(class_weight='balanced')

# 2. Resampling
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
rf.fit(X_balanced, y_balanced)

# 3. Adjust threshold
y_pred_proba = rf.predict_proba(X_test)[:, 1]
y_pred_custom = (y_pred_proba >= 0.3).astype(int)  # Lower threshold
```

---

### 8.6 Struggles with Rare Events

**Problem:**
- If event very rare (0.1% data)
- Single tree unlikely to see many examples
- Ensemble averages poor predictions

---

## 9. Random Forest for Regression

### 9.1 Differences from Classification

**Prediction:**
- Classification: Majority vote
- Regression: Average predictions

**Evaluation Metrics:**
- Classification: Accuracy, F1, AUC
- Regression: MSE, RMSE, R², MAE

**Parameters:** Same as classification

---

### 9.2 Regression Example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)

# Predict
y_pred = rf_reg.predict(X_test)

# Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R²: {r2_score(y_test, y_pred)}")

# Feature importance (same as classification)
print(rf_reg.feature_importances_)
```

---

## 10. Random Forest Variants

### 10.1 Extremely Randomized Trees (Extra Trees)

**Difference:**
- Feature threshold random (vs optimized)
- Feature split random (vs best)

**Effect:**
- Faster training (no optimization)
- More randomness
- Sometimes better generalization

**Code:**
```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100)
et.fit(X_train, y_train)
```

---

### 10.2 Gradient Boosting vs Random Forest

**Random Forest:**
- Bootstrap samples
- Parallel training
- Reduces variance

**Gradient Boosting:**
- Sequential trees
- Each tree corrects previous
- Reduces bias

---

## 11. Random Forest vs Other Models

| Aspect | Random Forest | Single Tree | SVM | Neural Network |
|--------|--------------|------------|-----|----------------|
| Non-linearity | Automatic | Yes | With kernel | Automatic |
| Interpretability | Medium | High | Low | Very Low |
| Training Speed | Medium | Fast | Slow | Very Slow |
| Prediction Speed | Medium | Very Fast | Medium | Medium |
| Memory | High | Low | Low | Medium |
| Feature Scaling | Not needed | Not needed | Critical | Recommended |
| Outliers | Robust | Not robust | Not robust | Not robust |
| Feature Importance | Yes | Yes | No | No |
| Parallelizable | Yes | No | No | Yes |

---

## 12. Implementation in sklearn

### 12.1 Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(rf, X_train, y_train, cv=5).mean()}")
print(f"OOB Score: {rf.oob_score_}")  # If oob_score=True
print(f"\n{classification_report(y_test, y_pred)}")
```

### 12.2 Regression

```python
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print(f"R²: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
```

### 12.3 Feature Importance

```python
# Gini importance
importance_gini = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance_gini
}).sort_values('Importance', ascending=False)

print(importance_df)

# Permutation importance
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10)
importance_perm = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

print(importance_perm)
```

### 12.4 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier()
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
print(f"Test accuracy: {grid.best_estimator_.score(X_test, y_test)}")
```

### 12.5 OOB Score

```python
# Enable OOB scoring
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)

print(f"OOB Score: {rf.oob_score_}")
# Unbiased estimate of test accuracy (no separate test set needed initially)
```

---

## 13. Common Issues & Solutions

### 13.1 Overfitting

**Symptoms:** Train accuracy 99%, test 70%

**Solutions:**
```python
# 1. Limit tree depth
rf = RandomForestClassifier(max_depth=10)

# 2. Increase min_samples_leaf
rf = RandomForestClassifier(min_samples_leaf=10)

# 3. Decrease max_features
rf = RandomForestClassifier(max_features='log2')

# 4. Increase min_samples_split
rf = RandomForestClassifier(min_samples_split=10)
```

---

### 13.2 Underfitting

**Symptoms:** Train and test both low

**Solutions:**
```python
# 1. More trees
rf = RandomForestClassifier(n_estimators=500)

# 2. Deeper trees
rf = RandomForestClassifier(max_depth=None)

# 3. Lower min_samples_leaf
rf = RandomForestClassifier(min_samples_leaf=1)

# 4. Higher max_features
rf = RandomForestClassifier(max_features='sqrt')
```

---

### 13.3 Class Imbalance

**Solutions:**
```python
# 1. Class weights
rf = RandomForestClassifier(class_weight='balanced')

# 2. SMOTE
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
rf.fit(X_balanced, y_balanced)

# 3. Threshold tuning
y_pred_proba = rf.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.3).astype(int)
```

---

## 14. Practice Problems

1. **Bootstrap:** Why random sampling with replacement? What's ~63.2%?

2. **Variance Reduction:** Why uncorrelated errors reduce variance?

3. **Feature Randomness:** Why random features matter? m = √p intuition?

4. **OOB Score:** What is it? How use as test score?

5. **Overfitting:** Deep trees vs shallow. Trade-off?

6. **Feature Importance:** Gini vs Permutation. Pros/cons?

7. **Imbalance:** Problem and solutions?

8. **Hyperparameters:** Which most important? Typical tuning order?

---

## 15. Key Takeaways

1. **Ensemble:** Multiple diverse trees aggregate predictions
2. **Bootstrap:** Random sampling with replacement
3. **Feature randomness:** Each split considers random features (ensures diversity)
4. **Voting:** Classification uses majority; regression uses averaging
5. **Variance reduction:** Uncorrelated trees reduce variance
6. **OOB score:** Unbiased test estimate without separate set
7. **Feature importance:** Gini-based (fast) or Permutation-based (robust)
8. **Hyperparameters:** n_estimators, max_depth, min_samples_leaf, max_features
9. **Advantages:** Automatic non-linearity, robust, parallel training, interpretable
10. **Disadvantages:** Memory, prediction speed, black box, imbalance struggles

---

**Next Topic:** AdaBoost (say "next" to continue)