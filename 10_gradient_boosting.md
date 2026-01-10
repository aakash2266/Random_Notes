# Gradient Boosting - Complete Guide

## 1. Fundamentals of Gradient Boosting

### 1.1 What is Gradient Boosting?

**Definition:** Sequential ensemble algorithm training weak learners to predict residuals of previous learner, reducing loss via gradient descent.

**Core Idea:**
```
"Correct mistakes with gradients" - Each tree learns from residuals of previous
```

**Key Characteristics:**
- Sequential training (like AdaBoost)
- Fits residuals (unlike AdaBoost which upweights)
- Gradient-based optimization (minimize loss function)
- Reduces bias and variance
- Slower but often more accurate than Random Forest

---

### 1.2 Gradient Boosting vs AdaBoost

| Aspect | Gradient Boosting | AdaBoost |
|--------|------------------|----------|
| Focus | Residuals | Misclassified samples |
| Optimization | Gradient descent (loss) | Exponential (error rate) |
| Weak Learner | Shallow trees | Stumps (depth=1) |
| Learning | Fits residuals | Upweights mistakes |
| Loss Function | Flexible (any loss) | Exponential loss |
| Regularization | Built-in (shrinkage) | Via parameters |
| Accuracy | Often higher | Good |
| Computational | Slower | Slower |

---

### 1.3 Boosting Methods Summary

```
Bagging (Random Forest):
  - Bootstrap samples
  - Parallel independent trees
  - Reduces variance

AdaBoost:
  - Weighted samples (upweight mistakes)
  - Sequential
  - Reduces bias via sample reweighting

Gradient Boosting:
  - Fit residuals
  - Sequential
  - Reduces bias via gradient descent
  - More flexible (any loss function)
```

---

## 2. Gradient Boosting Algorithm (Regression)

### 2.1 High-Level Process

```
1. Initialize: Predict constant (e.g., mean)
2. For iteration t = 1 to T:
   a. Calculate residuals: r_i = y_i - ŷ_i
   b. Fit weak learner (tree) to residuals
   c. Predictions: h_t(x) predicts residuals
   d. Update: ŷ_new = ŷ_old + learning_rate × h_t(x)
3. Final: Sum of all tree predictions
```

---

### 2.2 Detailed Algorithm (Regression)

**Initialization:**
```
F_0(x) = argmin_c Σ L(y_i, c)

For regression with MSE loss:
F_0(x) = mean(y)  (constant mean prediction)

For regression with MAE loss:
F_0(x) = median(y)  (constant median prediction)
```

**For each iteration t = 1 to T:**

**Step 1: Calculate Residuals (Negative Gradient)**
```
r_it = -∂L(y_i, F_{t-1}(x_i)) / ∂F_{t-1}(x_i)

For MSE loss L = (y - F)²:
r_it = y_i - F_{t-1}(x_i)  (standard residuals)

For other losses:
r_it = gradient of loss
```

**Step 2: Fit Weak Learner to Residuals**
```
Train tree h_t(x) to predict residuals:
h_t = argmin_h Σ (r_it - h(x_i))²

Standard regression tree on residual values
```

**Step 3: Calculate Optimal Step Size (Line Search)**
```
γ_t = argmin_γ Σ L(y_i, F_{t-1}(x_i) + γ × h_t(x_i))

For MSE: γ_t = learning_rate (fixed, typically 0.01-0.1)
For other losses: May optimize γ
```

**Step 4: Update Predictions**
```
F_t(x) = F_{t-1}(x) + learning_rate × h_t(x)

Where:
- F_{t-1}(x): Previous cumulative prediction
- h_t(x): New tree prediction (of residuals)
- learning_rate: Shrinkage (0 < lr ≤ 1)
```

---

### 2.3 Example: Simple Regression

**Data:**
```
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
```

**Iteration 0 (Initialize):**
```
F_0(x) = mean(y) = (2+4+5+4+5)/5 = 4
Predictions: [4, 4, 4, 4, 4]
```

**Iteration 1:**
```
Residuals: r = y - F_0 = [2-4, 4-4, 5-4, 4-4, 5-4] = [-2, 0, 1, 0, 1]

Fit tree h_1 to residuals:
Possible split: x ≤ 2 (leaf = -1, right leaf = 0.67)
Predictions: h_1(x) = [-1, -1, 0.67, 0.67, 0.67]

Update (learning_rate = 0.1):
F_1(x) = F_0(x) + 0.1 × h_1(x)
       = [4, 4, 4, 4, 4] + 0.1×[-1, -1, 0.67, 0.67, 0.67]
       = [3.9, 3.9, 4.067, 4.067, 4.067]

New residuals: [2-3.9, 4-3.9, 5-4.067, 4-4.067, 5-4.067] = [0.1, 0.1, 0.933, -0.067, 0.933]
```

**Iteration 2:** Repeat (fit tree to new residuals)

```
...continues...
```

---

### 2.4 Final Prediction

**For new sample x_new:**
```
ŷ = F_0(x_new) + learning_rate × h_1(x_new) + learning_rate × h_2(x_new) + ... + learning_rate × h_T(x_new)

ŷ = F_0(x_new) + learning_rate × Σ h_t(x_new)
```

---

## 3. Gradient Boosting for Classification

### 3.1 Loss Functions

**Binary Classification (Logistic Loss):**
```
L(y, p) = -y × log(p) - (1-y) × log(1-p)

Residual (negative gradient):
r_i = y_i - p_i

Where p_i = sigmoid(F_{t-1}(x_i))
```

**Multi-class (Log Loss):**
```
One tree per class (or multinomial)
Fits residual for each class
```

---

### 3.2 Predictions (Binary)

**After T iterations:**
```
Raw score: F(x) = F_0(x) + learning_rate × Σ h_t(x)

Probability: P(y=1|x) = sigmoid(F(x))

Class prediction: If P(y=1) > 0.5, predict 1, else 0
```

---

## 4. Loss Functions & Regularization

### 4.1 Common Loss Functions

**Regression:**
- **MSE (Mean Squared Error):** L = (y - ŷ)²
  - Default, penalizes large errors heavily
  
- **MAE (Mean Absolute Error):** L = |y - ŷ|
  - Robust to outliers
  
- **Huber Loss:** Combination (MSE near 0, MAE for outliers)
  - Robust but differentiable

**Classification:**
- **Log Loss (Logistic):** L = -y×log(p) - (1-y)×log(1-p)
  - Default for binary

- **Exponential Loss:** L = exp(-y×F(x))
  - AdaBoost loss

---

### 4.2 Learning Rate (Shrinkage)

**Parameter:** learning_rate (typically 0.01-0.1)

**Formula:**
```
F_t(x) = F_{t-1}(x) + learning_rate × h_t(x)
```

**Effect:**
- Lower learning_rate: Smaller steps, need more trees, better generalization
- Higher learning_rate: Faster convergence, fewer trees, overfitting risk

**Strategy:**
```
learning_rate = 0.1, n_estimators = 100
vs
learning_rate = 0.01, n_estimators = 1000

Similar final error, but second more robust
```

---

### 4.3 Subsampling (Stochastic Gradient Boosting)

**Idea:** Use random subset of data for each tree

```python
subsample = 0.8  # Use 80% of data
```

**Effect:**
- Adds randomness (like Random Forest)
- Reduces correlation between trees
- Faster training (fewer samples)
- Often better generalization

---

### 4.4 Column (Feature) Subsampling

**Idea:** Use random subset of features for each tree

```python
colsample_bytree = 0.8  # Use 80% of features per tree
colsample_bylevel = 0.8  # Use 80% per level
```

**Effect:**
- More diverse trees
- Reduce overfitting
- Faster training

---

## 5. Gradient Boosting Hyperparameters

### 5.1 Tree Parameters

**max_depth:**
- Depth of trees
- Typical: 3-8 (shallow)
- Deeper: More complexity, overfitting risk

**min_child_weight (or min_samples_leaf):**
- Minimum samples in leaf
- Typical: 1-10
- Higher: Simpler trees

**min_split_loss (or gamma):**
- Minimum loss reduction to split
- Typical: 0
- Higher: Fewer splits

---

### 5.2 Boosting Parameters

**n_estimators:**
- Number of trees
- Typical: 100-1000
- Monitor: Validation error plateaus

**learning_rate:**
- Shrinkage parameter
- Typical: 0.01-0.1
- Lower: More stable, need more trees

**subsample:**
- Row subsampling ratio
- Typical: 0.5-1.0
- Default: 1.0 (no subsampling)

**colsample:**
- Feature subsampling ratio
- Typical: 0.5-1.0
- Default: 1.0

---

### 5.3 Regularization Parameters

**Lambda (L2):**
- Ridge penalty on weights
- Typical: 1.0
- Higher: More regularization

**Alpha (L1):**
- Lasso penalty on weights
- Typical: 0
- Higher: More feature selection

**Gamma:**
- Minimum loss reduction
- Typical: 0
- Higher: Fewer splits (simpler tree)

---

## 6. Early Stopping

### 6.1 Problem: Overfitting

**Issue:** More trees → Eventually overfitting

```
Train error: Decreases monotonically
Test error: Decreases then increases (overfitting)
```

---

### 6.2 Early Stopping Solution

**Monitor validation error:**
```
1. During training, evaluate on validation set every N trees
2. If validation error increases for M consecutive evaluations: STOP
3. Use best model (before overfit)
```

**Effect:**
- Prevent overfitting
- Reduce training time
- Find optimal n_estimators automatically

**Code:**
```python
eval_set = [(X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10)
```

---

## 7. Feature Importance in Gradient Boosting

### 7.1 Gain-Based Importance

**Idea:** Sum gain from each feature across all trees

```
Importance(f) = Σ Gain_f / # trees
```

Where Gain = Loss reduction from split

**Interpretation:**
- Feature used in important splits → High importance
- More trees using feature → Higher importance

**Advantages:**
- Fast (computed during training)
- Captures feature contributions

---

### 7.2 Split-Based Importance

**Idea:** Count how often feature used for splitting

```
Importance(f) = # splits using f / # total splits
```

**Interpretation:**
- Frequency of use in trees

---

### 7.3 Permutation Importance

**Idea:** Shuffle feature, measure error increase

(Same as Random Forest)

---

## 8. Advantages of Gradient Boosting

### 8.1 High Accuracy

**Empirical:** Often best performance on tabular data

**Why:**
- Fits residuals (corrects mistakes)
- Gradient-based optimization (principled)
- Flexible loss functions
- Built-in regularization (learning_rate)

---

### 8.2 Handles Non-linearity

**Automatic:**
- Trees capture non-linear patterns
- Interactions learned

---

### 8.3 Feature Interactions

**Automatic:**
- Trees naturally discover interactions

---

### 8.4 Mixed Feature Types

**Native Support:**
- Numeric and categorical
- Handles missing (built-in)

---

### 8.5 Robust Regularization

**Built-in:**
- Learning rate (shrinkage)
- Tree depth limits
- Subsampling
- L1/L2 penalties

---

## 9. Disadvantages of Gradient Boosting

### 9.1 Slow Training

**Problem:**
- Sequential (can't parallelize)
- Tree fitting + residual calculation
- Many iterations (n_estimators = 100-1000)

**Time:** Hours for large datasets

---

### 9.2 Hard to Tune

**Many Parameters:**
- learning_rate, n_estimators
- max_depth, min_child_weight
- subsample, colsample
- L1/L2 penalties
- Loss function

**Large Search Space:** Tuning expensive

---

### 9.3 Overfitting Risk

**Problem:**
- If n_estimators too high
- If learning_rate too high
- If trees too deep
- If regularization too weak

**Solution:** Early stopping, CV tuning

---

### 9.4 Black Box

**Problem:**
- Ensemble of trees hard to interpret
- Feature importance only insight

---

### 9.5 Memory Intensive

**Problem:**
- Store all trees
- Large datasets problematic

---

## 10. Popular Gradient Boosting Implementations

### 10.1 Sklearn GradientBoostingClassifier

**Pros:**
- Simple, sklearn integration
- Good for learning

**Cons:**
- Slower than specialized
- Fewer features

**Code:**
```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)
```

---

### 10.2 XGBoost

**Advantages:**
- Fast (optimized C++ backend)
- GPU support
- Parallel tree building
- Handles missing natively
- Built-in cross-validation
- Feature importance (gain, split, cover)

**Popular:**
- Industry standard
- Kaggle competitions

**Code:**
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
```

---

### 10.3 LightGBM (Light Gradient Boosting Machine)

**Advantages:**
- Even faster than XGBoost
- Lower memory usage
- Leaf-wise tree growth (vs level-wise)
- Handles categorical natively
- GPU support

**Best For:**
- Large datasets
- Speed priority

**Code:**
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_train, y_train)
```

---

### 10.4 CatBoost

**Advantages:**
- Best for categorical features (no encoding!)
- Automatic overfitting detection
- GPU support
- Ordered boosting (reduces overfitting)

**Best For:**
- Categorical-heavy data
- Small datasets

**Code:**
```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train)
```

---

### 10.5 Comparison: XGBoost vs LightGBM vs CatBoost

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Speed | Fast | Faster | Medium |
| Memory | Medium | Low | Low |
| Categorical | Requires encoding | Native | Native |
| Tuning | Hard | Medium | Easier |
| Large Data | Good | Best | Medium |
| Small Data | Good | Good | Best |
| Popularity | Highest | High | Growing |

---

## 11. Gradient Boosting vs Random Forest

| Aspect | Gradient Boosting | Random Forest |
|--------|------------------|--------------|
| Training | Sequential | Parallel |
| Accuracy | Often higher | Good |
| Speed | Slower | Faster |
| Tuning | Hard | Easy |
| Overfitting | Risk (needs monitoring) | Less risk |
| Regularization | Built-in (learning_rate) | Limited |
| Interpretability | Low | Medium |
| Feature Importance | Yes | Yes |
| When Use | Accuracy priority | Speed/simplicity |

---

## 12. Implementation in sklearn

### 12.1 Classification

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Train
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)

# Predict
y_pred = gb.predict(X_test)
y_pred_proba = gb.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(gb, X_train, y_train, cv=5).mean()}")
print(f"\n{classification_report(y_test, y_pred)}")
```

### 12.2 Regression

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_reg.fit(X_train, y_train)

y_pred = gb_reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### 12.3 Feature Importance

```python
# Feature importance
importance = gb.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(importance_df)
```

### 12.4 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

gb = GradientBoostingClassifier()
grid = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

---

## 13. XGBoost Implementation

### 13.1 Classification

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Train
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# Predict
y_pred = xgb_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### 13.2 With Early Stopping

```python
# Prepare validation set
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2
)

# Train with early stopping
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,  # High, will stop early
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,  # Stop if no improvement for 20 rounds
    verbose=10
)

print(f"Best iteration: {xgb_model.best_iteration}")
```

### 13.3 Feature Importance

```python
# Different importance types
importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
importance_split = xgb_model.get_booster().get_score(importance_type='split')

# Plot
xgb.plot_importance(xgb_model, importance_type='gain')
```

---

## 14. LightGBM Implementation

### 14.1 Classification

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### 14.2 Categorical Features

```python
# Specify categorical columns
categorical_features = ['color', 'brand']  # Column names

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)

# LightGBM handles categorical automatically
lgb_model.fit(
    X_train, y_train,
    categorical_feature=categorical_features
)
```

---

## 15. Common Issues & Solutions

### 15.1 Overfitting

**Symptoms:** Train accuracy 95%, test 70%

**Solutions:**
```python
# 1. Lower learning_rate (slower learning)
gb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500)

# 2. Limit tree depth
gb = GradientBoostingClassifier(max_depth=3)

# 3. Increase min_samples_split/leaf
gb = GradientBoostingClassifier(min_samples_split=10, min_samples_leaf=5)

# 4. Early stopping
# Use validation set, stop when validation error increases

# 5. Subsampling
gb = GradientBoostingClassifier(subsample=0.8)

# 6. Regularization
gb = GradientBoostingClassifier(l2_regularization=1.0)  # XGBoost
```

---

### 15.2 Underfitting

**Symptoms:** Both train and test low

**Solutions:**
```python
# 1. More trees
gb = GradientBoostingClassifier(n_estimators=500)

# 2. Higher learning rate
gb = GradientBoostingClassifier(learning_rate=0.1)

# 3. Deeper trees
gb = GradientBoostingClassifier(max_depth=7)

# 4. Lower min_samples
gb = GradientBoostingClassifier(min_samples_split=2, min_samples_leaf=1)
```

---

### 15.3 Slow Training

**Causes:**
- Large n_estimators
- Deep trees
- Large dataset

**Solutions:**
```python
# 1. Use XGBoost/LightGBM (faster)
import xgboost as xgb
model = xgb.XGBClassifier()

# 2. Reduce trees
gb = GradientBoostingClassifier(n_estimators=50)

# 3. Smaller trees
gb = GradientBoostingClassifier(max_depth=3)

# 4. Subsampling (fewer rows per tree)
gb = GradientBoostingClassifier(subsample=0.5)

# 5. GPU acceleration (XGBoost, LightGBM)
xgb_model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
```

---

## 16. Practice Problems

1. **Residuals:** What is gradient boosting fitting? Formula?

2. **Learning Rate:** Why 0.1 better than 1.0? Trade-off?

3. **Early Stopping:** How works? What metric monitor?

4. **Loss Functions:** MSE vs MAE. When each?

5. **XGBoost vs sklearn:** Differences? When each?

6. **Feature Importance:** Gain vs Split vs Cover?

7. **Hyperparameter Tuning:** Most important? Order?

8. **Overfitting:** Multiple ways prevent. Examples?

---

## 17. Key Takeaways

1. **Core idea:** Fit residuals sequentially with gradient descent
2. **Residual fitting:** Each tree learns from previous mistakes
3. **Learning rate:** Controls step size (lower = more stable)
4. **Loss function:** Flexible (MSE, MAE, Log, custom)
5. **Early stopping:** Prevent overfitting via validation monitoring
6. **Regularization:** Built-in (learning_rate, depth, subsampling)
7. **Hyperparameters:** Many (learning_rate, max_depth, subsample, colsample)
8. **Implementations:** sklearn (simple), XGBoost (fast), LightGBM (very fast), CatBoost (categorical)
9. **Advantages:** High accuracy, flexible, automatic interactions
10. **Disadvantages:** Slow training, hard to tune, overfitting risk
11. **When use:** Accuracy priority, tabular data, moderate dataset size

---

**Next Topic:** Model Evaluation & Metrics (say "next" to continue)