# AdaBoost (Adaptive Boosting) - Complete Guide

## 1. Fundamentals of AdaBoost

### 1.1 What is AdaBoost?

**Definition:** Ensemble learning algorithm sequentially training weak learners, with each learner focusing on mistakes of previous learners.

**Core Idea:**
```
"Learn from mistakes" - Focus on hard samples, upweight them, retrain
```

**Key Characteristics:**
- Sequential training (not parallel like Random Forest)
- Adaptive weights: Focus on misclassified samples
- Weak learners (typically shallow trees: stumps, depth=1)
- Boosting: Each tree corrects previous
- Reduces bias (vs Bagging reduces variance)

---

### 1.2 Boosting vs Bagging

| Aspect | Boosting (AdaBoost) | Bagging (Random Forest) |
|--------|-------------------|----------------------|
| Training | Sequential | Parallel |
| Sample Selection | Weighted (adaptive) | Random (uniform) |
| Learner Focus | Mistakes of previous | Independent |
| Weak Learner | Shallow trees (stumps) | Deep trees |
| Reduces | Bias | Variance |
| Correlation | High (builds on previous) | Low (independent) |
| Overfitting | More risk | Less risk |
| Speed | Slower (sequential) | Faster (parallel) |

---

### 1.3 AdaBoost vs Logistic Regression

| Aspect | AdaBoost | Logistic Regression |
|--------|----------|-------------------|
| Type | Ensemble | Single Model |
| Non-linearity | Automatic (trees) | No (linear) |
| Interpretability | Low | High |
| Training | Iterative | Convex optimization |
| Parameters | Learner weights | Linear weights |
| Overfitting | Risk (needs tuning) | Low |

---

## 2. AdaBoost Algorithm (Classification)

### 2.1 High-Level Process

```
1. Initialize sample weights: w_i = 1/n (equal)
2. For iteration t = 1 to T:
   a. Train weak learner on weighted data
   b. Calculate error rate
   c. Calculate learner weight (alpha)
   d. Update sample weights (focus on misclassified)
3. Aggregate predictions with learner weights
```

---

### 2.2 Detailed Algorithm

**Initialization:**
```
Sample weights: w_i^(1) = 1/n for all samples
(All samples equally important initially)
```

**For each iteration t = 1 to T:**

**Step 1: Train Weak Learner**
```
Train classifier h_t on samples with weights w_i^(t)
(Misweighted samples are harder to classify → learner focuses on them)
```

**Step 2: Calculate Error Rate**
```
Error_t = Σ w_i^(t) × [h_t(x_i) ≠ y_i]

Where:
- [h_t(x_i) ≠ y_i] = 1 if misclassified, 0 if correct
- Weighted sum: Penalizes mistakes on important samples
```

**Step 3: Calculate Learner Weight (Alpha)**
```
Alpha_t = (1/2) × ln((1 - Error_t) / Error_t)

Or equivalently:
Alpha_t = ln(1/β_t) where β_t = Error_t / (1 - Error_t)
```

**Interpretation:**
```
Error_t = 0.5: Alpha_t = 0 (no skill, weight 0)
Error_t < 0.5: Alpha_t > 0 (good, positive weight)
Error_t = 0.0: Alpha_t = ∞ (perfect, very high weight)
Error_t > 0.5: Alpha_t < 0 (worse than random, negative weight)
```

**Step 4: Update Sample Weights**
```
w_i^(t+1) = w_i^(t) × exp(-Alpha_t × y_i × h_t(x_i))

Simplified:
If correct (h_t = y_i): w_i^(t+1) = w_i^(t) × exp(-Alpha_t)
If wrong (h_t ≠ y_i): w_i^(t+1) = w_i^(t) × exp(Alpha_t)
```

**Effect:**
- Correct predictions: Decrease weight (less important)
- Misclassified: Increase weight (more important)

**Normalization:**
```
w_i^(t+1) = w_i^(t+1) / Σ w_j^(t+1)
(Ensure weights sum to 1)
```

---

### 2.3 Final Prediction

**For New Sample x:**
```
y_pred = sign(Σ Alpha_t × h_t(x))

Where:
- Σ: Sum over all T weak learners
- Alpha_t: Weight of learner t (higher confidence → higher weight)
- h_t(x): Prediction of learner t (+1 or -1)
- sign: Final prediction based on sign of sum
```

**Probability:**
```
P(y=1|x) = 1 / (1 + exp(-2 × Σ Alpha_t × h_t(x)))
(Logistic transform to [0,1])
```

---

## 3. Example: AdaBoost Walkthrough

### 3.1 Simple Dataset

```
Data (4 samples):
x1=1, y1=+1
x2=2, y2=+1
x3=8, y3=-1
x4=9, y4=-1
```

---

### 3.2 Iteration 1

**Initialize weights:**
```
w = [0.25, 0.25, 0.25, 0.25]  (equal)
```

**Train weak learner (decision stump):**
```
Try splits: x ≤ 1.5, x ≤ 3, x ≤ 7, x ≤ 8.5, ...
Best split: x ≤ 5
- Left (x ≤ 5): Predict +1 (samples 1,2)
- Right (x > 5): Predict -1 (samples 3,4)
Predictions: [+1, +1, -1, -1]
```

**Calculate error:**
```
All correct!
Error = 0
Alpha = ln((1-0)/0) = ∞
(Perfect learner, infinite weight)
```

**Update weights:**
```
All correct → w *= exp(-∞) ≈ 0
w_new = [0, 0, 0, 0] → Normalize ERROR

In practice: If perfect, usually add minimum error
```

---

### 3.2 Realistic Iteration 1 (with noise)

**Weak learner predictions:** [+1, +1, -1, +1]  (misclassifies sample 4)

**Error:**
```
Error = 0.25 × 1 = 0.25 (only sample 4 wrong)
Alpha = (1/2) × ln((1-0.25)/0.25) = (1/2) × ln(3) ≈ 0.55
```

**Update weights:**
```
Sample 1: w=0.25 × exp(-0.55×(+1)×(+1)) = 0.25 × exp(-0.55) ≈ 0.14
Sample 2: w=0.25 × exp(-0.55) ≈ 0.14
Sample 3: w=0.25 × exp(-0.55) ≈ 0.14
Sample 4: w=0.25 × exp(-0.55×(-1)×(+1)) = 0.25 × exp(0.55) ≈ 0.41

Normalize: [0.14, 0.14, 0.14, 0.41] / 0.83 ≈ [0.17, 0.17, 0.17, 0.49]
```

**Effect:** Sample 4 (misclassified) weight increased to 0.49!

---

### 3.3 Iteration 2

**Train on weighted data:**
```
Weights: [0.17, 0.17, 0.17, 0.49]
Sample 4 much more important (weight 0.49)
Learner focuses on correctly classifying sample 4

Possible learner:
Try splits focusing on x > 8
Predictions: [+1, +1, -1, -1]  (all correct)
Error = 0
Alpha = ∞ (or large)
```

**Update weights:**
```
Weights decrease for correct, increase for wrong
This process repeats...
```

---

### 3.4 Final Prediction

**After T=2 iterations:**
```
Alpha_1 = 0.55
Alpha_2 = 0.70  (example)

For sample x=5:
h_1(5) = +1 (left side of first split)
h_2(5) = +1
Score = 0.55×(+1) + 0.70×(+1) = 1.25 > 0
Predict: +1
```

---

## 4. Weak Learner vs Strong Learner

### 4.1 Weak Learner

**Definition:** Classifier slightly better than random guessing

```
Error slightly < 0.5 (binary classification)
Example: Error = 0.45 (5% better than random 50%)
```

**Typical Choice: Decision Stump (depth=1)**
```
Single split (one feature, one threshold)
Only 2 leaf nodes
Very simple model
```

**Why Stumps?**
- Fast to train
- Low variance (simple)
- Weak (intentionally underfitting)
- Boosting strong enough (many stumps combine)

---

### 4.2 Boosting Can Convert Weak to Strong

**Key Insight:**
```
Weak learner: ~50% error
Boosting: Iteratively reduce error
Final: Strong learner (low error)
```

**Magic:** Weak + Weak + ... = Strong (if done right)

---

### 4.3 When Error ≥ 0.5?

**Problem:** Learner no better than random

```
Error ≥ 0.5: Alpha ≤ 0 (negative weight)
AdaBoost "flips" prediction (weight < 0)
Learner deliberately used wrong way
```

**Solution:** Stop training (no improvement)

---

## 5. AdaBoost Hyperparameters

### 5.1 Number of Iterations (n_estimators)

**Effect:**
- More iterations: Better performance (up to point)
- Too many: Overfitting

**Typical Range:** 50-500

**When Stop:**
```
Training error stops decreasing
Cross-validation error increases (overfitting)
```

```python
# Monitor training error
train_errors = []
for n in range(1, 201):
    ada = AdaBoostClassifier(n_estimators=n)
    ada.fit(X_train, y_train)
    error = 1 - ada.score(X_train, y_train)
    train_errors.append(error)
    
# Plot to find elbow
```

---

### 5.2 Base Estimator (base_estimator)

**Default:** Decision stump (depth=1)

**Alternative:**
```python
# Deeper trees
base_estimator = DecisionTreeClassifier(max_depth=3)
ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)

# Different learner
base_estimator = LogisticRegression()
ada = AdaBoostClassifier(base_estimator=base_estimator)
```

**Effect:**
- Deeper trees: More complex, overfitting risk
- Shallower: Simpler, may underfit

**Best:** Start with stumps (default)

---

### 5.3 Learning Rate (learning_rate)

**Default:** 1.0

**Formula:**
```
Alpha_t = learning_rate × ln((1 - Error_t) / Error_t)
```

**Effect:**
- High learning_rate (1.0): Big steps, fast convergence, overfitting risk
- Low learning_rate (0.1): Small steps, slower, better generalization

**Strategy:**
```
learning_rate = 0.1, increase n_estimators to compensate
Effect: Slower learning, better generalization
```

---

### 5.4 Random State

```python
ada = AdaBoostClassifier(random_state=42)
```

**For Reproducibility:** Set seed

---

## 6. Advantages of AdaBoost

### 6.1 Reduces Bias

**Single Tree:** Can overfit (high variance)

**AdaBoost:** Multiple weak learners → Reduces bias

**Why:** Weak learners simpler → Low variance baseline
         Boosting focuses on errors → Reduces bias

---

### 6.2 Feature Interactions

**Automatic:**
- Sequential trees learn interactions
- Each tree refines previous

---

### 6.3 Robust Algorithm

**Theoretically Sound:**
- Exponential decrease in error rate
- Margin-based framework (similar to SVM)
- Principled approach

**Empirical:** Works well in practice

---

### 6.4 Feature Importance

**Available:**
```python
feature_importance = ada.feature_importances_
```

**Interpretation:** Features used in early, high-alpha learners more important

---

### 6.5 Works with Any Weak Learner

**Flexibility:**
```python
# Trees
ada_tree = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))

# Logistic Regression
ada_lr = AdaBoostClassifier(base_estimator=LogisticRegression())

# Any sklearn classifier
```

---

## 7. Disadvantages of AdaBoost

### 7.1 Sensitive to Outliers

**Problem:**
```
Outliers misclassified early
Weights increased exponentially
Later learners focus heavily on outliers
Can overfit to noise
```

**Solution:**
- Remove outliers
- Use robust weak learners
- Lower learning_rate

---

### 7.2 Requires Good Weak Learners

**Problem:**
```
If base learner Error ≥ 0.5 (or close)
Can't improve
AdaBoost fails
```

**Solution:**
- Ensure weak learner Error < 0.5
- Check with single learner first
- May need feature engineering

---

### 7.3 Sequential Training (Slow)

**Problem:**
- Each iteration depends on previous
- Can't parallelize
- Slow for large datasets

**Comparison:**
```
Random Forest: Parallel (B trees independently)
AdaBoost: Sequential (must train t before t+1)
```

**Time:** AdaBoost slower for large datasets

---

### 7.4 Prone to Overfitting

**Problem:**
- If too many iterations or low learning_rate
- Can memorize training data
- Requires careful tuning

**Solution:**
- Monitor validation error
- Tune n_estimators via CV
- Use learning_rate < 1.0

---

### 7.5 Black Box

**Problem:**
- Ensemble of learners hard to interpret
- "Why?" decision unclear

**Solution:**
- Feature importance
- Inspect individual learners (early ones most important)

---

### 7.6 Imbalanced Data

**Problem:**
- Minority class samples upweighted
- Can overfit to minority
- Need careful tuning

**Solutions:**
```python
# 1. Balanced learning rate
ada = AdaBoostClassifier(algorithm='SAMME')  # vs 'SAMME.R'

# 2. Scale_pos_weight equivalent
# Adjust base estimator

# 3. Resampling
from imblearn.over_sampling import SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
ada.fit(X_balanced, y_balanced)
```

---

## 8. AdaBoost Variants

### 8.1 SAMME vs SAMME.R

**SAMME (Stagewise Additive Modeling using a Multiclass Exponential loss):**
```
Algorithm = 'SAMME'
Alpha_t = (1/2) × ln((1 - Error_t) / Error_t)
Works with any classifier
```

**SAMME.R (Real probability estimates):**
```
Algorithm = 'SAMME.R' (default)
Uses probability estimates
Usually faster convergence
Needs probability output
```

---

### 8.2 AdaBoost Regression (AdaBoostRegressor)

**Difference:**
- Regression output (continuous)
- Different loss function (linear, square, exponential)
- Aggregation: Weighted median (not average)

```python
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(n_estimators=100, learning_rate=0.1)
ada_reg.fit(X_train, y_train)

y_pred = ada_reg.predict(X_test)
```

---

## 9. AdaBoost vs Other Ensemble Methods

| Aspect | AdaBoost | Random Forest | Gradient Boosting |
|--------|----------|--------------|-------------------|
| Training | Sequential | Parallel | Sequential |
| Sample Selection | Weighted | Bootstrap | Residuals |
| Reduces | Bias | Variance | Bias |
| Weak Learner | Shallow trees | Deep trees | Shallow trees |
| Outlier Sensitive | High | Medium | Medium |
| Speed | Slow | Fast | Medium |
| Tuning Difficulty | Hard | Easy | Hard |

---

## 10. Implementation in sklearn

### 10.1 Classification

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Train (with decision stump default)
ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train, y_train)

# Custom base estimator
base_estimator = DecisionTreeClassifier(max_depth=2)
ada_custom = AdaBoostClassifier(
    base_estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.5
)
ada_custom.fit(X_train, y_train)

# Predict
y_pred = ada.predict(X_test)
y_pred_proba = ada.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(ada, X_train, y_train, cv=5).mean()}")
print(f"\n{classification_report(y_test, y_pred)}")
```

### 10.2 Regression

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

ada_reg = AdaBoostRegressor(
    n_estimators=50,
    learning_rate=0.1,
    random_state=42
)
ada_reg.fit(X_train, y_train)

y_pred = ada_reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### 10.3 Feature Importance

```python
# Feature importance
importance = ada.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(importance_df)

# Plot
import matplotlib.pyplot as plt
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.show()
```

### 10.4 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'base_estimator__max_depth': [1, 2, 3]
}

ada = AdaBoostClassifier()
grid = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

### 10.5 Monitoring Training

```python
# Track error reduction
train_errors = []
test_errors = []

for n in range(1, 101):
    ada = AdaBoostClassifier(n_estimators=n, learning_rate=0.1)
    ada.fit(X_train, y_train)
    
    train_error = 1 - ada.score(X_train, y_train)
    test_error = 1 - ada.score(X_test, y_test)
    
    train_errors.append(train_error)
    test_errors.append(test_error)

plt.plot(train_errors, label='Train')
plt.plot(test_errors, label='Test')
plt.xlabel('Number of Estimators')
plt.ylabel('Error')
plt.legend()
plt.show()
```

---

## 11. Common Issues & Solutions

### 11.1 Poor Accuracy

**Causes:**
- Too few iterations
- Base learner too weak (Error ≥ 0.5)
- Learning rate too high (oscillating)
- Overfitting (training good, test poor)

**Solutions:**
```python
# 1. More iterations
ada = AdaBoostClassifier(n_estimators=200)

# 2. Check base learner quality
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(X_train, y_train)
score = dt.score(X_test, y_test)
print(f"Base learner accuracy: {score}")  # Should be > 0.5

# 3. Lower learning rate
ada = AdaBoostClassifier(learning_rate=0.1, n_estimators=200)

# 4. Reduce overfitting
ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=0.05,
    base_estimator=DecisionTreeClassifier(max_depth=1)
)
```

---

### 11.2 Overfitting

**Symptoms:** Train accuracy 95%, test 70%

**Solutions:**
```python
# 1. Reduce iterations (early stopping)
ada = AdaBoostClassifier(n_estimators=30)

# 2. Lower learning rate (slow learning)
ada = AdaBoostClassifier(learning_rate=0.05, n_estimators=200)

# 3. Simpler base learner
base_estimator = DecisionTreeClassifier(max_depth=1)  # Stump only
ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50)

# 4. Remove outliers
# Or use robust learners
```

---

### 11.3 Outliers Affect Training

**Problem:** Outliers upweighted, learner chases them

**Solutions:**
```python
# 1. Remove outliers
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
z_scores = np.abs(X_scaled)
outliers = (z_scores > 3).any(axis=1)
X_clean = X_train[~outliers]
y_clean = y_train[~outliers]

ada.fit(X_clean, y_clean)

# 2. Robust preprocessing
# Use robust scaling, IQR removal, etc.
```

---

## 12. When to Use AdaBoost

### 12.1 Good For

- Medium-sized datasets
- Need bias reduction (underfitting)
- Clear patterns (learnable with weak learners)
- Tabular data

### 12.2 Not Good For

- Large datasets (slow sequential training)
- Very noisy data (outliers)
- Base learner can't achieve Error < 0.5
- Real-time predictions

---

## 13. Practice Problems

1. **Alpha Calculation:** Why α = (1/2)ln((1-ε)/ε)?

2. **Weight Updates:** Sample correct vs misclassified. Weights change how?

3. **Weak Learner:** Why Error < 0.5? What if Error ≥ 0.5?

4. **Learning Rate:** Effect on convergence and overfitting?

5. **Sequential:** Why can't parallelize like Random Forest?

6. **Outliers:** Why problematic for AdaBoost?

7. **SAMME vs SAMME.R:** Differences?

8. **Early Stopping:** How detect overfitting? What metric?

---

## 14. Key Takeaways

1. **Sequential boosting:** Each learner focuses on previous mistakes
2. **Adaptive weights:** Misclassified samples upweighted
3. **Learner weight (Alpha):** Based on error rate
4. **Weak learner:** Typically decision stump (depth=1)
5. **Reduces bias:** Multiple weak learners → Complex boundary
6. **Final prediction:** Weighted sum of learner predictions
7. **Parameters:** n_estimators, learning_rate, base_estimator
8. **Advantages:** Reduces bias, feature interactions, principled
9. **Disadvantages:** Outlier sensitive, sequential (slow), overfitting risk
10. **Use when:** Medium data, bias reduction priority, good base learner available

---

**Next Topic:** Gradient Boosting (say "next" to continue)