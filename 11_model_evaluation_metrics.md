# Model Evaluation & Metrics - Complete Guide

## 1. Fundamentals of Model Evaluation

### 1.1 Why Evaluate Models?

**Goals:**
```
1. Assess performance: How good is the model?
2. Compare models: Which is better?
3. Tune hyperparameters: What settings work best?
4. Detect problems: Overfitting? Underfitting? Bias?
5. Make business decisions: Deploy or iterate?
```

---

### 1.2 Training vs Test vs Validation Sets

**Training Set:**
- Used to fit model
- Model learns patterns
- Typically: 60-80% of data

**Validation Set:**
- Used to tune hyperparameters
- Monitor generalization during training
- Typically: 10-20% of data
- Early stopping, model selection

**Test Set:**
- Never seen during training
- Final evaluation of generalization
- Typically: 10-20% of data
- Never use for tuning!

**Split Strategy:**
```
100 samples
Train: 60, Validation: 20, Test: 20 (60-20-20 split)

OR

Train: 70, Validation: 15, Test: 15 (70-15-15 split)

Depends on dataset size
```

---

### 1.3 Key Evaluation Principles

**1. Never Use Test Set for Tuning:**
- Leads to overfitting to test set
- Inflated performance estimates
- False confidence

**2. Always Use Cross-Validation:**
- More robust estimate
- Uses all data
- Reduces variance of estimate

**3. Match Metric to Goal:**
- Accuracy for balanced
- Precision/Recall for imbalanced
- ROC-AUC for ranking

**4. Report Uncertainty:**
- Standard deviation of CV scores
- Confidence intervals
- Multiple runs with different seeds

---

## 2. Regression Metrics

### 2.1 Mean Squared Error (MSE) & RMSE

**MSE:**
```
MSE = (1/n) × Σ(y_i - ŷ_i)²
```

**RMSE (Root Mean Squared Error):**
```
RMSE = √MSE = √((1/n) × Σ(y_i - ŷ_i)²)
```

**Interpretation:**
- Average squared error
- RMSE in same units as y (more interpretable)
- Lower is better

**Characteristics:**
- Penalizes large errors heavily (quadratic)
- Sensitive to outliers
- Use when large errors costly

**Example:**
```
Predictions: [10, 20, 30]
Actuals: [12, 18, 35]
Errors: [2, -2, -5]
MSE = (4 + 4 + 25) / 3 = 11
RMSE = √11 ≈ 3.3
Average error magnitude: ~3.3 units
```

---

### 2.2 Mean Absolute Error (MAE)

**Formula:**
```
MAE = (1/n) × Σ|y_i - ŷ_i|
```

**Interpretation:**
- Average absolute error
- Same units as y
- Lower is better

**Characteristics:**
- Treats all errors equally (linear)
- Robust to outliers
- More interpretable than MSE/RMSE

**Example:**
```
Same data: errors = [2, -2, -5]
MAE = (2 + 2 + 5) / 3 = 3
Average absolute error: 3 units
```

**MSE vs MAE:**
```
MSE = 11, RMSE = 3.3, MAE = 3

MSE penalizes large error (5) heavily
MAE treats all errors equally

Outlier impact:
  If one large error: MSE worse
  MAE more stable
```

---

### 2.3 R² (Coefficient of Determination)

**Formula:**
```
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
```

Where:
- SS_res: Sum of squared residuals
- SS_tot: Total sum of squares (vs mean)
- ȳ: Mean of y

**Interpretation:**
- Percentage of variance explained
- Range: [0, 1] (can be negative if worse than mean)
- R² = 1: Perfect prediction
- R² = 0: No better than mean
- R² < 0: Worse than predicting mean

**Example:**
```
y = [1, 2, 3, 4, 5], mean = 3
ŷ = [1.1, 2.1, 2.9, 3.9, 5.1]

SS_tot = (1-3)² + (2-3)² + ... + (5-3)² = 10
SS_res = (1-1.1)² + (2-2.1)² + ... = 0.04
R² = 1 - (0.04/10) = 0.996 (excellent)
```

---

### 2.4 Adjusted R²

**Formula:**
```
Adj R² = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]
```

Where:
- n: Number of samples
- p: Number of features

**Purpose:** Penalizes adding more features

**When to Use:**
- Comparing models with different # features
- Feature selection

**Example:**
```
Model A: p=5, R²=0.90, n=100
Model B: p=10, R²=0.91, n=100

R² favors Model B (higher)
But Adj R² favors Model A (penalizes extra 5 features)
```

---

### 2.5 MAPE (Mean Absolute Percentage Error)

**Formula:**
```
MAPE = (1/n) × Σ|y_i - ŷ_i| / |y_i| × 100%
```

**Interpretation:**
- Percentage error
- Scale-independent
- Lower is better

**Characteristics:**
- Good for comparing across different scales
- Problem: Undefined when y_i = 0
- Can be misleading if many small y values

**Example:**
```
y = [100, 50], ŷ = [110, 55]
Errors: [10, 5]
MAPE = (10/100 + 5/50) / 2 × 100 = 10%
Average 10% error
```

---

### 2.6 Regression Metric Selection

| Metric | When to Use | Robustness |
|--------|------------|-----------|
| MSE/RMSE | General, large errors matter | Outlier-sensitive |
| MAE | Outliers present | Robust |
| R² | Variance explained | Sensitive to outliers |
| Adj R² | Multiple models, feature selection | Better for comparison |
| MAPE | Different scales, percentage important | Undefined at y=0 |

---

## 3. Classification Metrics - Confusion Matrix

### 3.1 Binary Classification Setup

**Confusion Matrix:**
```
                Predicted
              Negative  Positive
Actual Negative    TN      FP
       Positive    FN      TP
```

Where:
- TN (True Negative): Correct 0 (negative)
- TP (True Positive): Correct 1 (positive)
- FP (False Positive): Predicted 1, actually 0 (false alarm)
- FN (False Negative): Predicted 0, actually 1 (miss)

**Total:**
```
n = TP + TN + FP + FN
```

---

### 3.2 Accuracy

**Formula:**
```
Accuracy = (TP + TN) / n = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretation:**
- Percentage correct predictions
- Overall performance

**When to Use:**
- Balanced classes
- All errors equally costly

**Problem (Imbalanced):**
```
99% class 0, 1% class 1
Predict all 0:
  Accuracy = 99% (misleading!)
  But catches 0% of class 1
```

---

### 3.3 Precision & Recall

**Precision (Positive Predictive Value):**
```
Precision = TP / (TP + FP)

Of predicted positives, how many correct?
"Don't raise false alarms"
```

**Recall (Sensitivity, True Positive Rate):**
```
Recall = TP / (TP + FN)

Of actual positives, how many caught?
"Don't miss positives"
```

**Trade-off:**
- High Precision: Few false alarms, may miss positives
- High Recall: Catch most positives, many false alarms

**Examples:**
```
Disease detection (high Recall):
  TP=95, FP=5, FN=5, TN=895
  Precision = 95/100 = 95%
  Recall = 95/100 = 95%
  (Catch most, few false alarms)

Spam filter (high Precision):
  TP=90, FP=1, FN=9, TN=900
  Precision = 90/91 = 99%
  Recall = 90/99 = 91%
  (Few false positives, miss some spam)
```

---

### 3.4 F1-Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:**
- Harmonic mean of Precision and Recall
- Balance when both matter
- 0 (worst) to 1 (best)

**When to Use:**
- Imbalanced data
- Both false positives and false negatives costly
- Need single metric

**Property:**
```
F1 high only if both Precision and Recall high
F1 = 0 if either = 0
```

---

### 3.5 Specificity & Fall-Out

**Specificity (True Negative Rate):**
```
Specificity = TN / (TN + FP)

Of actual negatives, how many correctly identified?
```

**False Positive Rate (Fall-Out):**
```
FPR = FP / (TN + FP) = 1 - Specificity

Of actual negatives, how many incorrectly flagged?
```

---

### 3.6 Classification Metric Selection

| Metric | Use When | Problem |
|--------|----------|---------|
| Accuracy | Balanced classes | Misleading if imbalanced |
| Precision | FP costly (spam alerts) | Ignores false negatives |
| Recall | FN costly (disease, fraud) | Ignores false positives |
| F1 | Both costs similar, imbalanced | Not probability |
| Specificity | Negatives important | Depends on context |

---

## 4. Threshold-Dependent Metrics

### 4.1 Decision Threshold

**Default:** 0.5
```
If P(class=1) ≥ 0.5: Predict 1
Else: Predict 0
```

**Tuning:** Adjust for business needs
```
Lower threshold (e.g., 0.3):
  - Higher Recall (catch more positives)
  - Lower Precision (more false positives)

Higher threshold (e.g., 0.7):
  - Lower Recall (miss some positives)
  - Higher Precision (fewer false positives)
```

---

### 4.2 Threshold Sweep

**Process:**
```
1. Get probabilities: P(class=1)
2. For each threshold t in [0, 1]:
   a. Predictions: P ≥ t → class 1
   b. Calculate Precision, Recall, F1, etc.
3. Plot: Precision-Recall curve, ROC curve
4. Choose threshold matching business needs
```

---

## 5. ROC-AUC (Receiver Operating Characteristic)

### 5.1 ROC Curve

**Definition:**
- Plot TPR (y-axis) vs FPR (x-axis)
- Across all possible thresholds
- TPR = Recall = TP / (TP + FN)
- FPR = FP / (TN + FP)

**Interpretation:**
```
Perfect: Curve goes straight up (1.0) then right (0.0)
Random: Diagonal line from (0,0) to (1,1)
Poor: Below diagonal
```

**Threshold Movement:**
```
Threshold = 0: Predict all 1
  TP = all positives, FP = all negatives
  TPR = 1.0, FPR = 1.0 (top-right)

Threshold = 1: Predict all 0
  TP = 0, FP = 0
  TPR = 0, FPR = 0 (bottom-left)

Intermediate: Points between
```

---

### 5.2 AUC (Area Under Curve)

**Definition:**
```
AUC = Area under ROC curve
```

**Range:** [0.5 (random), 1.0 (perfect)]

**Interpretation:**
- Probability model ranks random positive higher than random negative
- AUC = 0.7: 70% chance model ranks positive > negative
- AUC = 0.5: Random guessing
- AUC = 1.0: Perfect ranking

**When to Use:**
- Threshold-independent metric
- Good for imbalanced data
- Ranking ability (not classification at threshold)

**Properties:**
```
AUC invariant to:
- Class imbalance (unlike Accuracy)
- Threshold choice
- Cost asymmetry

But ignores:
- Actual probability calibration
- Specific threshold performance
```

---

### 5.3 ROC-AUC vs PR-AUC

**ROC-AUC:**
- TPR vs FPR
- Good when negatives >> positives
- Can be misleading if extremely imbalanced (99:1)

**PR-AUC (Precision-Recall Area):**
- Precision vs Recall
- Better for imbalanced (focuses on minority)
- More informative than ROC-AUC for imbalanced

**Comparison:**
```
Imbalanced: 99% negative, 1% positive

ROC-AUC can be high even with poor minority performance
(because FPR = FP / huge_negatives)

PR-AUC better reflects true performance
(because Precision focuses on positives)
```

---

## 6. Multi-class Metrics

### 6.1 Averaging Strategies

**Macro-Averaging:**
```
Metric = (1/k) × Σ Metric_i

Treats all classes equally
Good: Balanced concern
Bad: Ignores class imbalance
```

**Weighted-Averaging:**
```
Metric = Σ (support_i / n) × Metric_i

Weights by support (frequency)
Good: Reflects real distribution
Default in sklearn
```

**Micro-Averaging:**
```
Metric = Σ TP_i / Σ (TP_i + FP_i)

Equivalent to Accuracy for multiclass
Same for all classes
```

**Example (3 classes):**
```
Class 0: Precision = 0.90, support = 50
Class 1: Precision = 0.80, support = 30
Class 2: Precision = 0.70, support = 20

Macro: (0.90 + 0.80 + 0.70) / 3 = 0.80
Weighted: (0.90×50 + 0.80×30 + 0.70×20) / 100 = 0.82
Micro: Accuracy
```

---

### 6.2 One-vs-Rest for Binary Metrics

**Approach:**
- Treat each class as binary (class vs rest)
- Calculate Precision, Recall, F1 per class
- Average across classes

**Code:**
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
# Shows per-class and averaged metrics
```

---

## 7. Probability Calibration

### 7.1 Problem: Poor Calibration

**Issue:**
```
Model predicts P(class=1) = 0.8
But actual frequency = 0.5
Predictions poorly calibrated
```

**Check:**
- Reliability diagram: Predicted vs Actual probability
- Calibration curve: For each bin, plot actual frequency

---

### 7.2 Brier Score

**Formula:**
```
Brier = (1/n) × Σ(y_i - p_i)²
```

Where p_i = predicted probability

**Interpretation:**
- Average squared probability error
- Lower is better
- Range: [0, 1]
- Brier = 0: Perfect calibration
- Brier = 0.25: Random guessing (p=0.5)

---

### 7.3 Log Loss

**Formula:**
```
Log Loss = -(1/n) × Σ[y_i × log(p_i) + (1-y_i) × log(1-p_i)]
```

**Interpretation:**
- Penalizes confident wrong predictions
- Lower is better
- Unbounded above

**Properties:**
- More sensitive to probability errors than Brier
- Preferred for probabilistic predictions

---

## 8. Cross-Validation

### 8.1 Why Cross-Validation?

**Problem (single train-test split):**
```
Estimate depends on specific split
Different splits → Different estimates
Variance high
```

**Solution (k-fold CV):**
```
Split into k folds
Train k times (k-1 folds), test on 1 fold
Average k scores
More robust estimate
```

---

### 8.2 k-Fold Cross-Validation

**Process:**
```
1. Split data into k equal folds (k=5 typical)
2. For fold i=1 to k:
   a. Training: folds except i
   b. Validation: fold i
   c. Train model, evaluate
3. Scores: [score_1, score_2, ..., score_k]
4. Mean score, std of scores
```

**Code:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean()}, Std: {scores.std()}")
```

---

### 8.3 Stratified k-Fold

**Purpose:** Preserve class distribution across folds

**Use When:** Classification with imbalanced data

**Effect:**
- Each fold has similar class proportions
- More representative folds
- More stable CV estimates

**Code:**
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

---

### 8.4 Leave-One-Out CV (LOOCV)

**Process:**
```
For each sample i:
  Train on n-1 samples (all except i)
  Test on sample i
n scores → Average
```

**Pros:** Maximally uses data, unbiased estimate

**Cons:** Slow (n iterations), high variance

**Use When:** Small dataset (n < 1000)

---

## 9. Hyperparameter Tuning Metrics

### 9.1 Grid Search & Metric

**Code:**
```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='f1'  # Metric to optimize
)
grid.fit(X, y)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_}")
```

**scoring parameter:**
```
Classification:
  'accuracy', 'precision', 'recall', 'f1'
  'roc_auc', 'precision_recall'

Regression:
  'r2', 'neg_mean_squared_error', 'mean_absolute_error'
  'neg_mean_absolute_percentage_error'
```

---

### 9.2 Custom Scoring

```python
from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    return some_calculation(y_true, y_pred)

scoring = make_scorer(custom_metric, greater_is_better=True)
grid = GridSearchCV(model, param_grid, scoring=scoring)
```

---

## 10. Business Metrics vs ML Metrics

### 10.1 Disconnect

**ML Metrics:**
- Accuracy, F1, AUC
- Statistical properties
- Model-centric

**Business Metrics:**
- Revenue, customer satisfaction
- Cost of errors
- Business-centric

**Alignment:**
```
Disease detection:
  ML: High Recall (catch disease)
  Business: Minimize cost (treatment + missed diagnosis)
  
Fraud detection:
  ML: High Precision (few false positives)
  Business: Minimize fraud losses
  
Recommendation:
  ML: High ranking AUC
  Business: User engagement, revenue
```

---

### 10.2 Cost-Sensitive Learning

**Idea:** Different errors have different costs

```
Cost Matrix:
              Predicted Negative  Predicted Positive
Actual Neg    0                   Cost_FP
Actual Pos    Cost_FN             0
```

**Weighted Metrics:**
```
Cost = Cost_FN × FN + Cost_FP × FP
Minimize cost instead of accuracy
```

---

## 11. Model Comparison

### 11.1 Statistical Significance Testing

**Question:** Is Model A significantly better than Model B?

**Method: Paired t-test**
```
CV scores from both models on same folds
Test if mean difference significant
```

**Code:**
```python
from scipy import stats

scores_A = cross_val_score(model_A, X, y, cv=5)
scores_B = cross_val_score(model_B, X, y, cv=5)

t_stat, p_value = stats.ttest_rel(scores_A, scores_B)
print(f"p-value: {p_value}")
# p < 0.05: Significant difference
```

---

### 11.2 Learning Curves

**Idea:** Plot performance vs dataset size

```
Training size on x-axis
Error on y-axis

Two curves:
  Train error: Decreases with more data
  Test error: Decreases with more data
```

**Interpretation:**
```
High train-test gap: Overfitting (need regularization or more data)
Both high: Underfitting (need more features or complexity)
Both low: Good fit
```

---

## 12. Baseline & Ensemble

### 12.1 Baselines

**Always establish baseline:**
```
Random: Accuracy = 1/k (k classes)
Majority: Accuracy = % majority class
Simple model: Logistic Regression vs complex

Compare against baseline
If not better: Model not useful
```

---

### 12.2 Ensemble Comparison

```python
# Individual models
rf_scores = cross_val_score(rf, X, y, cv=5)
gb_scores = cross_val_score(gb, X, y, cv=5)

# Ensemble (voting)
voting = VotingClassifier([('rf', rf), ('gb', gb)])
voting_scores = cross_val_score(voting, X, y, cv=5)

# Check: Ensemble usually beats individuals
print(f"RF: {rf_scores.mean()}")
print(f"GB: {gb_scores.mean()}")
print(f"Ensemble: {voting_scores.mean()}")
```

---

## 13. Implementation in sklearn

### 13.1 Comprehensive Evaluation

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, f1_score,
    accuracy_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Basic metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Classification report
print(classification_report(y_test, y_pred))

# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

# Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```

### 13.2 Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')

print(f"Mean F1: {scores.mean():.3f}")
print(f"Std F1: {scores.std():.3f}")
print(f"95% CI: [{scores.mean() - 1.96*scores.std():.3f}, {scores.mean() + 1.96*scores.std():.3f}]")
```

### 13.3 Hyperparameter Tuning with Metrics

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(
    GradientBoostingClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',  # Optimize for AUC
    n_jobs=-1
)

grid.fit(X_train, y_train)
print(f"Best AUC: {grid.best_score_:.3f}")
print(f"Test AUC: {roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]):.3f}")
```

---

## 14. Common Evaluation Pitfalls

### 14.1 Data Leakage

**Problem:** Information from test set leaks into training

**Examples:**
```
1. Scale entire dataset then split (leaks test statistics)
2. Feature engineering on full data before split
3. Use test set for hyperparameter tuning
4. Preprocessing same way inside and outside CV
```

**Prevention:**
```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, ... = train_test_split(X_scaled, y)

# CORRECT
X_train, X_test, ... = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 14.2 Metric Mismatches

**Problem:** Choose metric misaligned with goal

```
Imbalanced data, choose Accuracy:
  Misleading (high even with poor minority)
  
Choose only Precision:
  May ignore false negatives
  
Choose AUC for cost-sensitive:
  Ignores cost matrix
```

**Solution:** Align metric with business goal

---

### 14.3 Overfitting to Test Set

**Problem:** Report test score, tune on it

**Result:** Inflated performance estimate

**Solution:** Separate validation and test

---

## 15. Choosing Metrics - Decision Tree

```
Classification?
  └─ Balanced?
      ├─ Yes: Accuracy, Precision, Recall, F1
      └─ No (Imbalanced):
          └─ Cost asymmetric?
              ├─ FP costly: Precision
              ├─ FN costly: Recall
              └─ Both: F1, PR-AUC
                  
  └─ Need ranking?
      ├─ Yes: ROC-AUC (balanced), PR-AUC (imbalanced)
      └─ No: Above metrics

Regression?
  └─ Outliers?
      ├─ Yes: MAE
      └─ No: RMSE, R²
  
  └─ Percentage error important?
      ├─ Yes: MAPE
      └─ No: MAE, RMSE, R²
```

---

## 16. Key Takeaways

1. **Always split:** Train, Validation, Test (never mix!)
2. **Cross-validate:** More robust than single split
3. **Match metric to goal:** Don't always use Accuracy
4. **Regression:** MSE/RMSE (sensitive), MAE (robust), R² (explained variance)
5. **Classification balanced:** Accuracy, F1, AUC all useful
6. **Classification imbalanced:** PR-AUC better than ROC-AUC, F1 or Precision/Recall
7. **Threshold tuning:** Lower for Recall, higher for Precision
8. **Avoid data leakage:** Preprocess after split, within CV
9. **Report uncertainty:** Mean ± std from CV
10. **Business alignment:** ML metrics should support business goals

---

**Next Topic:** Imbalanced Classification (say "next" to continue)