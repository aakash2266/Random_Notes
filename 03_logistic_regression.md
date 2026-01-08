# Logistic Regression - Complete Guide

## 1. Fundamentals of Logistic Regression

### 1.1 What is Logistic Regression?

**Definition:** Supervised learning algorithm for binary classification; models probability of sample belonging to positive class.

**Key Point:** Despite name "regression," it's a CLASSIFICATION algorithm
- Predicts probability (0-1)
- Probability > 0.5 → Predict class 1
- Probability ≤ 0.5 → Predict class 0

**Binary Classification:**
```
y ∈ {0, 1}
```
- y=0: Negative class
- y=1: Positive class
- Examples: Disease/No disease, Spam/Not spam, Fraud/Legitimate

---

### 1.2 Linear Regression vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| Target | Continuous (ℝ) | Binary (0 or 1) |
| Predictions | Unbounded (-∞ to +∞) | Probability [0, 1] |
| Output | Numeric value | Probability |
| Cost Function | MSE | Log loss (Cross-entropy) |
| Decision Boundary | Line/Hyperplane | S-curve (Sigmoid) |
| Algorithm | OLS, Gradient Descent | Gradient Descent |

---

## 2. The Sigmoid Function

### 2.1 Logistic Function (Sigmoid)

**Equation:**
```
σ(z) = 1 / (1 + e^(-z))
```

Where:
- z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ (linear combination)
- σ(z) = probability of positive class

**Predictions:**
```
P(y=1|X) = σ(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Always outputs value in [0, 1]
- S-shaped curve
- z=0 → σ(z)=0.5 (decision boundary)
- z → +∞ → σ(z) → 1
- z → -∞ → σ(z) → 0
- Symmetric around 0.5

**Interpretation:**
```
P(y=1|X) = 0.8 means: 80% probability sample is positive
P(y=0|X) = 1 - 0.8 = 0.2 means: 20% probability sample is negative
```

---

### 2.2 Odds & Log-Odds

**Odds:**
```
Odds = P(y=1) / P(y=0) = P(y=1) / (1 - P(y=1))
```

- Odds > 1: Positive class more likely
- Odds = 1: Equal probability
- Odds < 1: Negative class more likely

**Log-Odds (Logit):**
```
log(Odds) = log(P(y=1) / P(y=0)) = β₀ + β₁x₁ + ... + βₚxₚ
```

**Interpretation:**
- Linear in log-odds space (linear in features)
- β_j = change in log-odds for 1 unit increase in x_j
- exp(β_j) = odds ratio (multiplicative change in odds)

**Example:**
- β_j = 0.693 (approximately ln(2))
- exp(0.693) ≈ 2
- One unit increase in x_j → Odds multiplied by 2 (doubled)

---

## 3. Cost Function & Loss Function

### 3.1 Why Not Mean Squared Error?

**Problem:** Using MSE with sigmoid non-convex
- Multiple local minima
- Gradient descent may get stuck
- Unreliable optimization

**Solution:** Use Log Loss (Cross-Entropy Loss)

---

### 3.2 Log Loss (Cross-Entropy)

**Binary Log Loss:**
```
L(y, ŷ) = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

Where:
- y: Actual label (0 or 1)
- ŷ = σ(z) = predicted probability

**Interpretation:**
- y=1, ŷ=0.9: Loss = -log(0.9) ≈ 0.105 (small, good prediction)
- y=1, ŷ=0.1: Loss = -log(0.1) ≈ 2.303 (large, bad prediction)
- y=0, ŷ=0.1: Loss = -log(0.9) ≈ 0.105 (small, good prediction)
- y=0, ŷ=0.9: Loss = -log(0.1) ≈ 2.303 (large, bad prediction)

**Why log loss?**
- Differentiable (gradient descent works)
- Convex (single global minimum)
- Penalizes confident wrong predictions heavily
- Probabilistic interpretation (maximum likelihood)

---

### 3.3 Total Cost Function

**Mean Log Loss (over all samples):**
```
J(β) = (1/n) × Σ[y_i × log(ŷ_i) + (1-y_i) × log(1-ŷ_i)]
```

**Gradient (for descent):**
```
∂J/∂β_j = (1/n) × Σ(ŷ_i - y_i) × x_ij
```

**Update rule (Gradient Descent):**
```
β_new = β_old - α × ∂J/∂β
```

---

## 4. Maximum Likelihood Estimation (MLE)

### 4.1 Probabilistic Interpretation

**Likelihood Function:**
```
L(β) = ∏ P(y_i=1|X_i)^(y_i) × P(y_i=0|X_i)^(1-y_i)
     = ∏ ŷ_i^(y_i) × (1-ŷ_i)^(1-y_i)
```

**Log-Likelihood:**
```
log L(β) = Σ[y_i × log(ŷ_i) + (1-y_i) × log(1-ŷ_i)]
```

**Maximum Likelihood Estimation:**
- Maximize log-likelihood
- Equivalent to minimizing negative log-likelihood (log loss)
- Logistic regression = MLE with sigmoid function

**Why MLE?**
- Theoretically sound
- Gives probability estimates
- Asymptotically unbiased and efficient

---

## 5. Logistic Regression Assumptions

### 5.1 Key Assumptions

**1. Binary Outcome:**
- Target must be binary (0/1)
- For multiclass: Use multinomial logistic or one-vs-rest

**2. Linear Relationship in Log-Odds:**
- log-odds linearly related to features
- Not about y vs X relationship (that's non-linear)
- Check: Plot log-odds vs each feature

**3. Independence:**
- Observations independent
- No temporal/spatial autocorrelation
- Same as linear regression

**4. No Multicollinearity:**
- Features not highly correlated (r > 0.8-0.9)
- VIF < 5-10
- Same as linear regression

**5. Sufficient Sample Size:**
- At least 10-20 events per predictor
- Small samples lead to unstable estimates
- Rule: n/p ≥ 20 or n ≥ max(10p, 100)

**6. No Perfect Separation:**
- One class perfectly separated from other (all x₁>5 → all y=1)
- Causes infinite coefficients
- Check: Try fitting; if coefficients huge → problem

---

### 5.2 Assumption Checking

**Linearity in Log-Odds:**
```python
# For each feature, plot log-odds vs feature
# Should be approximately linear
```

**Multicollinearity:**
- VIF > 10: Problematic
- Drop highly correlated feature
- Use regularization (Ridge/Lasso)

**Separation:**
- Try fitting; if convergence issues → check
- Use perfect separation detection
- Bayesian methods help (priors stabilize)

---

## 6. Model Evaluation Metrics

### 6.1 Classification Metrics

**Confusion Matrix:**
```
                Predicted
              Neg    Pos
Actual Neg    TN     FP
       Pos    FN     TP
```

- TN: True negatives (correct 0)
- TP: True positives (correct 1)
- FP: False positives (predicted 1, actually 0)
- FN: False negatives (predicted 0, actually 1)

---

### 6.2 Accuracy, Precision, Recall

**Accuracy:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Percentage correct predictions
- Misleading if imbalanced (predict all majority → high accuracy)
- Use when classes balanced

**Precision:**
```
Precision = TP / (TP + FP)
```
- Of positive predictions, how many correct?
- "Don't raise false alarms"
- Use when FP costly (spam, fraud alerts)

**Recall (Sensitivity, TPR):**
```
Recall = TP / (TP + FN)
```
- Of actual positives, how many caught?
- "Don't miss positives"
- Use when FN costly (disease, security threats)

**Trade-off:** Precision vs Recall inversely related
- Lower threshold → Higher recall, lower precision
- Higher threshold → Lower recall, higher precision

---

### 6.3 F1-Score, Specificity, ROC-AUC

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision & recall
- Balance when both matter
- 0 (worst) to 1 (best)
- Use when classes imbalanced and both costs similar

**Specificity (TNR):**
```
Specificity = TN / (TN + FP)
```
- Of actual negatives, how many correctly identified?
- "Avoid false alarms"

**ROC-AUC:**
- ROC curve: TPR (y) vs FPR (x) across thresholds
- AUC: Area under curve [0.5 (random), 1 (perfect)]
- Threshold-independent metric
- Good for imbalanced data (captures ranking ability)

---

### 6.4 Precision-Recall Curve & PR-AUC

**PR Curve:**
- Precision (y) vs Recall (x) across thresholds
- Better for imbalanced data (AUC-ROC can be misleading)
- PR-AUC directly measures minority class performance

**When to use:**
- Imbalanced classification
- Focus on positive class performance
- More informative than ROC-AUC when p >> n (99-1 split)

---

### 6.5 Which Metric?

| Scenario | Metric |
|----------|--------|
| Balanced classes | Accuracy, F1 |
| Imbalanced | Precision, Recall, F1, PR-AUC |
| FP costly | Precision |
| FN costly | Recall |
| Ranking | ROC-AUC, PR-AUC |
| Threshold tuning | All thresholds: ROC, PR curves |

---

## 7. Threshold Tuning

### 7.1 Decision Threshold

**Default:** 0.5
```
if P(y=1) ≥ 0.5: predict 1
else: predict 0
```

**Tuning:** Adjust threshold based on business cost
- Lower threshold: Higher recall, lower precision
- Higher threshold: Lower recall, higher precision

---

### 7.2 Threshold Selection Process

**1. Get predictions:**
```python
probs = model.predict_proba(X_test)[:, 1]  # P(y=1)
```

**2. Sweep thresholds:**
```python
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for t in thresholds:
    predictions = (probs >= t).astype(int)
    # Compute metrics (precision, recall, F1)
```

**3. Plot & choose:**
- Precision-Recall curve
- Choose threshold matching business needs

**Example:**
- Disease detection: Low threshold (high recall, catch more sick)
- Spam filter: High threshold (high precision, avoid blocking legitimate)

---

## 8. Regularization in Logistic Regression

### 8.1 Why Regularization?

**Problem:** 
- Overfitting (model memorizes noise)
- Large coefficients (unstable estimates)
- Perfect separation (infinite coefficients)

**Solution:** Add penalty for large coefficients

---

### 8.2 L2 Regularization (Ridge)

**Cost Function:**
```
J(β) = Log Loss + λ × Σβ_j²
```

**Effect:**
- Shrinks coefficients toward zero (but not to zero)
- All features remain
- Handles multicollinearity

**When:** Multicollinearity present, keep all features

---

### 8.3 L1 Regularization (Lasso)

**Cost Function:**
```
J(β) = Log Loss + λ × Σ|β_j|
```

**Effect:**
- Shrinks coefficients to exactly zero
- Automatic feature selection
- Interpretable (sparse model)

**When:** Feature selection needed, many irrelevant features

---

### 8.4 Elastic Net

**Cost Function:**
```
J(β) = Log Loss + λ₁ × Σ|β_j| + λ₂ × Σβ_j²
```

**Combines:**
- L1: Feature selection
- L2: Multicollinearity handling

**When:** Both needs present

---

### 8.5 Parameter Selection

**Cross-Validation:**
1. Define parameter grid: λ = [0.0001, 0.001, 0.01, 0.1, 1, 10]
2. For each λ: k-fold CV score
3. Choose λ with best score

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

C_values = [0.001, 0.01, 0.1, 1, 10, 100]  # 1/λ
lr = LogisticRegression()
grid = GridSearchCV(lr, {'C': C_values}, cv=5)
grid.fit(X_train, y_train)
print(f"Best C: {grid.best_params_}")
```

**Note:** sklearn uses C = 1/λ (inverse relationship)
- Large C: Less regularization
- Small C: More regularization

---

## 9. Multiclass Logistic Regression

### 9.1 Binary to Multiclass

**Binary (k=2):**
```
P(y=1|X) = σ(z)
P(y=0|X) = 1 - σ(z)
```

**Multiclass (k>2):**
```
P(y=k|X) = softmax(z_k)
```

---

### 9.2 Softmax Function

**Equation:**
```
P(y=k|X) = e^(z_k) / Σⱼ e^(z_j)
```

Where:
- z_k = β₀ₖ + β₁ₖx₁ + ... (class-specific linear combination)
- Normalized exponential (probabilities sum to 1)

**Properties:**
- Generalizes sigmoid to multiclass
- Always outputs valid probability distribution
- ∑ P(y=k) = 1

---

### 9.3 Multiclass Classification Approaches

**1. One-vs-Rest (OvR):**
- Train k binary classifiers (class k vs rest)
- Choose class with highest probability
- Simple, works well

**2. One-vs-One (OvO):**
- Train k(k-1)/2 classifiers (each pair)
- Vote on final class
- More computation, sometimes better

**sklearn default:** Automatically chooses based on multi_class parameter

---

### 9.4 Cost Function for Multiclass

**Cross-Entropy Loss:**
```
J(β) = -Σ Σ y_ik × log(ŷ_ik)
```

Where:
- y_ik = 1 if sample i belongs to class k, else 0
- ŷ_ik = P(y=k|X_i) from softmax

**Same as binary:** Just generalized

---

## 10. Advantages & Disadvantages

### 10.1 Advantages

1. **Interpretability:**
   - Linear in log-odds
   - Coefficients easy to interpret
   - exp(β) = odds ratio

2. **Probabilistic:**
   - Direct probability estimates
   - Can threshold flexibly
   - Uncertainty quantification

3. **Fast:**
   - Simple algorithm
   - Fast training/prediction
   - Scales to large datasets

4. **Baseline:**
   - Good baseline for classification
   - Often performs well with good features

5. **Regularization:**
   - Simple to add (L1, L2, Elastic Net)
   - Helps with multicollinearity

---

### 10.2 Disadvantages

1. **Linear Decision Boundary:**
   - Can't handle XOR problem
   - Non-linear patterns missed
   - Needs feature engineering for curves

2. **Assumes Log-Linear Relationship:**
   - May not hold in practice
   - Violate with caution

3. **Sensitive to Outliers:**
   - Outliers can bias coefficients
   - Solution: Remove, robust methods

4. **Not Automatic Feature Selection:**
   - Lasso enables, but manual check needed
   - All features remain (Ridge)

5. **Worse than Complex Models:**
   - Deep learning, tree models often better
   - Trade-off: Interpretability vs performance

---

## 11. Comparison with Other Classifiers

| Aspect | Logistic Regression | Decision Tree | SVM | Neural Network |
|--------|-------------------|----------------|-----|----------------|
| Decision Boundary | Linear | Axis-aligned | Non-linear | Arbitrary |
| Interpretability | High | High | Low | Very low |
| Training Speed | Fast | Fast | Slow | Slow |
| Probability | Yes | No | No | Yes |
| Multiclass | Yes (softmax) | Yes | Yes (OvR) | Yes |
| Non-linear | Manual features | Automatic | Kernel trick | Learned |
| Data Requirements | Small | Medium | Medium | Large |

---

## 12. Implementation in sklearn

### 12.1 Basic Usage

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Create and train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # P(y=1)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
```

### 12.2 Regularization

```python
# L2 (Ridge) - default
model_l2 = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')

# L1 (Lasso) - needs specific solver
model_l1 = LogisticRegression(C=1.0, penalty='l1', solver='liblinear')

# Elastic Net
model_elastic = LogisticRegression(C=1.0, penalty='elasticnet', solver='saga', l1_ratio=0.5)

# C: Inverse of regularization strength (C = 1/λ)
# Lower C = More regularization (simpler model)
# Higher C = Less regularization (complex model)
```

### 12.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(LogisticRegression(), params, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

### 12.4 Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

# Get probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = np.arange(0, 1, 0.1)
for t in thresholds:
    y_pred_custom = (y_pred_proba >= t).astype(int)
    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    print(f"Threshold: {t:.1f}, Precision: {precision:.3f}, Recall: {recall:.3f}")

# Plot precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

### 12.5 Feature Importance

```python
# Get coefficients
coefficients = model.coef_[0]
feature_names = X_train.columns

# Sort by absolute value
importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coeff': np.abs(coefficients)
}).sort_values('Abs_Coeff', ascending=False)

print(importance)

# Interpretation: exp(coef) = odds ratio
importance['Odds_Ratio'] = np.exp(importance['Coefficient'])
print(importance)
```

### 12.6 Class Weights for Imbalanced

```python
# Automatic balancing
model = LogisticRegression(class_weight='balanced')

# Or manual weights
class_weights = {0: 1, 1: 99}  # Weight minority more
model = LogisticRegression(class_weight=class_weights)
```

---

## 13. Common Issues & Solutions

### 13.1 Convergence Issues

**Problem:** Model doesn't converge (warning in training)

**Causes:**
- Too few iterations
- Learning rate too low
- Features not scaled
- Perfect separation

**Solutions:**
```python
# Increase iterations
model = LogisticRegression(max_iter=10000)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Check for perfect separation
# If coefficients huge → problem
```

---

### 13.2 Multicollinearity

**Problem:** Coefficients unstable

**Detection:**
- Large coefficients
- High VIF (>5-10)
- Correlation matrix (r > 0.8)

**Solutions:**
- Remove correlated feature
- Use regularization (Ridge)
- PCA for uncorrelated features

---

### 13.3 Imbalanced Classes

**Problem:** Model biased toward majority

**Solutions:**
```python
# Class weights
model = LogisticRegression(class_weight='balanced')

# Threshold tuning
y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Lower for minority focus
y_pred = (y_pred_proba >= threshold).astype(int)

# SMOTE before training
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
model.fit(X_train_sm, y_train_sm)
```

---

## 14. Interpretation & Communication

### 14.1 Coefficient Interpretation

**Example Model:**
```
log-odds = -2 + 0.5*age - 1.2*income + 0.8*credit_score
```

**Interpretation:**
- **Intercept (-2):** At age=0, income=0, credit_score=0 (baseline, often meaningless)
- **age (0.5):** One year increase in age → log-odds increase 0.5 → odds multiply by e^0.5 ≈ 1.65 (65% more likely positive)
- **income (-1.2):** One unit increase in income → log-odds decrease 1.2 → odds multiply by e^(-1.2) ≈ 0.30 (70% less likely positive)
- **credit_score (0.8):** One point increase → odds multiply by e^0.8 ≈ 2.23 (123% more likely positive)

### 14.2 Predictions Communication

```
"Model predicts 75% probability customer will churn"
vs
"Model predicts customer will churn"
```

First is better (includes uncertainty). Use probability, not binary prediction for business.

---

## 15. Practice Problems

1. **Sigmoid vs OLS:** Why can't we use MSE with logistic regression? Why log loss better?

2. **Log-odds Interpretation:** If β_age = 0.1, how much do odds change per year?

3. **Precision-Recall:** In medical diagnosis, should we tune precision or recall high? Why?

4. **Threshold Selection:** Fraud detection. FN cost = $1000 (miss fraud). FP cost = $10 (false alarm). How adjust threshold?

5. **Multicollinearity:** Two correlated features. How does this affect coefficient interpretation?

6. **Regularization:** Why does L1 shrink to zero but L2 shrinks toward zero?

7. **Multiclass:** How does softmax generalize sigmoid? Formula?

8. **Class Imbalance:** 1% positive, 99% negative. Why is accuracy misleading? What metric better?

---

## 16. Key Takeaways

1. **Logistic regression = Classification** despite name
2. **Sigmoid function** bounds predictions [0,1]
3. **Log-odds linear** in features: log(odds) = β₀ + Σβ_j×x_j
4. **Log loss (cross-entropy)** convex cost function
5. **Threshold tuning** balances precision-recall based on costs
6. **Regularization** handles multicollinearity and overfitting
7. **Interpretable:** Coefficients → odds ratios
8. **Fast & simple:** Good baseline for classification
9. **Linear boundary:** Manual feature engineering for non-linear
10. **Probabilistic:** Gives uncertainty estimates (unlike trees)

---

**Next Topic:** Decision Trees (say "next" to continue)