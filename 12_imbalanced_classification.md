# Imbalanced Classification - Complete Guide

## 1. Fundamentals of Imbalanced Classification

### 1.1 What is Class Imbalance?

**Definition:** Classification problem where classes have significantly different frequencies.

**Examples:**
```
Fraud detection: 99% legitimate, 1% fraud
Disease diagnosis: 95% healthy, 5% diseased
Churn prediction: 90% retained, 10% churned
Anomaly detection: 99.9% normal, 0.1% anomaly
```

**Problem:** Standard algorithms biased toward majority class

---

### 1.2 Naive Approach (Wrong!)

**Predict all samples as majority class:**
```
Dataset: 99% class 0, 1% class 1
Model: Always predict 0

Accuracy = 99% (excellent!)
But: Catches 0% of class 1 (useless)
```

**Why This Happens:**
- Loss minimization: Accuracy maximized by predicting majority
- Decision boundary biased toward majority
- Minority class samples ignored

---

### 1.3 Imbalance Ratio

**Definition:** Ratio of majority to minority class

```
Imbalance Ratio = n_majority / n_minority

Mild: 2:1 (e.g., 67%-33%)
Moderate: 10:1 (e.g., 91%-9%)
Severe: 100:1 (e.g., 99%-1%)
Extreme: 1000:1 (e.g., 99.9%-0.1%)
```

**Severity:**
- Mild: Standard algorithms usually work
- Moderate: Need adjustments
- Severe: Requires special techniques
- Extreme: Challenging (few positive examples)

---

## 2. Problem Analysis

### 2.1 Why Standard Metrics Fail

**Accuracy is Misleading:**
```
Dataset: 99% negative, 1% positive

Model A (always predict negative):
  Accuracy = 99% (looks great!)
  Recall = 0% (useless)
  
Model B (reasonable model):
  Accuracy = 85% (looks bad!)
  Recall = 90% (actually better)
```

**Reason:** Accuracy weighted equally to all classes

---

### 2.2 Better Metrics for Imbalanced Data

**Use Instead of Accuracy:**

1. **Precision & Recall:**
   - Precision: Of predicted positive, how many correct?
   - Recall: Of actual positive, how many caught?
   - Choose based on cost

2. **F1-Score:**
   - Harmonic mean of Precision and Recall
   - Single metric balancing both

3. **PR-AUC (Precision-Recall Area Under Curve):**
   - Better than ROC-AUC for imbalanced
   - Focuses on minority class performance

4. **Matthews Correlation Coefficient (MCC):**
   - Correlation between predicted and actual
   - Balanced metric, ranges [-1, 1]

---

### 2.3 Cost Matrix

**Different Errors, Different Costs:**

```
        Predicted Negative  Predicted Positive
Actual Negative    0              Cost_FP
Actual Positive    Cost_FN        0
```

**Examples:**
```
Disease Detection:
  Cost_FN (miss disease) = $10,000 (treatment cost)
  Cost_FP (false alarm) = $100 (unnecessary test)
  → Favor high Recall

Spam Detection:
  Cost_FN (miss spam) = $1 (user annoyance)
  Cost_FP (block legitimate) = $50 (lost customer)
  → Favor high Precision
```

---

## 3. Data-Level Solutions

### 3.1 Oversampling (Resample Minority)

**Idea:** Create more minority class samples

**Random Oversampling:**
```
Duplicate random minority samples
Effect: More minority in training

Before: [0,0,0,0,0,0,0,0,0,1]  (90-10)
After: [0,0,0,0,0,1,1,1,1,1]   (50-50)
```

**Pros:**
- Simple
- No data loss
- Uses all information

**Cons:**
- Overfitting risk (duplicates exact same samples)
- Model may memorize duplicates
- Not recommended alone

---

### 3.2 Undersampling (Resample Majority)

**Idea:** Remove majority class samples

**Random Undersampling:**
```
Randomly remove majority samples
Effect: Fewer majority in training

Before: [0,0,0,0,0,0,0,0,0,1]  (90-10)
After:  [0,0,0,0,1]             (80-20)
```

**Pros:**
- Fast training (fewer samples)
- Works well sometimes

**Cons:**
- Information loss (discard data)
- May lose important patterns
- Final model trained on biased distribution

---

### 3.3 SMOTE (Synthetic Minority Oversampling Technique)

**Idea:** Generate synthetic minority samples (not just duplicate)

**Algorithm:**
```
For each minority sample:
  1. Find k nearest minority neighbors
  2. Randomly select one neighbor
  3. Create synthetic sample between them
  4. Synthetic = sample + random × (neighbor - sample)
```

**Example (2D):**
```
Minority sample A = (1, 1)
Nearest neighbor B = (3, 2)
Synthetic = (1, 1) + 0.5 × ((3, 2) - (1, 1)) = (2, 1.5)
```

**Pros:**
- No information loss
- Creates diverse new samples
- Better than naive oversampling
- Popular and effective

**Cons:**
- Can create overlapping classes
- Increased dimensionality issues (high-d space)
- More complex than random sampling

**Code:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
model.fit(X_balanced, y_balanced)
```

---

### 3.4 ADASYN (Adaptive Synthetic Sampling)

**Idea:** Intelligent oversampling - focus on hard-to-learn samples

**Difference from SMOTE:**
- SMOTE: Uniform synthetic samples
- ADASYN: More synthetics near difficult boundary

**When Better:** When harder to learn samples more important

**Code:**
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN()
X_balanced, y_balanced = adasyn.fit_resample(X_train, y_train)
```

---

### 3.5 Combination: Over + Under

**SMOTE + Tomek:**
```
1. Oversample (SMOTE)
2. Remove Tomek links (noise at boundary)
3. Effect: Balance + cleaner boundary
```

**Code:**
```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('tomek', TomekLinks())
])

X_balanced, y_balanced = pipeline.fit_resample(X_train, y_train)
```

---

## 4. Algorithm-Level Solutions

### 4.1 Class Weights

**Idea:** Weight minority class higher during training

**Effect:** Minority misclassification penalized more

**Code (sklearn):**
```python
from sklearn.ensemble import RandomForestClassifier

# Automatic balancing
rf = RandomForestClassifier(class_weight='balanced')

# Manual weights
class_weights = {0: 1, 1: 9}  # Minority 9x heavier
rf = RandomForestClassifier(class_weight=class_weights)

rf.fit(X_train, y_train)
```

**How It Works:**
```
Weight_0 = 1 / (frequency_0)
Weight_1 = 1 / (frequency_1)

Example (90-10 split):
Weight_0 = 1 / 0.9 ≈ 1.1
Weight_1 = 1 / 0.1 = 10

Minority errors penalized 10x more
```

**Pros:**
- Simple (one parameter)
- No data duplication
- Works with most algorithms

**Cons:**
- Less effective than resampling
- May need manual tuning
- Not all algorithms support it

---

### 4.2 Threshold Tuning

**Problem:** Default threshold 0.5 biased toward majority

**Solution:** Adjust threshold for business needs

```
Lower threshold (0.2-0.3):
  More samples predicted positive
  Higher Recall, lower Precision
  Good for: Don't want to miss positives

Higher threshold (0.7-0.8):
  Fewer samples predicted positive
  Lower Recall, higher Precision
  Good for: Want high confidence
```

**Process:**
```
1. Get probabilities: P(class=1)
2. Try different thresholds
3. Calculate Precision, Recall, F1 for each
4. Choose based on cost/business needs
```

**Code:**
```python
y_pred_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0, 1, 0.1)
for t in thresholds:
    y_pred = (y_pred_proba >= t).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"t={t:.1f}: Precision={precision:.3f}, Recall={recall:.3f}")
```

---

### 4.3 Anomaly Detection Approach

**Reframe:** Imbalanced as anomaly detection

**Treat minority as anomalies:**
- Isolation Forest
- One-Class SVM
- Autoencoders

**Advantage:** Designed for rare events

**Disadvantage:** May miss patterns if minority has structure

---

## 5. Ensemble & Boosting Solutions

### 5.1 Balanced Bagging

**Idea:** Each bootstrap resample has balanced classes

**Effect:** Each tree trained on balanced data

**Code:**
```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
brf.fit(X_train, y_train)
```

---

### 5.2 EasyEnsemble

**Idea:** Multiple undersampled models averaged

**Process:**
```
1. Undersample majority randomly
2. Train model on balanced subset
3. Repeat many times
4. Ensemble (vote or average)
```

**Effect:** Each model sees different majority subset

**Code:**
```python
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42
)
eec.fit(X_train, y_train)
```

---

### 5.3 RUSBoost (Random Under-Sampling Boosting)

**Idea:** Boosting + undersampling

**Combines:**
- Boosting (AdaBoost benefit)
- Undersampling (speed)

---

### 5.4 SMOTEBoost

**Idea:** Boosting + SMOTE

**Each iteration:**
- Calculate sample weights
- SMOTE to balance
- Train weak learner

**Advantage:** Boosting + intelligent sampling

---

## 6. Ensemble Voting with Different Models

**Idea:** Train models on different resampled versions

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

# Model 1: SMOTE + LR
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train, y_train)
lr = LogisticRegression()
lr.fit(X_smote, y_smote)

# Model 2: Class weights + SVM
svc = SVC(class_weight='balanced', probability=True)
svc.fit(X_train, y_train)

# Voting
voting = VotingClassifier(
    estimators=[('lr', lr), ('svc', svc)],
    voting='soft'
)
voting.fit(X_train, y_train)  # Dummy fit
predictions = voting.predict(X_test)
```

---

## 7. Model-Specific Approaches

### 7.1 Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

# Class weights
lr = LogisticRegression(class_weight='balanced')

# Custom weights (fine-tune)
class_weight = {0: 1, 1: 5}
lr = LogisticRegression(class_weight=class_weight)

lr.fit(X_train, y_train)
```

---

### 7.2 Decision Trees & Random Forests

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
```

---

### 7.3 SVM

```python
from sklearn.svm import SVC

svm = SVC(
    class_weight='balanced',  # or {0: 1, 1: 10}
    probability=True,
    random_state=42
)
svm.fit(X_train, y_train)
```

---

### 7.4 XGBoost

```python
import xgboost as xgb

# Two approaches:
# 1. Scale pos weight
scale_pos_weight = n_negative / n_positive
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# 2. Use balanced class weight interpretation
xgb_model = xgb.XGBClassifier(
    tree_method='hist',
    max_depth=5,
    learning_rate=0.1
)

# With SMOTE
X_balanced, y_balanced = SMOTE().fit_resample(X_train, y_train)
xgb_model.fit(X_balanced, y_balanced)
```

---

## 8. Evaluation for Imbalanced Data

### 8.1 Metrics to Use

**Primary:**
- **Precision-Recall AUC (PR-AUC):** Better than ROC-AUC
- **F1-Score:** Balance Precision and Recall
- **Recall (sensitivity):** How many positives caught

**Secondary:**
- **Precision:** False alarm rate
- **Specificity:** True negative rate
- **Matthews Correlation Coefficient (MCC):** Balanced correlation

**Avoid:**
- **Accuracy:** Misleading for imbalanced
- **ROC-AUC:** Can be high even with poor minority

---

### 8.2 Confusion Matrix Interpretation

```
                Predicted
              Neg    Pos
Actual Neg    TN     FP
       Pos    FN     TP

For imbalanced (1% minority):
- High TN (many correct negatives) OK
- FN and FP more important
- Focus on minority class performance
```

---

### 8.3 Cross-Validation Strategy

**Stratified k-Fold Critical:**

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Each fold preserves class proportion
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
```

**Why Stratified:**
- Each fold has ~same class ratio as overall
- More representative folds
- Stable CV estimates

**Avoid:** Regular k-fold (some folds may lack minorities)

---

## 9. Common Pitfalls & Solutions

### 9.1 Data Leakage with Resampling

**Wrong:**
```python
# WRONG: Resample before split
X_balanced, y_balanced = SMOTE().fit_resample(X, y)
X_train, X_test = train_test_split(X_balanced, y_balanced)
model.fit(X_train, y_train)
# Synthetic samples in test → Inflated performance
```

**Correct:**
```python
# CORRECT: Split first, resample only train
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_balanced, y_train_balanced = SMOTE().fit_resample(X_train, y_train)
model.fit(X_train_balanced, y_train_balanced)
# Test set clean, unbiased evaluation
```

---

### 9.2 Overfitting to Synthetic Samples

**Risk:** SMOTE creates outliers, model overfits

**Mitigation:**
```python
# 1. Use regularization
rf = RandomForestClassifier(
    max_depth=5,  # Limit tree depth
    min_samples_leaf=5
)

# 2. Use SMOTE + Tomek (clean boundary)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('tomek', TomekLinks())
])

# 3. Threshold tuning (reduces overfitting sensitivity)
```

---

### 9.3 Wrong Metric for Decision

**Wrong:**
```
Imbalanced data, maximize Accuracy
Result: High accuracy, low recall
Useless model
```

**Correct:**
```
Define costs first
If FN costly: Maximize Recall
If FP costly: Maximize Precision
If both: Use F1 or PR-AUC
```

---

## 10. Decision Framework

### 10.1 Choosing an Approach

```
Imbalance Ratio?

Mild (2:1 - 10:1):
  → Try class_weight first
  → If not enough, SMOTE

Moderate (10:1 - 100:1):
  → SMOTE recommended
  → Or SMOTE + Tomek
  → Or Balanced Random Forest

Severe (100:1+):
  → SMOTE + Tomek + class weights
  → Or anomaly detection reframing
  → Or ensemble multiple approaches
  → May need domain-specific feature engineering

Has labeled positive samples?
  → SMOTE, resampling, class weights

No positive samples (pure anomaly)?
  → One-class SVM, Isolation Forest
  → PCA reconstruction error
```

---

### 10.2 Step-by-Step Approach

```
1. Understand the cost structure
   - Cost of FN vs FP?
   - Business impact?

2. Choose metric accordingly
   - Optimize for Recall? Precision? F1?

3. Try simple first
   - Class weights (baseline)

4. Progress to resampling
   - SMOTE if simple doesn't work

5. Try ensembles
   - Multiple models, multiple approaches

6. Evaluate properly
   - Stratified CV
   - Use proper metric
   - Test set never in resampling
```

---

## 11. Implementation Example

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc

# Imbalanced dataset
X, y = imbalanced_data

# Split (before any resampling!)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# Pipeline: Scale → SMOTE → RF
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        class_weight='balanced',
        max_depth=10,
        n_estimators=100,
        random_state=42
    ))
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
print(f"F1 (CV): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# Train final model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.named_steps['rf'].predict_proba(
    pipeline.named_steps['smote'].fit_transform(
        pipeline.named_steps['scaler'].transform(X_test)
    )
)[:, 1]

print(f"F1: {f1_score(y_test, y_pred):.3f}")
print(f"PR-AUC: {auc(*precision_recall_curve(y_test, y_pred_proba)[:2]):.3f}")
print(classification_report(y_test, y_pred))
```

---

## 12. Practice Problems

1. **Why Accuracy Fails:** 99% negative, 1% positive. Predict all negative. Why accuracy misleading?

2. **Cost Matrix:** FN cost = $1000, FP cost = $10. What threshold prefer?

3. **SMOTE:** How works? Advantage over random oversampling?

4. **Data Leakage:** When resampling, why must split first?

5. **Metrics:** Choose between Accuracy, Precision, Recall, F1, PR-AUC for imbalanced. Why?

6. **Class Weights:** How computed for 'balanced'? Formula?

7. **Stratified CV:** Why important for imbalanced? What happens without?

8. **Ensemble:** Multiple resampling approaches combined. Advantage?

---

## 13. Key Takeaways

1. **Imbalance problem:** Standard models biased toward majority
2. **Accuracy misleading:** Use Precision, Recall, F1, PR-AUC instead
3. **Cost matrix:** Different errors have different costs
4. **Resampling:** SMOTE better than naive oversampling
5. **Split first:** Never resample before splitting (data leakage)
6. **Class weights:** Simple, effective, no data duplication
7. **Threshold tuning:** Adjust for business needs
8. **Evaluation:** Stratified CV, proper metrics, don't use test for tuning
9. **Ensemble:** Combine multiple approaches for better results
10. **Framework:** Mild → class weights, Moderate → SMOTE, Severe → combination approaches

---

**Next Topic:** Feature Engineering (say "next" to continue)