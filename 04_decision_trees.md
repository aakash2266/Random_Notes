# Decision Trees - Complete Guide

## 1. Fundamentals of Decision Trees

### 1.1 What is a Decision Tree?

**Definition:** Tree-structured model that recursively partitions feature space into regions, with each region assigned a class label (classification) or constant value (regression).

**Structure:**
- **Root node:** Top node, all samples
- **Internal nodes:** Decision nodes (split on feature)
- **Leaf nodes:** Terminal nodes (predictions)
- **Branches:** Connect nodes (feature conditions)

**Visual Example:**
```
                    [All Data]
                        |
                [Age < 30?]
               /            \
            Yes             No
             |               |
        [Play=Yes]    [Income < 50k?]
                        /            \
                      Yes            No
                       |              |
                 [Play=No]     [Play=Yes]
```

**Key Point:** Tree-based, interpretable, non-parametric algorithm

---

### 1.2 Decision Trees for Classification vs Regression

| Aspect | Classification Tree | Regression Tree |
|--------|-------------------|-----------------|
| Target | Categorical (classes) | Continuous (numbers) |
| Leaf Value | Class label (majority vote) | Mean/median value |
| Prediction | Class | Numeric value |
| Impurity Metric | Gini/Entropy | MSE/MAE |
| Output | Discrete | Continuous |
| Examples | Spam/Not spam | House price |

---

## 2. Tree Building: Splitting Criteria

### 2.1 Information Gain & Entropy

**Entropy (Measure of Disorder):**
```
Entropy(S) = -Σ p_i × log₂(p_i)
```

Where:
- S: Dataset
- p_i: Proportion of class i
- log₂: Base-2 logarithm

**Interpretation:**
- Entropy = 0: Pure (all one class)
- Entropy = 1: Maximum disorder (equal classes)
- Higher entropy → More mixed

**Example (Binary):**
```
- [100 Yes, 0 No]: Entropy = 0 (pure)
- [50 Yes, 50 No]: Entropy = 1 (maximum)
- [75 Yes, 25 No]: Entropy = -0.75×log₂(0.75) - 0.25×log₂(0.25) ≈ 0.81
```

**Information Gain:**
```
IG(S, A) = Entropy(S) - Σ (|S_v| / |S|) × Entropy(S_v)
```

Where:
- A: Attribute (feature)
- S_v: Subset where attribute=v
- Higher IG → Better split

**Interpretation:**
- IG measures entropy reduction from split
- Choose feature with highest IG
- IG = 0: No improvement (no split)

---

### 2.2 Gini Impurity (Alternative to Entropy)

**Gini Index:**
```
Gini(S) = 1 - Σ p_i²
```

**Interpretation:**
- Gini = 0: Pure node
- Gini = 0.5 (binary): Maximum disorder
- Lower Gini → Purer

**Gini vs Entropy:**
- Both measures of impurity
- Gini: Computationally faster (no log)
- Entropy: Information-theoretic interpretation
- Usually similar results

**Gini Gain (Weighted Gini Reduction):**
```
GG(S, A) = Gini(S) - Σ (|S_v| / |S|) × Gini(S_v)
```

---

### 2.3 Gain Ratio (Handles Feature Cardinality Bias)

**Problem with Information Gain:**
- Biased toward high-cardinality features
- Feature with many unique values → Higher IG (overfitting)

**Solution: Gain Ratio**
```
GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)
```

Where:
```
SplitInfo(S, A) = -Σ (|S_v| / |S|) × log₂(|S_v| / |S|)
```

**Effect:**
- Penalizes high-cardinality features
- Prevents overfitting
- Used in C4.5 algorithm

---

### 2.4 Chi-Square Test (Statistical Split)

**Idea:** Statistically test if split is significant

**Chi-Square Statistic:**
```
χ² = Σ (Observed - Expected)² / Expected
```

**Process:**
1. Calculate expected frequency (under independence)
2. Compare observed vs expected
3. Higher χ² → More likely split (significant)

**When to use:**
- Statistical rigor desired
- Avoid spurious splits
- More conservative

---

## 3. Tree Growing Algorithms

### 3.1 ID3 (Iterative Dichotomiser 3)

**Algorithm:**
```
1. Start with all data at root
2. For each node:
   a. Calculate information gain for all features
   b. Split on feature with highest IG
   c. Recursively apply to child nodes
3. Stop when: Pure nodes or no improvement
```

**Characteristics:**
- Uses entropy/information gain
- Greedy (locally optimal, not global)
- Binary or multi-way splits
- No pruning

---

### 3.2 C4.5 (Improvement of ID3)

**Improvements:**
1. **Gain Ratio:** Handles high-cardinality bias
2. **Pruning:** Reduces overfitting
3. **Missing Values:** Handles imputation
4. **Continuous Features:** Converts to binary splits

**Continuous Feature Split:**
- For feature x, try all possible split points
- Choose point minimizing impurity

**Example:** Age splits → Age ≤ 30, 30 < Age ≤ 50, Age > 50

---

### 3.3 CART (Classification And Regression Trees)

**Characteristics:**
- Binary splits only (x ≤ t vs x > t)
- Gini index (classification) or MSE (regression)
- Pruning for robustness
- Handles missing via surrogate splits

**Why Binary?**
- Simpler, easier interpretation
- More robust (less overfitting risk)
- Computational efficiency

---

## 4. Stopping Criteria & Pruning

### 4.1 Early Stopping (When to Stop Growing)

**Common Criteria:**
1. **Pure Node:** All samples one class (Entropy=0)
2. **Max Depth:** Reached predefined depth limit
3. **Min Samples:** Node has fewer than min_samples_split
4. **Min Impurity Decrease:** Split reduces impurity below threshold
5. **No Improvement:** No split improves metric

**sklearn Parameters:**
```python
DecisionTreeClassifier(
    max_depth=5,               # Max depth
    min_samples_split=2,       # Min samples to split
    min_samples_leaf=1,        # Min samples in leaf
    min_impurity_decrease=0    # Min impurity reduction
)
```

---

### 4.2 Pruning (Post-Pruning)

**Problem:** Fully grown trees overfit

**Solution:** Remove nodes after tree built

**Error-Based Pruning:**
1. Grow full tree on training data
2. For each node, calculate error on validation set
3. Remove node if error decreases (or increases < threshold)
4. Recursively prune

**Cost-Complexity Pruning:**
```
Error_c = Error + c × #leaves
```

Where c = complexity penalty

- c=0: Full tree
- c → ∞: Single node (root)
- Find optimal c via cross-validation

**Why Pruning?**
- Reduces overfitting
- Simpler, more interpretable trees
- Better generalization

---

### 4.3 Reduced Error Pruning

**Process:**
1. Grow full tree on training data
2. Use separate validation set
3. Bottom-up: For each node, if removing subtree doesn't hurt validation error → Remove
4. Stop when no node removal helps

**Advantage:** Simple, effective

**Disadvantage:** Requires validation set (loses training data)

---

## 5. Handling Continuous & Categorical Features

### 5.1 Continuous Features (Numeric)

**Approach:**
- Sort feature values
- Try split at each unique value (or midpoints)
- Choose split minimizing impurity

**Example (Age):**
```
Data: [25, 35, 40, 45, 50]
Try splits: Age ≤ 25, Age ≤ 35, Age ≤ 40, ...
Choose split with best Gini/IG
```

**Computational Cost:** O(n log n) per feature (sort + try splits)

---

### 5.2 Categorical Features (Discrete)

**Approach 1: Multi-way Split**
- Split into k branches (one per category)
- Choose subset minimizing impurity

**Example (Color: Red, Blue, Green):**
```
If Color = Red: Left
If Color = Blue or Green: Right
```

**Approach 2: Binary Split (C4.5, CART)**
- Group categories into two groups
- Try all possible groupings (exponential in categories)
- Choose grouping minimizing impurity

**Approach 3: Ordinal Encoding**
- If ordering exists (Low, Medium, High)
- Treat as continuous: Low ≤ Medium < High

---

### 5.3 Missing Values

**Approach 1: Drop**
- Remove samples with missing
- Simple but loses data

**Approach 2: Imputation**
- Mean/median (numeric)
- Mode (categorical)
- Model-based imputation

**Approach 3: Surrogate Splits (CART)**
- Identify "surrogate" features similar performance
- Use surrogate if primary missing
- More sophisticated

**sklearn:** Default drops missing at each split

---

## 6. Decision Trees for Regression

### 6.1 Regression Trees (CART for Regression)

**Difference from Classification:**
- Leaf value: Mean (or median) of target in region
- Impurity metric: MSE or MAE (not Gini/Entropy)
- Prediction: Continuous value

**Splitting Criterion (MSE):**
```
MSE_split = Σ (Size_i / Size_total) × Var(y_i)
```

Where:
- Var(y_i): Variance in each child region
- Choose split minimizing MSE_split

---

### 6.2 Regression Tree Example

**Data:** (x, y) pairs
```
x = [1, 2, 3, 4, 5]
y = [2, 4, 3, 5, 6]
```

**Split at x ≤ 2.5:**
```
Left (x ≤ 2.5): y = [2, 4] → mean = 3
Right (x > 2.5): y = [3, 5, 6] → mean = 4.67
```

**Predictions:**
```
If x ≤ 2.5: Predict 3
If x > 2.5: Predict 4.67
```

---

## 7. Advantages of Decision Trees

### 7.1 Interpretability

**Clear Explanations:**
- Easy to visualize
- Follow path: "If x₁ > 5 AND x₂ ≤ 10 → Class A"
- Non-technical stakeholders understand

**Feature Importance:**
- How much each feature contributes to predictions
- Gini decrease across all splits

---

### 7.2 Handling Different Data Types

**Mixed Features:**
- Numeric, categorical, ordinal all together
- No scaling needed
- Handles missing naturally (with surrogates)

**Non-linear Relationships:**
- Automatic capture of interactions
- Polynomial features not needed
- Axis-aligned boundaries

---

### 7.3 No Assumptions

- No linearity assumption
- No distribution assumptions
- Works with any data

---

### 7.4 Feature Selection

- Automatically identifies important features
- Can ignore irrelevant features
- Feature importance from Gini/entropy decrease

---

## 8. Disadvantages of Decision Trees

### 8.1 Overfitting

**Problem:** Trees grow until pure (memorize noise)

**Signature:** High training accuracy, low test accuracy

**Solution:**
- Pruning (remove branches)
- Early stopping (max_depth, min_samples_split)
- Ensemble methods (Random Forest, Gradient Boosting)

---

### 8.2 Instability

**Problem:** Small data changes → Large tree changes

**Example:**
- Change 1 sample → Different splits throughout tree
- Coefficients in LR stable; trees unstable

**Solution:**
- Ensemble methods average many trees
- Reduce tree complexity (limit depth)

---

### 8.3 Bias Toward Large-Cardinality Features

**Problem:** Features with many unique values → Higher IG → Overused

**Example:** ID column (unique per sample) → Perfect split (useless)

**Solutions:**
- Gain ratio (penalizes cardinality)
- Feature selection before training
- Limit max_features

---

### 8.4 Greedy Algorithm

**Problem:** Locally optimal ≠ globally optimal

**Example:** Feature A not best first split, but best combined with B later

**Result:** Trees sometimes suboptimal

**Note:** Finding globally optimal tree is NP-complete

---

### 8.5 Axis-Aligned Boundaries

**Problem:** Can't capture diagonal boundaries efficiently

**Example:** y = x pattern
```
Tree needs many splits (staircase approximation)
Linear model: Single line (efficient)
```

**Solution:** Feature engineering (x-y interactions) or other models

---

## 9. Feature Importance

### 9.1 Gini-Based Importance

**Idea:** Sum Gini decrease from each feature across all splits

**Formula:**
```
Importance(f) = Σ Gini_decrease_using_f / Total_Gini_decrease
```

**Normalized:** Sum to 1.0

**Interpretation:**
- Feature used in important splits (high in tree, large samples) → High importance
- Feature rarely used → Low importance

**Advantages:**
- Fast (computed during training)
- Built into sklearn

**Disadvantages:**
- Biased toward high-cardinality features
- Biased toward correlated features (one chosen, others ignored)
- Can miss non-linear interactions

---

### 9.2 Permutation Importance

**Idea:** Shuffle feature values, measure accuracy drop

**Process:**
1. Train model
2. For each feature:
   a. Randomly shuffle column
   b. Measure accuracy drop (importance)
3. Unshuffle, repeat for next feature

**Advantages:**
- Model-agnostic (works any model)
- No bias toward cardinality
- Captures interactions

**Disadvantages:**
- Slow (shuffle + predict for each feature)
- Can be negative (feature hurts model)

---

### 9.3 SHAP Values

**Idea:** Game-theoretic fair attribution (Shapley values)

**Interpretation:** Each feature's contribution to prediction

**Advantages:**
- Theoretically sound (Shapley axioms)
- Per-sample explanations (not just global)
- Handles interactions

**Disadvantages:**
- Computationally expensive
- Complex to understand

**sklearn:** Use shap library

---

## 10. Hyperparameter Tuning

### 10.1 Key Hyperparameters

**Tree Depth:**
- `max_depth`: Maximum tree depth
- Lower → Simpler, underfitting risk
- Higher → Complex, overfitting risk
- Typical: 5-20

**Leaf Size:**
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf
- Higher → Simpler tree, reduce overfitting
- Typical: 2-10

**Impurity:**
- `criterion`: 'gini' or 'entropy' (classification)
- Usually similar results

**Features:**
- `max_features`: Features considered per split
- 'sqrt', 'log2', None (all)
- Limits feature consideration, adds randomness

---

### 10.2 Tuning Strategy

**Cross-Validation:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {
    'max_depth': [3, 5, 7, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}

dt = DecisionTreeClassifier()
grid = GridSearchCV(dt, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

**Process:**
1. Start with defaults
2. Tune max_depth first (biggest impact)
3. Tune min_samples_split/leaf
4. Final CV score

---

## 11. Tree Visualization & Interpretation

### 11.1 Visualizing Trees

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(dt, 
          feature_names=X.columns,
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True)
plt.show()
```

**Tree Components:**
- **Gini/Entropy:** Impurity at node
- **Samples:** Number of samples in node
- **Value:** Class counts [n_class_0, n_class_1, ...]
- **Color:** Class distribution (darker = purer)

---

### 11.2 Text Representation

```python
from sklearn.tree import export_text

tree_rules = export_text(dt, feature_names=list(X.columns))
print(tree_rules)
```

**Output Example:**
```
|--- x1 <= 5.0
|   |--- x2 <= 3.0
|   |   |--- class: 0
|   |--- x2 > 3.0
|   |   |--- class: 1
|--- x1 > 5.0
|   |--- class: 1
```

---

## 12. Decision Trees for Multiclass Classification

### 12.1 Multiclass Splits

**Gini for Multiclass:**
```
Gini(S) = 1 - Σ p_i²
```

Where p_i = proportion of class i (generalizes binary)

**Entropy for Multiclass:**
```
Entropy(S) = -Σ p_i × log₂(p_i)
```

Same generalization

**Leaf Prediction:** Majority class

---

### 12.2 Handling K Classes

**One-vs-All Approach:**
- Train k trees (tree_i: class_i vs rest)
- Choose tree with highest probability

**Direct Multiclass:**
- Single tree, k classes at leaves
- More efficient (one model)
- sklearn default: Single tree

---

## 13. Comparison with Other Models

| Aspect | Decision Tree | Linear Model | SVM | Neural Network |
|--------|--------------|-------------|-----|----------------|
| Interpretability | Very High | High | Low | Very Low |
| Non-linearity | Automatic | Manual | Via Kernel | Learned |
| Training Speed | Fast | Fast | Slow | Very Slow |
| Prediction Speed | Fast | Fast | Medium | Medium |
| Handles Categorical | Native | Via Encoding | Via Encoding | Via Encoding |
| Scaling Needed | No | Yes | Yes | Yes |
| Stability | Low | High | High | High |
| Overfitting Risk | High | Medium | Medium | High |

---

## 14. Implementation in sklearn

### 14.1 Classification

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Create and train
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt.fit(X_train, y_train)

# Predict
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(dt, X_train, y_train, cv=5).mean()}")
print(f"\n{classification_report(y_test, y_pred)}")
```

### 14.2 Regression

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create and train
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train, y_train)

# Predict
y_pred = dt_reg.predict(X_test)

# Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R²: {r2_score(y_test, y_pred)}")
```

### 14.3 Feature Importance

```python
# Get importance
importance = dt.feature_importances_

# Sort and visualize
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(importance_df)

# Plot
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.show()
```

### 14.4 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier()
grid = GridSearchCV(dt, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_dt = grid.best_estimator_
print(f"Test Accuracy: {best_dt.score(X_test, y_test)}")
```

---

## 15. Handling Common Issues

### 15.1 Overfitting

**Symptom:** Train accuracy 99%, test accuracy 70%

**Solutions:**
```python
# 1. Reduce complexity
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20)

# 2. Prune (sklearn doesn't have built-in post-pruning, but limiting depth helps)

# 3. Use ensemble (Random Forest, Gradient Boosting)
```

---

### 15.2 Class Imbalance

**Problem:** Tree biased toward majority class

**Solutions:**
```python
# 1. Class weights
dt = DecisionTreeClassifier(class_weight='balanced')

# 2. Sampling (oversample minority, undersample majority)

# 3. Threshold tuning on probabilities
y_pred_proba = dt.predict_proba(X_test)
y_pred_custom = (y_pred_proba[:, 1] >= 0.3).astype(int)
```

---

### 15.3 Large Trees

**Problem:** Tree complex, hard to interpret

**Solutions:**
```python
# 1. Limit max_depth
dt = DecisionTreeClassifier(max_depth=5)

# 2. Increase min_samples
dt = DecisionTreeClassifier(min_samples_split=50, min_samples_leaf=20)

# 3. Feature selection before training
```

---

## 16. Practice Problems

1. **Information Gain:** Dataset [80 Y, 20 N]. Split: Left [60Y, 5N], Right [20Y, 15N]. Calculate IG.

2. **Gini vs Entropy:** When to use each? Differences?

3. **Overfitting:** Why do deep trees overfit? How to prevent?

4. **Feature Importance:** Why high-cardinality features biased? How fix?

5. **Pruning:** What is error-based pruning? How works?

6. **Continuous Features:** How handle numeric features in tree?

7. **Multiclass:** How does tree handle 3+ classes? Leaf value?

8. **Imbalance:** 1% positive, 99% negative. How adjust tree?

---

## 17. Key Takeaways

1. **Tree structure:** Root → Internal nodes (splits) → Leaves (predictions)
2. **Splitting criteria:** Entropy/Gini measure disorder; Information Gain measures reduction
3. **Algorithms:** ID3 (IG), C4.5 (Gain Ratio), CART (Gini, binary)
4. **Overfitting:** Major issue; solve via pruning, depth limits, or ensembles
5. **Interpretability:** Excellent; follow path to understand prediction
6. **Feature Importance:** Gini-based (fast but biased), Permutation (model-agnostic), SHAP (theoretically sound)
7. **Non-linear:** Automatic capture of interactions, axis-aligned boundaries
8. **Stability:** Low; small data changes → big tree changes
9. **Greedy:** Locally optimal, may not be global optimum
10. **Practical:** Often used in ensembles (RF, GB) for better performance

---

**Next Topic:** k-Nearest Neighbors (say "next" to continue)