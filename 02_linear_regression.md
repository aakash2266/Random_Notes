# Linear Regression - Complete Guide

## 1. Fundamentals of Linear Regression

### 1.1 What is Linear Regression?

**Definition:** Supervised learning algorithm modeling linear relationship between input features (X) and continuous target (y).

**Equation:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

Where:
- y: Target variable (dependent)
- x₁, x₂, ..., xₚ: Features (independent variables)
- β₀: Intercept (y when all x=0)
- β₁, β₂, ..., βₚ: Coefficients (slopes)
- ε: Error term (residuals)

**Assumptions:**
1. Linear relationship between X and y
2. Independence: Observations independent
3. Homoscedasticity: Constant error variance
4. Normality: Errors normally distributed
5. No multicollinearity: Features not highly correlated

---

### 1.2 Simple vs Multiple Linear Regression

**Simple Linear Regression (1 feature):**
```
y = β₀ + β₁x + ε
```
- Easy to visualize (2D plot)
- Useful for understanding relationships
- Limited real-world applicability

**Multiple Linear Regression (p features):**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```
- More realistic (multiple factors)
- Hard to visualize (>2D)
- Standard in practice

---

## 2. Cost Function & Optimization

### 2.1 Residuals & Sum of Squared Errors (SSE)

**Residual:**
```
e_i = y_i - ŷ_i
```
- Difference between actual and predicted
- Positive: Underpredicted
- Negative: Overpredicted

**Sum of Squared Errors (SSE):**
```
SSE = Σ(y_i - ŷ_i)² = Σe_i²
```
- Total prediction error
- Why square? Penalizes large errors more, removes sign

**Mean Squared Error (MSE):**
```
MSE = SSE / n = Σ(y_i - ŷ_i)² / n
```
- Average squared error
- Lower is better

---

### 2.2 Ordinary Least Squares (OLS)

**Goal:** Minimize SSE

**Solution (Matrix form):**
```
β = (X^T X)^(-1) X^T y
```

Where:
- X: Feature matrix (n×p)
- y: Target vector (n×1)
- β: Coefficient vector (p×1)

**Closed-form solution:**
- Analytical (exact) solution
- No iteration needed
- Works for small-medium datasets
- Computationally expensive for very large datasets (O(p³))

**Properties:**
- Unbiased: E[β̂] = β
- Minimum variance (Best Linear Unbiased Estimator - BLUE)
- Assumes X^T X invertible (no multicollinearity)

---

### 2.3 Gradient Descent (Alternative)

**Idea:** Iteratively move in direction of steepest descent

**Update rule:**
```
β_new = β_old - α × ∇J(β)
```

Where:
- α: Learning rate (step size)
- ∇J(β): Gradient of cost function

**Gradient (for MSE):**
```
∂J/∂β_j = -2/n × Σ(y_i - ŷ_i) × x_ij
```

**Advantages:**
- Works with very large datasets
- Iterative (can stop early)
- Handles regularization naturally

**Disadvantages:**
- Requires learning rate tuning
- Can converge to local minimum (non-convex losses)
- Slower per iteration than OLS (but fewer total iterations for large n)

**Variants:**
- Batch GD: All samples per iteration (slow for huge data)
- Stochastic GD (SGD): One sample per iteration (noisy, fast)
- Mini-batch GD: k samples per iteration (balance)

---

## 3. Model Evaluation Metrics

### 3.1 Regression Metrics

**R² (Coefficient of Determination):**
```
R² = 1 - (SSE / SST)
```
Where SST = Σ(y_i - ȳ)² (total sum of squares)

- Range: [0, 1] (can be negative if model worse than mean)
- R² = 1: Perfect prediction
- R² = 0: Model as good as just predicting mean
- Interpretation: % of variance in y explained by model

**Adjusted R²:**
```
Adj R² = 1 - [(1-R²) × (n-1)/(n-p-1)]
```
- Penalizes adding more features
- Prevents overfitting bias of R²
- Always use for model comparison

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ|y_i - ŷ_i|
```
- Average absolute deviation
- More interpretable than MSE (same units as y)
- Robust to outliers

**Root Mean Squared Error (RMSE):**
```
RMSE = √(1/n × Σ(y_i - ŷ_i)²)
```
- Square root of MSE
- Penalizes large errors more than MAE
- Same units as y

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (1/n) × Σ|y_i - ŷ_i| / |y_i| × 100%
```
- Percentage error (scale-independent)
- Useful for comparing across different scales
- Problem: Undefined when y_i = 0

---

### 3.2 Metric Selection

| Metric | When to Use | Robustness |
|--------|------------|-----------|
| MSE/RMSE | General purpose, penalizes outliers | Outlier-sensitive |
| MAE | Outliers present, interpretability | Outlier-robust |
| R² | Model fit percentage | Affected by outliers |
| MAPE | Relative error, different scales | Undefined at zero |

---

## 4. Assumptions & Diagnostics

### 4.1 Linearity

**Assumption:** Relationship between X and y is linear

**Check:**
- Scatter plot: Points roughly follow line
- Residual plot: No pattern (residuals random)
- Add polynomial features if non-linear

**Issue:** Model underfits if relationship non-linear
**Solution:** Feature engineering, polynomial features

---

### 4.2 Independence

**Assumption:** Observations independent (no autocorrelation)

**Check:**
- Durbin-Watson test (time series)
- DW near 2: Independent
- DW << 2: Positive autocorrelation
- DW >> 2: Negative autocorrelation

**Issue:** Biased standard errors, inefficient estimates
**Solution:** 
- Time series models (ARIMA) if temporal
- Remove duplicates/correlated samples

---

### 4.3 Homoscedasticity (Constant Variance)

**Assumption:** Error variance constant across predictions

**Check:**
- Residual plot: Vertical spread roughly equal at all x values
- Scale-location plot: No trend
- Breusch-Pagan test

**Issue:** Heteroscedasticity
- Biased standard errors
- Unreliable confidence intervals
- OLS not efficient

**Solution:**
- Weighted Least Squares (WLS)
- Robust standard errors
- Transform y (e.g., log transformation)

---

### 4.4 Normality of Residuals

**Assumption:** Errors normally distributed

**Check:**
- Q-Q plot: Points on diagonal = normal
- Histogram: Bell-shaped
- Shapiro-Wilk test

**Why important:**
- Confidence intervals derived from normality
- t-tests assume normality
- Point estimates (β̂) okay even if violated (Large n)

**Issue:** Non-normal errors
**Solution:**
- Transform y (log, sqrt)
- Use robust methods
- Large samples (CLT helps)

---

### 4.5 No Multicollinearity

**Assumption:** Features not highly correlated

**Check:**
- Correlation matrix: r < 0.8-0.9
- VIF (Variance Inflation Factor) < 5-10
- VIF = 1/(1-R²) where R² from regressing feature on others

**Issue:** Multicollinearity
- Unstable coefficients (small data change → large coefficient change)
- Large standard errors
- Hard to interpret feature importance
- Predictions still okay (if collinear features stay together)

**Solution:**
- Remove one of correlated features
- PCA (create uncorrelated components)
- Regularization (Ridge, Lasso)
- Domain knowledge (keep interpretable feature)

---

## 5. Regularization Techniques

### 5.1 Why Regularization?

**Problem:** Linear regression can overfit
- Too many features relative to samples (p > n)
- Multicollinearity causes large coefficients
- Model memorizes noise in training data

**Solution:** Add penalty for large coefficients

---

### 5.2 Ridge Regression (L2 Regularization)

**Cost Function:**
```
J(β) = SSE + λ × Σβ_j²
```

Where λ (lambda) = regularization strength

**Characteristics:**
- Shrinks coefficients toward zero (but not to zero)
- All features remain in model
- Less interpretable (no automatic feature selection)
- Handles multicollinearity well

**β (Ridge) = (X^T X + λI)^(-1) X^T y**
- λ increases → More shrinkage
- λ=0 → OLS
- λ=∞ → β → 0

**When to use:**
- Multicollinearity present
- Many features, want to keep all
- Predictions priority over interpretability

---

### 5.3 Lasso Regression (L1 Regularization)

**Cost Function:**
```
J(β) = SSE + λ × Σ|β_j|
```

**Characteristics:**
- Shrinks coefficients to exactly zero
- Automatic feature selection
- Some features eliminated from model
- More interpretable

**Why L1 shrinks to zero?**
- L1 penalty creates "corners" in constraint region
- Solution path hits corners (zero coefficients)
- L2 penalty creates "circles" (coefficients non-zero)

**Advantages over Ridge:**
- Feature selection (sparse model)
- Better interpretability
- Fewer features → faster prediction

**When to use:**
- Feature selection needed
- Many irrelevant features
- Interpretability important
- p > n (more features than samples)

---

### 5.4 Elastic Net (L1 + L2)

**Cost Function:**
```
J(β) = SSE + λ₁ × Σ|β_j| + λ₂ × Σβ_j²
```

**Combines strengths:**
- L1: Feature selection
- L2: Handles multicollinearity

**When to use:**
- Many correlated features
- Want feature selection AND multicollinearity handling
- Balanced approach

**Parameters:**
- α: L1 ratio (0=Ridge, 1=Lasso)
- λ: Overall strength

---

### 5.5 Regularization Tuning

**Cross-Validation:**
1. Grid search λ values: [0.001, 0.01, 0.1, 1, 10, 100]
2. For each λ: k-fold CV score
3. Choose λ with best CV score

**Trade-off:**
- Low λ: Underfitting (high bias)
- High λ: Simpler model, may underfit
- Optimal λ: Balance bias-variance

---

## 6. Advanced Topics

### 6.1 Polynomial Regression

**Idea:** Add polynomial features to capture non-linearity

**Example (degree=2):**
```
y = β₀ + β₁x + β₂x² + ε
```

**How:**
- PolynomialFeatures(degree=2) generates x, x²
- Apply linear regression to expanded features
- Still linear regression! (just different features)

**Advantages:**
- Captures non-linear patterns
- Simple to implement
- Interpretable

**Disadvantages:**
- Overfitting risk (high degree)
- Extrapolation poor (outside training range)
- Multicollinearity (x and x² correlated)

**Best practice:**
- Start degree=2, increase if needed
- Use regularization
- Validate on test set

---

### 6.2 Feature Interactions

**Idea:** Add products of features

**Example:**
```
y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁×x₂) + ε
```

**When to use:**
- Domain knowledge suggests interaction
- E.g., Price depends on location×size (bigger effect in expensive areas)

**How:**
- Create manually: df['x1_x2'] = df['x1'] * df['x2']
- Or use PolynomialFeatures(degree=2, include_bias=False)

---

### 6.3 Categorical Features

**One-Hot Encoding:**
- Convert categorical to binary columns
- Example: Color (Red, Blue, Green) → 3 columns

**Dummy Variable Trap:**
- Include k-1 dummy variables (not k)
- Avoids perfect multicollinearity
- Drop one category (baseline reference)

**Example:**
- Color: Red, Blue, Green → Create Red, Blue (drop Green)
- Green implicitly: Red=0, Blue=0

---

## 7. Implementation Considerations

### 7.1 Feature Scaling

**When necessary:**
- Gradient descent: Always scale (affects learning rate)
- OLS: Not necessary (scale-invariant)
- Interpretation: Easier with scaled features

**Methods:**
- Standardization: (x - mean) / std (z-score)
- Normalization: (x - min) / (max - min) range [0,1]

---

### 7.2 Train-Test Split

**Typical split:** 80-20 or 70-30 (train-test)

**Best practice:**
1. Split data first
2. Fit model on train
3. Evaluate on test
4. Never use test for tuning (avoid overfitting bias)

**Cross-Validation:**
- Better estimate of generalization
- k-fold CV (typically k=5 or 10)
- More robust than single split

---

### 7.3 Handling Missing Values

**Options:**
1. Drop rows with missing (simple, loses data)
2. Impute mean/median (simple, assumes linear pattern)
3. Impute with model (complex, may overfit)
4. Keep as feature (indicator for missingness)

**sklearn:**
- SimpleImputer(strategy='mean')
- IterativeImputer (more sophisticated)

---

## 8. Disadvantages & Limitations

### 8.1 Linear Regression Limitations

**Assumes Linearity:**
- Poor for non-linear relationships
- Polynomial features help (but add complexity)

**Sensitive to Outliers:**
- Outliers pull regression line
- Solution: Robust regression, remove/impute outliers

**Assumes Continuous Target:**
- Classification problems need logistic regression
- Otherwise predictions can be <0 or >1 for binary

**Multicollinearity:**
- Unstable coefficients
- Hard to interpret individual feature effects
- Predictions still okay

**Limited to Relationships in Data:**
- Can't learn patterns not present
- More data or features needed

---

### 8.2 When NOT to Use Linear Regression

1. **Non-linear relationship:** Use polynomial features, tree models, or neural networks
2. **Classification:** Use logistic regression or other classifiers
3. **Very high-dimensional (p >> n):** Use regularization or feature selection
4. **Extreme outliers:** Use robust regression or tree-based models
5. **Time series with trend/seasonality:** Use ARIMA, Prophet
6. **Complex interactions:** Tree models learn automatically

---

## 9. Interpretation of Coefficients

### 9.1 Simple Linear Regression

**y = β₀ + β₁x**

**β₀ (Intercept):**
- Predicted y when x=0
- Sometimes has no meaning (outside data range)

**β₁ (Slope):**
- "One unit increase in x → β₁ unit increase in y"
- Interpretation depends on scale

---

### 9.2 Multiple Regression

**y = β₀ + β₁x₁ + β₂x₂ + ...**

**β_j (Coefficient for x_j):**
- "Holding other variables constant, one unit increase in x_j → β_j unit increase in y"
- Ceteris paribus (all else equal) interpretation

**Important:** 
- Causation requires experiment
- Correlation doesn't imply causation
- Confounders can bias interpretation

---

### 9.3 Standardized Coefficients

**Why standardize?**
- Compare relative importance (same scale)
- Unit-dependent coefficients hard to compare

**Standardization:**
```
x_std = (x - mean) / std
y_std = (y - mean) / std
```

**Fit model on standardized variables:**
```
y_std = β₁*x₁_std + β₂*x₂_std + ...
```

**Interpretation:**
- "One std increase in x_j → β_j* std increase in y"
- |β_j*| larger → More important feature

---

## 10. Common Mistakes & Best Practices

### 10.1 Common Mistakes

1. **Using test set for validation:** Introduces overfitting bias. Use k-fold CV on training data.

2. **Ignoring assumptions:** Check linearity, independence, homoscedasticity, normality, multicollinearity.

3. **Not scaling features:** Gradient descent sensitive to scale. Always scale for GD.

4. **Too many features:** Overfitting, multicollinearity. Use regularization or feature selection.

5. **Not checking for outliers:** Outliers bias regression line. Remove or handle robustly.

6. **Causation from correlation:** Just because x and y correlated doesn't mean x causes y.

7. **Extrapolation:** Linear regression predicts poorly outside training range.

---

### 10.2 Best Practices

1. **EDA First:** Scatter plots, histograms, correlation matrix. Understand data before modeling.

2. **Feature Engineering:** Create meaningful features using domain knowledge.

3. **Regularization:** Use Ridge/Lasso for high-dimensional or multicollinear data.

4. **Cross-Validation:** Estimate generalization error reliably.

5. **Check Assumptions:** Residual plots, Q-Q plot, VIF. Violate with caution.

6. **Compare Models:** Baseline vs polynomial vs regularized. Pick based on CV score.

7. **Interpret Carefully:** Remember correlation ≠ causation. Report uncertainty (confidence intervals).

8. **Document:** Store coefficients, preprocessing parameters, performance metrics for reproducibility.

---

## 11. Linear Regression in sklearn

### 11.1 Basic Usage

```python
from sklearn.linear_model import LinearRegression

# Simple regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Coefficients
print(model.coef_)      # β₁, β₂, ...
print(model.intercept_) # β₀

# Score (R²)
print(model.score(X_test, y_test))
```

### 11.2 Ridge & Lasso

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

### 11.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Grid search for best alpha
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
model = Ridge()
grid = GridSearchCV(model, {'alpha': alphas}, cv=5)
grid.fit(X_train, y_train)

print(f"Best alpha: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
```

---

## 12. Practice Problems

1. **Explain OLS:** How does OLS minimize SSE? What's the closed-form solution?

2. **R² Interpretation:** Model has R²=0.75. What does this mean?

3. **Regularization Trade-off:** How does increasing λ in Ridge affect bias and variance?

4. **Multicollinearity:** Features x₁ and x₂ have correlation 0.95. What problems does this cause? How to detect (VIF)?

5. **Assumptions Violation:** How would you detect if homoscedasticity violated? How to fix?

6. **Feature Scaling:** Why scale features for gradient descent but not OLS?

7. **Outliers:** How would outliers affect regression line and R²? How to handle?

8. **Coefficient Interpretation:** Model: `price = 50000 + 200*sqft + 5000*bedrooms`. Interpret each coefficient.

---

## 13. Key Takeaways

1. **Linear regression models:** y = β₀ + Σβ_j×x_j + ε
2. **OLS minimizes SSE:** Closed-form solution or gradient descent
3. **Assumptions matter:** Check linearity, independence, homoscedasticity, normality, no multicollinearity
4. **Regularization prevents overfitting:** Ridge (all features), Lasso (feature selection), Elastic Net (both)
5. **Metrics:** R² (variance explained), MSE/RMSE (error), MAE (robust)
6. **Interpretation:** β_j = effect of x_j on y, holding others constant
7. **Causation requires experiments:** Correlation ≠ causation
8. **Always validate:** Use k-fold CV on training data, evaluate on held-out test

---

**Next Topic:** Logistic Regression (say "next" to continue)