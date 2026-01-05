# Statistics for Data Science: Comprehensive Notes with Example Scenarios

## 1. Descriptive Statistics

### 1.1 Measures of Central Tendency

**Mean (Average)**
- Definition: Sum of all values divided by the number of values
- Formula: μ = (Σx) / n
- Use Case: When data is normally distributed without extreme outliers
- Example: E-commerce product review scores average
  - Reviews: [4, 5, 3, 4, 5, 2, 4]
  - Mean = (4+5+3+4+5+2+4) / 7 = 27/7 = 3.86
  - Insight: Average customer satisfaction is 3.86/5

**Median**
- Definition: Middle value when data is sorted
- Formula: Position = (n+1)/2
- Use Case: When dealing with skewed data or outliers
- Example: Customer order values (in INR)
  - Values: [500, 1200, 800, 2500, 900, 15000, 600]
  - Sorted: [500, 600, 800, 900, 1200, 2500, 15000]
  - Median = 900 (4th position out of 7)
  - Insight: 50% of orders are below 900 INR; median is less affected by the 15000 INR outlier than mean

**Mode**
- Definition: Most frequently occurring value
- Use Case: Categorical data or when finding the most common occurrence
- Example: Product categories purchased
  - Categories: [Electronics, Clothing, Electronics, Books, Electronics, Clothing, Clothing, Books]
  - Mode = Clothing (appears 3 times)
  - Insight: Most customers purchase clothing items

### 1.2 Measures of Spread/Dispersion

**Variance (σ²)**
- Definition: Average squared deviation from the mean
- Formula: σ² = Σ(x - μ)² / n
- Interpretation: Higher variance = more spread in data
- Example: Website daily traffic variation
  - Daily visits: [100, 150, 120, 110, 130]
  - Mean = 122
  - Deviations squared: [484, 784, 4, 144, 64]
  - Variance = 1480 / 5 = 296

**Standard Deviation (σ)**
- Definition: Square root of variance
- Formula: σ = √(σ²)
- Interpretation: Easier to understand than variance (same units as original data)
- Example (continued from above):
  - Standard Deviation = √296 ≈ 17.2 visits
  - Insight: Traffic typically varies by ±17.2 visits from the mean of 122

**Coefficient of Variation (CV)**
- Definition: Standard deviation as a percentage of the mean
- Formula: CV = (σ / μ) × 100%
- Use Case: Comparing variability across datasets with different units/scales
- Example: Comparing price variability across product categories
  - Electronics: μ = 5000, σ = 1200 → CV = 24%
  - Clothing: μ = 800, σ = 150 → CV = 18.75%
  - Insight: Electronics prices are more variable relative to their mean

### 1.3 Percentiles and Quartiles

**Quartiles**
- Q1 (25th percentile): 25% of data below this value
- Q2 (50th percentile): Median
- Q3 (75th percentile): 75% of data below this value
- IQR (Interquartile Range) = Q3 - Q1

Example: Customer spending analysis
- Order values sorted: [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
- Q1 = 500 (25th percentile)
- Q2 = 1100 (median)
- Q3 = 1700 (75th percentile)
- IQR = 1200
- Interpretation: Middle 50% of customers spend between 500-1700 INR

---

## 2. Probability Fundamentals

### 2.1 Basic Concepts

**Probability Definition**
- P(A) = Number of favorable outcomes / Total number of possible outcomes
- Range: 0 ≤ P(A) ≤ 1

Example: Customer conversion prediction
- Total website visitors: 1000
- Visitors who purchased: 150
- P(Purchase) = 150/1000 = 0.15 or 15%

### 2.2 Conditional Probability

**Definition**: Probability of event A occurring given that event B has occurred
- Formula: P(A|B) = P(A and B) / P(B)

Example: E-commerce customer behavior
- Total customers: 1000
- Customers who received email: 400
- Customers who purchased after email: 120
- P(Purchase | Received Email) = 120/400 = 0.30 or 30%
- Insight: Email recipients have a 30% conversion rate

### 2.3 Bayes' Theorem

**Formula**: P(A|B) = [P(B|A) × P(A)] / P(B)

**Real-world ML scenario**: Spam detection
- P(Spam | Contains "Free"): What's the probability an email is spam given it contains "Free"?
- Prior P(Spam) = 0.02 (2% of emails are spam)
- P(Contains "Free" | Spam) = 0.9 (90% of spam emails contain "Free")
- P(Contains "Free") = 0.1 (10% of all emails contain "Free")
- P(Spam | "Free") = (0.9 × 0.02) / 0.1 = 0.18 or 18%
- Insight: An email containing "Free" has 18% chance of being spam

---

## 3. Probability Distributions

### 3.1 Normal Distribution (Gaussian)

**Characteristics**:
- Bell-shaped, symmetric curve
- Defined by mean (μ) and standard deviation (σ)
- 68% of data within ±1σ, 95% within ±2σ, 99.7% within ±3σ

**When to use**: Many real-world phenomena (heights, test scores, measurement errors)

**Example**: Product delivery time analysis
- Mean delivery time = 5 days
- Standard deviation = 1 day
- Distribution: N(5, 1)
- 68% of deliveries: 4-6 days
- 95% of deliveries: 3-7 days
- Business decision: Set SLA at 7 days to cover 99.7% of deliveries

### 3.2 Binomial Distribution

**Characteristics**:
- Discrete distribution
- n independent trials, each with probability p of success
- Parameters: n (number of trials), p (probability of success)

**When to use**: Fixed number of independent yes/no outcomes

**Example**: Customer survey responses
- Survey 100 customers (n=100)
- Probability each customer rates product ≥4/5 = 0.7 (p=0.7)
- Expected number of positive reviews = np = 100 × 0.7 = 70
- Standard deviation = √(np(1-p)) = √(100 × 0.7 × 0.3) = √21 ≈ 4.58
- Interpretation: Expect ~70 ±4.58 positive reviews (likely 65-75)

### 3.3 Poisson Distribution

**Characteristics**:
- Discrete distribution
- Models count of events in fixed time/space interval
- Parameter: λ (lambda) = average rate

**When to use**: Counting rare events in fixed intervals

**Example**: Server error rate monitoring
- Average errors per hour = 2 (λ=2)
- P(exactly 3 errors in next hour) follows Poisson(2)
- This helps set alert thresholds: if errors > 5 in an hour, trigger alert

---

## 4. Inferential Statistics

### 4.1 Sampling and Sampling Distributions

**Population vs Sample**:
- Population: Entire group of interest
- Sample: Subset used for analysis

**Standard Error (SE)**:
- Formula: SE = σ / √n
- Interpretation: Variability of sample means

Example: E-commerce product ratings
- Population standard deviation = 1.2
- Sample size = 100 reviews
- SE = 1.2 / √100 = 1.2 / 10 = 0.12
- Interpretation: Average rating of different 100-review samples varies by ±0.12

### 4.2 Confidence Intervals

**Definition**: Range of values likely to contain population parameter

**Formula**: CI = Sample Mean ± (Z-score × Standard Error)
- 95% CI: Z-score = 1.96
- 99% CI: Z-score = 2.576

**Example**: Average customer order value
- Sample mean = 2500 INR
- Standard error = 150 INR
- 95% CI = 2500 ± (1.96 × 150) = 2500 ± 294
- Confidence Interval: [2206, 2794] INR
- Interpretation: We're 95% confident the true population mean lies between 2206-2794 INR

### 4.3 Hypothesis Testing

**Null Hypothesis (H₀)**: Status quo / No effect
**Alternative Hypothesis (H₁)**: What you're testing

**P-value**: Probability of observing data if H₀ is true
- If p-value < 0.05: Reject H₀ (statistically significant)
- If p-value ≥ 0.05: Fail to reject H₀ (not significant)

**Example**: A/B testing email campaign
- H₀: Email redesign has no effect on click-through rate
- H₁: Email redesign changes click-through rate
- Original CTR = 5%
- New design sample: 1000 emails, 65 clicks (6.5% CTR)
- Calculate p-value using proportion test
- If p-value = 0.02 (< 0.05): Reject H₀, new design is significantly better

#### Common Hypothesis Tests

**1. One-sample t-test**
- Test if sample mean differs from population mean
- Example: Is average customer satisfaction score (sample) significantly different from 4.0?

**2. Two-sample t-test**
- Compare means of two independent groups
- Example: Compare average order value between mobile and desktop users
- H₀: Both channels have same average order value
- If p < 0.05: Channels have significantly different order values

**3. Chi-square test**
- Test association between categorical variables
- Example: Is there association between product category and customer segment?
- H₀: Product choice is independent of customer segment

---

## 5. Correlation and Regression

### 5.1 Correlation Analysis

**Pearson Correlation Coefficient (r)**:
- Range: -1 to +1
- r = 1: Perfect positive relationship
- r = -1: Perfect negative relationship
- r = 0: No linear relationship

**Interpretation Guide**:
- |r| < 0.3: Weak correlation
- 0.3 ≤ |r| < 0.7: Moderate correlation
- |r| ≥ 0.7: Strong correlation

**Example**: E-commerce data analysis
- Correlation between product price and quantity sold: r = -0.65
- Interpretation: Moderate negative correlation—as price increases, sales volume decreases

### 5.2 Linear Regression

**Simple Linear Regression**: Y = β₀ + β₁X + ε
- Y: Dependent variable (outcome)
- X: Independent variable (predictor)
- β₀: Y-intercept
- β₁: Slope (change in Y per unit increase in X)
- ε: Error term

**Example**: Predicting customer spending
- Regression: Spending = 500 + 150 × (Years as customer)
- Interpretation:
  - New customers spend ~500 INR
  - Spending increases by 150 INR per year of loyalty
  - A 5-year customer expected spend: 500 + (150 × 5) = 1250 INR

**R² (Coefficient of Determination)**:
- Percentage of variance in Y explained by X
- R² = 0.72 means 72% of spending variation is explained by years as customer
- Remaining 28% due to other factors (preferences, income, etc.)

### 5.3 Multiple Regression

**Formula**: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε

**Example**: Predicting house prices
- Price = 1,000,000 + (5,000 × Area) + (50,000 × Bedrooms) - (10,000 × Distance_to_city)
- For a 2000 sq ft house, 4 bedrooms, 5km from city:
- Price = 1,000,000 + (5000×2000) + (50,000×4) - (10,000×5)
- Price = 1,000,000 + 10,000,000 + 200,000 - 50,000 = 11,150,000

---

## 6. Bayesian Statistics

### 6.1 Bayesian vs Frequentist Approach

| Aspect | Frequentist | Bayesian |
|--------|------------|----------|
| Probability | Long-run frequency | Degree of belief |
| Parameters | Fixed, unknown | Random variables |
| Prior beliefs | Not used | Explicitly incorporated |
| Decision-making | Based on p-values | Posterior distribution |

### 6.2 Posterior Distribution

**Bayes' Rule (Extended)**: P(θ|Data) = [P(Data|θ) × P(θ)] / P(Data)

- P(θ|Data): Posterior (updated belief after seeing data)
- P(Data|θ): Likelihood (data probability given parameter)
- P(θ): Prior (initial belief)
- P(Data): Evidence

**Example**: Click fraud detection
- Prior belief: 5% of clicks are fraudulent
- New data: 100 clicks analyzed, 8 appear suspicious
- Update your belief using Bayes' theorem
- Posterior: Updated probability that new clicks are fraudulent given the evidence

---

## 7. Practical Application Scenarios

### Scenario 1: Product Recommendation System Analysis

**Question**: Is recommendation algorithm A better than algorithm B?

**Statistical Approach**:
1. A/B test with 5000 users each
2. Measure: conversion rate on recommended products
3. Algorithm A: 12% conversion (600 conversions)
4. Algorithm B: 15% conversion (750 conversions)
5. Hypothesis test: Is 3% difference statistically significant?
6. Calculate p-value using chi-square test
7. If p < 0.05: Algorithm B is statistically better
8. Calculate confidence interval for effect size
9. Business decision: Rollout Algorithm B if effect size is practically meaningful

### Scenario 2: Inventory Forecasting

**Question**: How much stock should you maintain?

**Statistical Approach**:
1. Analyze historical demand: mean = 100 units/day, σ = 15 units
2. Assume normal distribution
3. Set service level = 95% (minimize stockouts)
4. Find Z-score for 95% = 1.645
5. Safety stock = Z × σ = 1.645 × 15 = 24.68 ≈ 25 units
6. Maintain minimum inventory = average demand + safety stock
7. Result: Keep minimum 125 units to achieve 95% service level

### Scenario 3: Customer Lifetime Value (CLV) Prediction

**Question**: Which customers are most valuable?

**Statistical Approach**:
1. Gather data: purchase frequency, average order value, customer tenure
2. Calculate correlation between features and CLV
3. Build multiple regression model
4. CLV = β₀ + β₁(frequency) + β₂(order_value) + β₃(tenure)
5. Segment customers by predicted CLV
6. Focus retention efforts on high-CLV customers
7. Validate model using R² and residual analysis

### Scenario 4: Quality Control - Manufacturing Defects

**Question**: Is defect rate within acceptable limits?

**Statistical Approach**:
1. Hypothesis: Defect rate ≤ 2% (acceptable)
2. Sample 200 units from production
3. Find 6 defects = 3% rate
4. Perform one-proportion z-test
5. If p-value < 0.05: Defect rate significantly > 2%, investigate production
6. Use control charts to monitor ongoing quality

---

## 8. Common Pitfalls and Misconceptions

| Pitfall | Explanation | Solution |
|---------|-------------|----------|
| P-value confusion | P-value ≠ probability H₀ is true | Interpret as: if H₀ true, P(data this extreme) = p-value |
| Correlation = Causation | Two variables correlating doesn't mean one causes the other | Consider confounding variables and causal mechanisms |
| Small sample sizes | Low statistical power, unreliable estimates | Increase sample size or use Bayesian priors |
| Ignoring data distribution | Assuming normality when data is skewed | Check Q-Q plots, histograms; use non-parametric tests |
| Multiple testing problem | Running many tests increases false positives | Apply Bonferroni correction or adjust significance level |
| Survivor bias | Only analyzing successes ignores failures | Include full dataset (successful + failed cases) |

---

## 9. Python Implementation References

### Quick Implementation Checklist

For your Django + ML projects:
1. **Exploratory Data Analysis (EDA)**
   ```
   - Use pandas for descriptive statistics (describe(), quantile())
   - Use matplotlib/seaborn for visualization
   - Check correlations with corr() and heatmaps
   ```

2. **Hypothesis Testing**
   ```
   - scipy.stats for t-tests, chi-square, z-tests
   - Calculate p-values and confidence intervals
   - Visualize distributions and results
   ```

3. **Regression Models**
   ```
   - scikit-learn for LinearRegression, preprocessing
   - Validate using cross-validation (train/test split)
   - Evaluate with R², RMSE, MAE metrics
   ```

4. **Bayesian Methods**
   ```
   - PyMC3 or arviz for posterior sampling
   - Visualize prior, likelihood, posterior distributions
   - Use for uncertainty quantification
   ```

---

## 10. Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Mean | μ = Σx / n |
| Variance | σ² = Σ(x-μ)² / n |
| Standard Deviation | σ = √σ² |
| Z-score | z = (x - μ) / σ |
| Standard Error | SE = σ / √n |
| Confidence Interval | CI = x̄ ± (Z × SE) |
| Correlation | r = Cov(X,Y) / (σₓ × σᵧ) |
| Linear Regression | ŷ = β₀ + β₁x |
| Bayes' Theorem | P(A\|B) = P(B\|A) × P(A) / P(B) |
| Chi-square | χ² = Σ(O-E)² / E |

---

## Study Tips for Data Science Context

1. **Understand the "Why"**: Don't memorize formulas—understand when and why to use each test
2. **Practice with Real Data**: Apply concepts to your e-commerce project data
3. **Visual Learning**: Create plots for distributions, relationships, and test results
4. **Incremental Learning**: Master one concept before moving to the next
5. **Connect to ML**: Recognize how statistics underlies machine learning (loss functions, regularization, uncertainty estimation)
6. **Document Examples**: Keep a personal library of problems and solutions from your projects

