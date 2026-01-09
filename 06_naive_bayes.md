# Naive Bayes - Complete Guide

## 1. Fundamentals of Naive Bayes

### 1.1 What is Naive Bayes?

**Definition:** Probabilistic classifier based on Bayes' theorem with conditional independence assumption.

**Core Equation (Bayes' Theorem):**
```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
```

Where:
- P(Class|Features): Posterior (what we want)
- P(Features|Class): Likelihood (probability of data given class)
- P(Class): Prior (class probability before seeing data)
- P(Features): Evidence (normalizing constant)

**Why "Naive"?**
- Assumes features conditionally independent given class
- Rarely true in practice
- Still works surprisingly well despite unrealistic assumption

---

### 1.2 Conditional Independence Assumption

**Assumption:**
```
P(x₁, x₂, ..., xₚ|Class) = P(x₁|Class) × P(x₂|Class) × ... × P(xₚ|Class)
```

**Meaning:**
- Given class, features independent
- Feature values don't influence each other (only class influences)

**Reality:**
- Usually violated (features correlated)
- But works anyway (empirically robust)

**Why Works Despite Violation:**
- Bias-variance trade-off
- Simpler model (fewer parameters) generalizes better
- Posterior probabilities more important than exact probabilities

---

### 1.3 Naive Bayes vs Other Classifiers

| Aspect | Naive Bayes | Logistic Regression | Decision Tree |
|--------|------------|-------------------|--------------|
| Type | Probabilistic | Discriminative | Non-parametric |
| Assumptions | Independence | Linear in log-odds | No assumptions |
| Parameters | Probabilities | Weights | Splits |
| Training Speed | Very fast | Fast | Medium |
| Prediction Speed | Very fast | Very fast | Fast |
| Interpretability | High | High | High |
| Multiclass | Native | Softmax | Native |
| Uncertainty | Probabilities | Probabilities | No |

---

## 2. Bayes' Theorem Deep Dive

### 2.1 Derivation

**Joint Probability:**
```
P(A, B) = P(A|B) × P(B) = P(B|A) × P(A)
```

**Rearrange:**
```
P(A|B) = P(B|A) × P(A) / P(B)
```

**In Classification:**
```
P(Class|X) = P(X|Class) × P(Class) / P(X)
```

---

### 2.2 Components Breakdown

**Prior: P(Class)**
- Probability of class before seeing features
- Estimated from training data: count(Class) / total
- Example: P(Spam) = 300 / 1000 = 0.3

**Likelihood: P(X|Class)**
- Probability of observing features given class
- Depends on feature distributions (Gaussian, multinomial, etc.)

**Evidence: P(X)**
- Total probability of observing features
- Constant for all classes (doesn't affect comparison)
- Often omitted: Compare unnormalized posteriors

**Posterior: P(Class|X)**
- Final probability: probability class given features
- What we want for predictions

---

### 2.3 Decision Rule

**For Binary Classification:**
```
If P(Class=1|X) > P(Class=0|X): Predict Class 1
Else: Predict Class 0
```

**Equivalently (omit evidence):**
```
If P(X|Class=1) × P(Class=1) > P(X|Class=0) × P(Class=0): Predict 1
Else: Predict 0
```

**Multiclass:**
```
Predict class = argmax_c [P(X|Class=c) × P(Class=c)]
```

---

## 3. Gaussian Naive Bayes (Continuous Features)

### 3.1 Assumption

**Features Normally Distributed:**
```
P(xⱼ|Class=c) ~ N(μ_jc, σ_jc²)
```

Where:
- μ_jc: Mean of feature j in class c
- σ_jc²: Variance of feature j in class c

---

### 3.2 Probability Density Function

**Gaussian PDF:**
```
P(xⱼ|Class=c) = (1 / √(2π σ_jc²)) × exp(-(xⱼ - μ_jc)² / (2 σ_jc²))
```

**Interpretation:**
- Bell curve centered at μ_jc
- Width determined by σ_jc
- Higher at center, lower at tails

---

### 3.3 Likelihood Estimation

**From Training Data:**

For each class c and feature j:
```
μ_jc = mean(X_j[y=c])  # Mean of feature j in class c
σ_jc = std(X_j[y=c])   # Std dev of feature j in class c
```

**Example:**
```
Data: Age=[25, 30, 35, 40], Spam=[0, 0, 1, 1]

For Spam=0 (age):
  μ_0 = (25 + 30) / 2 = 27.5
  σ_0 = std([25, 30]) ≈ 3.5

For Spam=1 (age):
  μ_1 = (35 + 40) / 2 = 37.5
  σ_1 = std([35, 40]) ≈ 3.5
```

---

### 3.4 Prediction Example

**Given:** New sample with Age=32, Words=1500

**Calculate Posteriors:**
```
P(Spam=0|Age, Words) ∝ P(Age|Spam=0) × P(Words|Spam=0) × P(Spam=0)
                      = N(32; 27.5, 3.5²) × P(Words|Spam=0) × 0.5

P(Spam=1|Age, Words) ∝ P(Age|Spam=1) × P(Words|Spam=1) × P(Spam=1)
                      = N(32; 37.5, 3.5²) × P(Words|Spam=1) × 0.5
```

**Prediction:** Choose class with higher posterior

---

## 4. Multinomial Naive Bayes (Discrete Features)

### 4.1 Use Case

**Ideal For:**
- Count data (word frequencies, click counts)
- Text classification (bag-of-words)
- Document categorization

---

### 4.2 Multinomial Distribution

**Assumption:**
```
P(xⱼ|Class=c) = (nⱼc + α) / (Nc + α×V)
```

Where:
- nⱼc: Count of feature j in class c
- Nc: Total count of features in class c
- V: Vocabulary size (number of unique features)
- α: Smoothing parameter (default α=1, Laplace smoothing)

---

### 4.3 Text Classification Example

**Data:**
```
Email 1: "Free money win lottery" → Spam
Email 2: "Free trial review new product" → Spam
Email 3: "Meeting schedule tomorrow" → Not Spam
Email 4: "Project deadline monday" → Not Spam
```

**Vocabulary:** {Free, Money, Win, Lottery, Trial, Review, New, Product, Meeting, Schedule, Tomorrow, Project, Deadline, Monday}

**Count Matrices:**

Spam class counts:
```
Free: 2, Money: 1, Win: 1, Lottery: 1, Trial: 1, Review: 1, New: 1, Product: 1, others: 0
Total: 9
```

Not Spam counts:
```
Meeting: 1, Schedule: 1, Tomorrow: 1, Project: 1, Deadline: 1, Monday: 1, others: 0
Total: 6
```

**Probability Estimation (with Laplace smoothing α=1):**

For Spam:
```
P(Free|Spam) = (2 + 1) / (9 + 14) = 3/23 ≈ 0.13
P(Money|Spam) = (1 + 1) / (9 + 14) = 2/23 ≈ 0.09
P(Meeting|Spam) = (0 + 1) / (9 + 14) = 1/23 ≈ 0.04
```

For Not Spam:
```
P(Free|Not Spam) = (0 + 1) / (6 + 14) = 1/20 = 0.05
P(Meeting|Not Spam) = (1 + 1) / (6 + 14) = 2/20 = 0.10
```

**Prediction (Email: "Free money"):**
```
P(Spam|"Free money") ∝ P(Free|Spam) × P(Money|Spam) × P(Spam)
                     = 0.13 × 0.09 × 0.5 ≈ 0.0059

P(Not Spam|"Free money") ∝ P(Free|NotSpam) × P(Money|NotSpam) × P(NotSpam)
                         = 0.05 × 0.05 × 0.5 ≈ 0.0013

Predict: Spam (higher posterior)
```

---

### 4.4 Bag-of-Words Representation

**Idea:** Document = unordered collection of word counts

**Process:**
1. Tokenize: "I love machine learning" → ["I", "love", "machine", "learning"]
2. Count: {I: 1, love: 1, machine: 1, learning: 1}
3. Remove stopwords: {love: 1, machine: 1, learning: 1}
4. Vector: [1, 1, 1] (if vocabulary ordered)

**Advantages:**
- Simple, interpretable
- Fast

**Disadvantages:**
- Loses word order
- Long documents overrepresented

**Variants:**
- TF-IDF: Weight by frequency-inverse document frequency
- N-grams: Capture word sequences

---

## 5. Bernoulli Naive Bayes (Binary Features)

### 5.1 Use Case

**Ideal For:**
- Binary features (present/absent)
- Document presence (word appears yes/no)
- Click prediction (clicked yes/no)

---

### 5.2 Bernoulli Distribution

**Assumption:**
```
P(xⱼ=1|Class=c) = pⱼc
P(xⱼ=0|Class=c) = 1 - pⱼc
```

Where pⱼc: Probability of feature j=1 in class c

**Estimation:**
```
pⱼc = (count(xⱼ=1 in class c) + α) / (count(samples in class c) + 2α)
```

(α=1 Laplace smoothing)

---

### 5.3 Likelihood

**For Sample x:**
```
P(x|Class=c) = ∏ pⱼc^(xⱼ) × (1-pⱼc)^(1-xⱼ)
             = ∏ P(xⱼ|Class=c)
```

**Interpretation:**
- If xⱼ=1: Multiply by pⱼc (how likely feature j in class c)
- If xⱼ=0: Multiply by (1-pⱼc) (how likely feature j absent in class c)

---

### 5.4 Bernoulli vs Multinomial

| Aspect | Bernoulli | Multinomial |
|--------|-----------|------------|
| Data | Binary (0/1) | Counts (0,1,2,3,...) |
| Meaning | Feature presence | Feature frequency |
| Use | Word exists | Word count |
| When Better | Few features | Many features/counts |
| Text | Binary BoW | Count BoW |

---

## 6. Laplace Smoothing (Handling Zero Probabilities)

### 6.1 Problem: Zero Probability

**Example:**
```
Training data: Spam emails: "money", "free", "win"
No spam emails: no "lottery"

For Spam: P(lottery|Spam) = 1 / 10 (saw 1 lottery, 10 total words)
For Not Spam: P(lottery|Not Spam) = 0 / 5 (never saw, 5 total words)
```

**Problem:** P(lottery|Not Spam) = 0

**Consequence:**
- If test email contains "lottery" and class hasn't seen it
- Probability becomes 0
- Product of all features: 0
- Can't compare classes fairly

---

### 6.2 Laplace Smoothing Solution

**Formula (Multinomial):**
```
P(xⱼ|Class=c) = (count(xⱼ in class c) + α) / (total_count_class_c + α × V)
```

Where:
- α: Smoothing parameter (typically 1)
- V: Number of unique features

**Effect:**
- Adds α to numerator (pseudocount)
- Adds α×V to denominator (total adjustment)
- No zero probabilities
- All features at least get small probability

**Example (α=1, Multinomial):**
```
V = 3 (vocabulary: {money, free, lottery})

Before Laplace:
P(lottery|Not Spam) = 0 / 5 = 0

After Laplace (α=1):
P(lottery|Not Spam) = (0 + 1) / (5 + 1×3) = 1/8 ≈ 0.125
```

---

### 6.3 Effect on Predictions

**Without Smoothing:**
- Unseen features → Zero probability
- Often incorrect (feature seen elsewhere → should affect prediction)

**With Smoothing:**
- Unseen features → Small probability
- Fairer comparison

**Trade-off:**
- High α: More smoothing, less discriminative
- Low α (≈0.1-1): Minimal smoothing, better balance
- Tune via cross-validation

---

## 7. Advantages of Naive Bayes

### 7.1 Simplicity & Interpretability

**Easy to Understand:**
- Probabilistic interpretation
- Feature probabilities interpretable
- Follow logic: P(feature|class) important

**Transparent:**
- Can show which features drive predictions
- P(class|features) directly interpretable as probability

---

### 7.2 Fast Training & Prediction

**Training:** O(n×d) linear in samples and features
- Count occurrences
- Compute probabilities
- No optimization needed

**Prediction:** O(d) linear in features
- Multiply feature probabilities
- Fast real-time predictions

---

### 7.3 Works with High-Dimensional Data

**Text Representation:**
- Thousands of words (features)
- Still fast (O(k) where k = non-zero features)
- Natural for sparse data

---

### 7.4 Probabilistic Framework

**Probability Estimates:**
- Direct P(class|data) outputs
- Can threshold flexibly
- Uncertainty quantification

---

### 7.5 Handles Both Discrete & Continuous

**Variants:**
- Gaussian (continuous)
- Multinomial (discrete counts)
- Bernoulli (binary)
- Mix in hybrid models

---

## 8. Disadvantages of Naive Bayes

### 8.1 Independence Assumption Violation

**Problem:** Assumes features independent given class
- Usually false in practice
- Correlated features violate assumption

**Example:**
```
Text: "machine learning"
Words "machine" and "learning" not independent
Together → Likely ML topic
Separately → Individual interpretation

Naive Bayes ignores correlation
```

**Impact:**
- Ignores feature interactions
- Sometimes underfits
- But empirically robust (works anyway)

---

### 8.2 Poor with Small Data

**Problem:**
- Few samples → Unreliable probability estimates
- Smoothing helps but not enough
- Prior dominates (little evidence to update)

**Example:**
```
10 samples total, only 1 spam
P(Spam) = 0.1
Even with evidence, posterior heavily influenced by tiny prior
```

**Solution:**
- More data
- Strong priors from domain knowledge

---

### 8.3 Categorical Feature Encoding Issues

**Problem:**
- High-cardinality features cause sparsity
- Each category sparse in training → Estimated probabilities unreliable

**Example:**
```
Feature: User ID (1000 unique values)
Each user appears once in training
P(User_ID=i|Spam) = 1/5 or 0/5 (unreliable)
```

**Solution:**
- Feature engineering (group categories)
- Reduce cardinality before training

---

### 8.4 Zero-Frequency Problem (Without Smoothing)

**Problem:** Unseen feature-class pairs have zero probability
- Product becomes zero
- Can't recover

**Solution:** Laplace smoothing (adds small pseudocounts)

---

### 8.5 Ignores Feature Interactions

**Problem:**
```
Features: x₁, x₂ highly correlated given class
Naive Bayes: Treats as independent
Result: Double-counts information
```

**Effect:**
- Overconfident predictions
- Poor probability calibration

**When Matters:**
- Strong feature interactions
- Need accurate probabilities (not just classification)

---

### 8.6 Imbalanced Data

**Problem:**
- Prior P(Class) biased toward majority
- Even with evidence, hard to flip prediction

**Example:**
```
P(Spam) = 0.01 (1% spam)
Need strong evidence from features to overcome low prior
```

**Solution:**
- Adjust class weights
- Resampling
- Threshold tuning

---

## 9. Text Classification with Naive Bayes

### 9.1 Pipeline

**1. Text Preprocessing:**
```python
# Tokenize, lowercase, remove punctuation
text = "I Love Machine Learning!"
tokens = text.lower().split()  # ['i', 'love', 'machine', 'learning!']

# Remove punctuation
tokens = ['i', 'love', 'machine', 'learning']

# Remove stopwords
stopwords = {'i', 'a', 'the', 'and', ...}
tokens = [t for t in tokens if t not in stopwords]  # ['love', 'machine', 'learning']
```

**2. Vectorization (Bag-of-Words):**
```python
# TF-IDF or Count vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)  # Sparse matrix
```

**3. Train Naive Bayes:**
```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
```

**4. Predict:**
```python
y_pred = nb.predict(X_test)
```

---

### 9.2 Common Preprocessing Steps

**Tokenization:** "hello world" → ["hello", "world"]

**Lowercasing:** "Hello" → "hello"

**Stopword Removal:** Remove common words (a, the, is) → Focus on meaningful

**Stemming:** "running", "runs" → "run"

**Lemmatization:** "running" → "run" (more sophisticated)

**Vectorization:** Words → Numbers (counts or TF-IDF)

---

### 9.3 Text Features

**Unigrams:** Individual words ["machine", "learning"]

**Bigrams:** Two-word sequences ["machine learning", "deep learning"]

**N-grams:** Captures some word order

**TF-IDF:** Downweights common words, emphasizes discriminative

---

## 10. Gaussian Naive Bayes Implementation

### 10.1 When to Use

- Continuous numeric features
- Features approximately normally distributed
- No time constraint

---

### 10.2 Probability Estimation

**From data:**
```
For each class c and feature j:
  μ_jc = mean(feature j | class c)
  σ_jc = std(feature j | class c)
```

**Use Gaussian PDF** for likelihood

---

### 10.3 Advantages

- Works well with continuous data
- Fast training & prediction
- No parameter tuning needed

---

### 10.4 Disadvantages

- Assumes normality (violations don't hurt much)
- Poor with outliers
- Equal variance assumption (can relax)

---

## 11. Handling Continuous Features in Naive Bayes

### 11.1 Discretization

**Convert continuous to categorical:**
```python
# Binning: Divide into equal-width or equal-frequency bins
age_bins = [0, 18, 35, 50, 65, 100]
age_binned = pd.cut(age, bins=age_bins, labels=['child', 'young', 'middle', 'mature', 'senior'])
```

**Then use Bernoulli or Multinomial NB**

**Advantage:** Captures non-linear relationships

**Disadvantage:** Information loss (boundaries arbitrary)

---

### 11.2 Kernel Estimation

**Use kernel density estimation for PDF:**

Instead of Gaussian PDF, use KDE

More flexible but slower

---

## 12. Comparison with Other Classifiers

| Aspect | Naive Bayes | Logistic Regression | Decision Tree |
|--------|------------|-------------------|--------------|
| Training Time | Very fast | Fast | Medium |
| Prediction Time | Very fast | Very fast | Fast |
| Interpretability | High | High | High |
| Probability | Yes | Yes | No |
| Non-linear | Limited | No | Yes |
| Assumptions | Independence | Linearity | None |
| High-dim | Good | Moderate | Poor |
| Small Data | Good (priors) | Moderate | Poor |

---

## 13. Implementation in sklearn

### 13.1 Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Create and train
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"CV Score: {cross_val_score(gnb, X_train, y_train, cv=5).mean()}")
print(f"\n{classification_report(y_test, y_pred)}")

# Feature means and variances (learned parameters)
print(f"Means: {gnb.theta_}")
print(f"Variances: {gnb.var_}")
```

### 13.2 Multinomial Naive Bayes (Text)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Train Naive Bayes
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_tfidf, y_train)

# Predict
y_pred = mnb.predict(X_test_tfidf)
y_pred_proba = mnb.predict_proba(X_test_tfidf)

# Feature log probabilities (learned)
print(f"Log probabilities shape: {mnb.feature_log_prob_.shape}")
```

### 13.3 Bernoulli Naive Bayes

```python
from sklearn.naive_bayes import BernoulliNB

# Create and train
bnb = BernoulliNB(alpha=1.0, binarize=0.5)  # binarize: threshold for 1
bnb.fit(X_train_binary, y_train)

# Predict
y_pred = bnb.predict(X_test_binary)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### 13.4 Hyperparameter: Smoothing (Alpha)

```python
from sklearn.model_selection import GridSearchCV

# Grid search for best alpha
alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
param_grid = {'alpha': alphas}

mnb = MultinomialNB()
grid = GridSearchCV(mnb, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_tfidf, y_train)

print(f"Best alpha: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_}")
print(f"Test accuracy: {grid.best_estimator_.score(X_test_tfidf, y_test)}")
```

---

## 14. Common Issues & Solutions

### 14.1 Poor Accuracy

**Potential Causes:**
- Independence assumption severely violated
- Wrong variant (Gaussian for counts?)
- Poor preprocessing (text)
- Class imbalance

**Solutions:**
```python
# 1. Try different variant
# Gaussian, Multinomial, Bernoulli

# 2. Better preprocessing
# Remove stopwords, stemming, lemmatization

# 3. Class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
# Adjust sample weights before fitting (sklearn doesn't support directly)

# 4. Feature engineering
# Create interaction features, domain knowledge
```

---

### 14.2 Probability Calibration Issues

**Problem:** Predicted probabilities poorly calibrated
- P(class) = 0.8 but actual frequency ~0.5

**Cause:** Independence assumption violation

**Solutions:**
```python
# 1. Calibrate predictions
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(mnb, cv=5)
calibrated_nb.fit(X_train_tfidf, y_train)
y_pred_proba_calibrated = calibrated_nb.predict_proba(X_test_tfidf)

# 2. Threshold tuning
y_pred_custom = (y_pred_proba[:, 1] >= 0.5).astype(int)
```

---

### 14.3 High-Cardinality Features

**Problem:** Too many unique categories → Sparsity

**Solutions:**
```python
# 1. Feature grouping
# Combine rare categories into "Other"

# 2. Feature selection
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=100)
X_selected = selector.fit_transform(X_train, y_train)

# 3. Dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_train)
```

---

## 15. Practice Problems

1. **Bayes' Theorem:** Derive Naive Bayes from Bayes' theorem. Why "naive"?

2. **Independence Assumption:** When violated, what happens? Still works?

3. **Laplace Smoothing:** Why needed? What if α too high?

4. **Gaussian vs Multinomial:** When each? Examples?

5. **Text Classification:** Pipeline from raw text to predictions?

6. **Probability Calibration:** Why poor without smoothing? How fix?

7. **Zero Frequency:** Problem and solution?

8. **Feature Interactions:** Naive Bayes can't capture. What model can?

---

## 16. Key Takeaways

1. **Bayes' theorem:** P(Class|Features) ∝ P(Features|Class) × P(Class)
2. **Naive assumption:** Features independent given class (simplifies computation)
3. **Variants:** Gaussian (continuous), Multinomial (counts), Bernoulli (binary)
4. **Laplace smoothing:** Add pseudocounts to avoid zero probabilities
5. **Advantages:** Fast, simple, good with high-d and small data
6. **Disadvantages:** Independence assumption, ignores interactions, poor calibration
7. **Text classification:** Excellent for NLP tasks
8. **Implementation:** sklearn provides GaussianNB, MultinomialNB, BernoulliNB
9. **Tuning:** Mainly alpha (smoothing), preprocessing important for text
10. **When to use:** Fast baseline, text, high-dimensional, small data, probabilistic predictions

---

**Next Topic:** Support Vector Machines (say "next" to continue)