# Feature Engineering - Complete Guide

## 1. Fundamentals of Feature Engineering

### 1.1 What is Feature Engineering?

**Definition:** Process of creating, transforming, selecting features to improve model performance.

**Core Idea:**
```
"Garbage in, garbage out" - Quality features → Quality predictions
```

**Components:**
1. **Feature Creation:** Generate new features from raw data
2. **Feature Transformation:** Transform existing features
3. **Feature Selection:** Choose important features
4. **Feature Scaling:** Normalize/standardize features

---

### 1.2 Why Feature Engineering Matters?

**Impact:** Often 80% of model performance comes from features

**Examples:**
```
Raw data: Year, Month, Day, Hour, Minute
Feature engineering: Day of week, Hour of day, Is_holiday, Days_since_event

Raw text: "The quick brown fox"
Feature engineering: Word count, Sentiment, TF-IDF, Word embeddings

Raw images: Pixel values
Feature engineering: Edge detection, Color histograms, Convolution filters
```

**Trade-off:**
- Good features: Simple model outperforms complex model on bad features
- Bad features: Complex model struggles

**Time Investment:**
- Data collection: 10%
- Data cleaning: 20%
- Feature engineering: 40%
- Model building: 20%
- Tuning: 10%

---

### 1.3 Feature Engineering Workflow

```
1. Understand the problem
   └─ What predicts target?
   
2. Exploratory Data Analysis (EDA)
   └─ Data types, distributions, missing values, outliers
   
3. Create features
   └─ Domain knowledge, interactions, transformations
   
4. Select features
   └─ Remove low-variance, highly correlated, irrelevant
   
5. Scale features
   └─ Normalize/standardize for certain algorithms
   
6. Model & Iterate
   └─ Try models, evaluate, refine features
```

---

## 2. Feature Creation

### 2.1 Mathematical Transformations

**Log Transformation:**
```python
import numpy as np

# For right-skewed data
x_log = np.log(x + 1)  # +1 to handle zeros
```

**When to use:**
- Right-skewed distributions
- Large range (0-1000000)
- Linear relationships in log space

**Example:**
```
Income: [1000, 2000, 5000, 1000000]
log(Income): [6.9, 7.6, 8.5, 13.8]
More symmetric distribution
```

---

**Power Transformation:**
```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
```

**Box-Cox Transformation:**
```python
from scipy.stats import boxcox

X_transformed, lambda_ = boxcox(X)
```

---

**Polynomial Features:**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
# Creates x, y, x^2, xy, y^2
```

**When to use:**
- Non-linear relationships
- Polynomial regression
- Tree models don't need (they create implicitly)

---

### 2.2 Domain-Specific Features

**Date/Time Features:**
```python
import pandas as pd

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday
df['quarter'] = df['date'].dt.quarter
df['is_quarter_end'] = df['date'].dt.is_quarter_end
df['is_holiday'] = df['date'].dt.date.isin(holiday_dates)
df['days_since'] = (df['date'] - df['date'].min()).dt.days
```

**When to use:**
- Time series forecasting
- Event detection (holidays, weekends)
- Seasonal patterns

---

**Geographic Features:**
```python
# Latitude, Longitude → Distance to center
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # miles
    return c * r

df['distance_to_center'] = df.apply(
    lambda row: haversine(row['lon'], row['lat'], center_lon, center_lat),
    axis=1
)
```

---

**Text Features:**
```python
import re

# Text length
df['text_length'] = df['text'].str.len()

# Word count
df['word_count'] = df['text'].str.split().str.len()

# Average word length
df['avg_word_length'] = df['text_length'] / df['word_count']

# Number of uppercase
df['n_uppercase'] = df['text'].str.count(r'[A-Z]')

# Punctuation count
df['n_punctuation'] = df['text'].str.count(r'[!?.]')

# Sentiment score
from textblob import TextBlob
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
```

---

### 2.3 Interaction Features

**Multiplication:**
```python
df['age_income'] = df['age'] * df['income']
```

**Division (Ratios):**
```python
df['debt_to_income'] = df['debt'] / df['income']
df['ctr'] = df['clicks'] / df['impressions']
```

**When to use:**
- Features have multiplicative relationship
- Create ratios that are meaningful
- For linear models (trees learn interactions)

**Example:**
```
Features: age, income
Interaction: age * income
Interpretations: Financial strength (older + richer = stronger)
```

---

**Binning Interactions:**
```python
df['age_income_group'] = (
    pd.cut(df['age'], bins=3, labels=['young', 'middle', 'old']) + 
    pd.cut(df['income'], bins=3, labels=['low', 'med', 'high'])
).astype(str)
# Creates groups like "young_low", "old_high"
```

---

### 2.4 Aggregation Features (for time series, grouped data)

**Rolling Statistics:**
```python
df['price_ma_7'] = df['price'].rolling(window=7).mean()  # 7-day moving average
df['price_std_30'] = df['price'].rolling(window=30).std()  # 30-day std dev
df['volume_sum_5'] = df['volume'].rolling(window=5).sum()  # 5-day cumulative
```

**Lag Features (for time series):**
```python
df['price_lag_1'] = df['price'].shift(1)  # Previous day price
df['price_lag_7'] = df['price'].shift(7)  # Week ago price
df['price_change'] = df['price'] - df['price_lag_1']  # Daily change
```

**GroupBy Aggregation:**
```python
df['user_avg_purchase'] = df.groupby('user_id')['purchase'].transform('mean')
df['user_std_purchase'] = df.groupby('user_id')['purchase'].transform('std')
df['user_purchase_count'] = df.groupby('user_id').size()
```

---

## 3. Feature Transformation

### 3.1 Scaling & Normalization

**StandardScaler (Z-score normalization):**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# x_scaled = (x - mean) / std
```

**When to use:**
- Gradient descent algorithms (SVM, Logistic Regression, Neural Networks)
- Features on different scales
- Features normally distributed

---

**MinMaxScaler (0-1 normalization):**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)

# x_scaled = (x - min) / (max - min)
```

**When to use:**
- Need bounded range [0, 1]
- Neural networks, image data
- Preserve zero values

---

**RobustScaler (percentile-based):**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)

# x_scaled = (x - median) / IQR
# Robust to outliers
```

**When to use:**
- Data with outliers
- Don't want to rescale entire range

---

### 3.2 Encoding Categorical Features

**One-Hot Encoding (for nominal categories):**
```python
df_encoded = pd.get_dummies(df, columns=['color'])
# color = 'red' → color_red=1, color_blue=0, color_green=0
```

**When to use:**
- Nominal (no order): color, brand, country
- Few categories (< 10)
- Algorithms need numeric (Linear, SVM, NN)

**Pitfall: Dummy Variable Trap**
```python
# WRONG: Include all k categories
df = pd.get_dummies(df, columns=['color'], drop_first=False)
# Red, Blue, Green (3 dummies for 3 categories)
# Perfect multicollinearity!

# CORRECT: Drop one category
df = pd.get_dummies(df, columns=['color'], drop_first=True)
# Red, Blue (2 dummies for 3 categories, Green implicit)
```

---

**Label Encoding (for ordinal categories):**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
# 'high school' → 0, 'bachelor' → 1, 'master' → 2
```

**When to use:**
- Ordinal (has order): education, rating, skill level
- Tree models (understand order implicitly)

---

**Target Encoding (for high-cardinality):**
```python
# For each category: mean target value
df['country_target_encoded'] = df.groupby('country')['target'].transform('mean')
```

**When to use:**
- High cardinality (many categories)
- Strong relationship category → target

**Risk:** Overfitting (encode on training, use same for test)

---

**Frequency Encoding:**
```python
df['color_freq'] = df['color'].map(df['color'].value_counts())
# Red appears 100x → encode as 100
```

**When to use:**
- Frequency matters
- High cardinality

---

### 3.3 Handling Missing Values

**Mean/Median Imputation:**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median'
X_imputed = imputer.fit_transform(X_train)
```

**Pros:** Simple, fast
**Cons:** Ignores data distribution, can create biased estimates

---

**Forward/Backward Fill (time series):**
```python
df['value'] = df['value'].fillna(method='ffill')  # Forward fill
df['value'] = df['value'].fillna(method='bfill')  # Backward fill
```

**When to use:** Time series, sequential data

---

**Interpolation:**
```python
df['value'] = df['value'].interpolate(method='linear')  # or 'polynomial'
```

**When to use:** Smooth time series

---

**KNN Imputation:**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X_train)
```

**Pros:** Uses similar samples, preserves relationships
**Cons:** Slow for large datasets

---

**Missing as Feature:**
```python
df['value_is_missing'] = df['value'].isna().astype(int)
df['value'] = df['value'].fillna(df['value'].mean())
```

**When to use:** Missingness itself informative

---

## 4. Feature Selection

### 4.1 Why Feature Selection?

**Benefits:**
- Reduces overfitting (fewer features)
- Improves interpretability
- Reduces training time
- Reduces memory usage
- Avoids curse of dimensionality

---

### 4.2 Variance Filtering

**Remove Low Variance Features:**
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)  # Remove if variance < 0.01
X_selected = selector.fit_transform(X)
```

**Interpretation:**
- Low variance → Doesn't vary much → Doesn't help predict
- Example: Feature = [1,1,1,1,1,2,1] (almost constant)

---

### 4.3 Correlation-Based Selection

**Remove Highly Correlated Features:**
```python
import numpy as np

corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df = df.drop(columns=to_drop)
```

**VIF (Variance Inflation Factor):**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# VIF > 10: High multicollinearity, drop feature
```

---

### 4.4 Univariate Statistical Tests

**SelectKBest (Select k best features):**
```python
from sklearn.feature_selection import SelectKBest, f_classif  # or f_regression

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
```

**Scoring Functions:**
- Classification: f_classif, chi2, mutual_info_classif
- Regression: f_regression, mutual_info_regression

**Pros:** Fast, interpretable
**Cons:** Ignores feature interactions

---

### 4.5 Model-Based Feature Selection

**Feature Importance (Tree-based):**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Select top k
top_features = importance.head(10)['Feature'].tolist()
X_selected = X[top_features]
```

---

**Permutation Importance:**
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)
```

---

### 4.6 Recursive Feature Elimination (RFE)

**Idea:** Recursively remove features with lowest importance

```python
from sklearn.feature_selection import RFE

rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10, step=1)
X_selected = rfe.fit_transform(X_train, y_train)

# Selected features
selected_features = X_train.columns[rfe.support_].tolist()
```

**Process:**
```
1. Train model on all features
2. Remove feature with lowest importance
3. Repeat until k features remain
```

---

### 4.7 Regularization (Built-in Feature Selection)

**Lasso (L1) automatically selects features:**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', solver='liblinear')
lr.fit(X_train, y_train)

# Features with non-zero coefficients selected
selected_features = X_train.columns[lr.coef_[0] != 0].tolist()
```

**Elastic Net (L1 + L2):**
```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(l1_ratio=0.5)  # Balance between L1 and L2
model.fit(X_train, y_train)
```

---

### 4.8 Feature Selection Strategy

```
1. Start simple
   └─ Domain knowledge, remove obvious bad features

2. Remove low variance
   └─ Constant/near-constant features

3. Remove highly correlated
   └─ Keep one, drop others

4. Statistical tests
   └─ SelectKBest for quick filtering

5. Model-based
   └─ Importance from tree model

6. Iterative
   └─ Remove one, evaluate, repeat

7. Regularization
   └─ Lasso for automatic selection
```

---

## 5. Handling Special Cases

### 5.1 Outliers

**Detection:**
```python
import numpy as np

# Z-score > 3
z_scores = np.abs((X - X.mean()) / X.std())
outliers = (z_scores > 3).any(axis=1)

# IQR method
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = (X < Q1 - 1.5*IQR) | (X > Q3 + 1.5*IQR)
```

**Treatment:**
```python
# 1. Remove
X_clean = X[~outliers]

# 2. Cap (Winsorizing)
X_capped = X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95))

# 3. Transform
X_log = np.log(X + 1)  # Makes extreme values less extreme

# 4. Keep as feature
df['is_outlier'] = outliers.astype(int)
```

---

### 5.2 Class Imbalance (Feature Perspective)

**Create minority-specific features:**
```python
# Example: For fraud detection
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['is_large_transaction'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
# Frauds often weekend, large amounts
```

---

### 5.3 Dimensionality Reduction

**PCA (Principal Component Analysis):**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_train)

print(f"Original dims: {X_train.shape[1]}")
print(f"Reduced dims: {X_pca.shape[1]}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
```

**When to use:**
- Too many features (curse of dimensionality)
- Features correlated
- Visualization

---

**UMAP/t-SNE (Non-linear):**
```python
from umap import UMAP

umap = UMAP(n_components=2)
X_umap = umap.fit_transform(X_train)
```

**Better for:** Preserving local structure, visualization

---

## 6. Feature Engineering for Different Data Types

### 6.1 Images

**Feature Extraction:**
```python
# Edge detection
from skimage import filters
edges = filters.sobel(image)

# Color histograms
from skimage.color import rgb2hsv
hsv = rgb2hsv(image)
h_hist = np.histogram(hsv[:,:,0], bins=180)

# CNN features (transfer learning)
from keras.applications import VGG16
model = VGG16(weights='imagenet', include_top=False)
features = model.predict(image_batch)
```

---

### 6.2 Text

**Bag-of-Words:**
```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(max_features=1000, stop_words='english')
X_bow = vec.fit_transform(texts)
```

**TF-IDF:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vec.fit_transform(texts)
```

**Word Embeddings:**
```python
from gensim.models import Word2Vec

model = Word2Vec(sentences, vector_size=100, window=5)
# Each word → 100-dim vector
```

---

### 6.3 Time Series

**Lagged Features:**
```python
df['price_lag_1'] = df['price'].shift(1)
df['price_lag_7'] = df['price'].shift(7)
```

**Rolling Statistics:**
```python
df['price_ma7'] = df['price'].rolling(7).mean()
df['price_vol7'] = df['price'].rolling(7).std()
```

**Fourier Features (for seasonality):**
```python
df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
```

---

## 7. Feature Engineering Best Practices

### 7.1 Do's

✅ **Understand the domain** - Domain knowledge beats pure statistics
✅ **Check correlations** - Know relationships between features
✅ **Visualize distributions** - Understand data before transforming
✅ **Create interpretable features** - Not just statistical, make sense
✅ **Validate on test set** - Ensure improvements generalize
✅ **Document features** - Remember why you created each feature
✅ **Iterate** - Feature engineering is experimental

---

### 7.2 Don'ts

❌ **Don't leak data** - Never use test set statistics in preprocessing
❌ **Don't create too many features** - Curse of dimensionality, overfitting
❌ **Don't forget scaling** - Some algorithms sensitive to scale
❌ **Don't ignore business context** - Features should make business sense
❌ **Don't engineer blindly** - Don't create features just because, test value
❌ **Don't engineer after splitting** - Must fit on train only

---

## 8. Feature Engineering Workflow Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
df = pd.read_csv('data.csv')

# 1. Exploratory Analysis
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 2. Handle Missing Values
df['age'].fillna(df['age'].mean(), inplace=True)
df = df.dropna()

# 3. Create Features
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 60, 100])
df['salary_per_age'] = df['salary'] / df['age']
df['experience_sqrt'] = np.sqrt(df['experience'])

# 4. Encode Categorical
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])
df = pd.get_dummies(df, columns=['department'], drop_first=True)

# 5. Remove Low Variance
low_var_features = df.var() < 0.01
df = df.drop(columns=df.columns[low_var_features])

# 6. Remove Highly Correlated
corr_features = find_highly_correlated(df, threshold=0.95)
df = df.drop(columns=corr_features)

# 7. Split Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 8. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Feature Selection
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# 10. Model & Evaluate
model = RandomForestClassifier()
model.fit(X_train_selected, y_train)
score = model.score(X_test_selected, y_test)

print(f"Model Score: {score:.3f}")

# 11. Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns[selector.get_support()],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance)
```

---

## 9. Feature Engineering for Different Models

### 9.1 Linear Models (LR, Lasso, Ridge)

**Best Practices:**
- Scale features (critical)
- Remove multicollinearity
- Create interactions (x1 × x2)
- Polynomial features for non-linearity

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LogisticRegression(penalty='l2'))
])
```

---

### 9.2 Tree Models (Decision Tree, Random Forest, XGBoost)

**Best Practices:**
- No scaling needed
- No need to remove multicollinearity
- No polynomial features (auto-learned)
- Can use categorical directly (some implementations)

```python
model = RandomForestClassifier()  # No scaling, no poly features needed
```

---

### 9.3 SVM

**Best Practices:**
- Scale features (critical)
- Remove outliers
- Feature selection (reduce dimensions)

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=50)),
    ('svm', SVC(kernel='rbf'))
])
```

---

### 9.4 Neural Networks

**Best Practices:**
- Scale features (0-1 or -1 to 1)
- No need to remove multicollinearity
- Embedding layers for categorical
- Batch normalization for very different scales

```python
from keras import layers, models

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

---

## 10. Common Feature Engineering Mistakes

### 10.1 Data Leakage

**Wrong:**
```python
# Leak: Test statistics used for training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Using all data including test!
X_train, X_test = train_test_split(X_scaled, y)
```

**Correct:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
X_test_scaled = scaler.transform(X_test)  # Apply to test
```

---

### 10.2 Creating Too Many Features

**Risk:** Curse of dimensionality, overfitting

**Solution:**
- Start with few, add incrementally
- Use feature selection
- Cross-validate to catch overfitting

---

### 10.3 Ignoring Business Context

**Wrong:** Create features purely statistically

**Right:** Features should be interpretable, align with business

```python
# Interpretable
df['debt_to_income_ratio'] = df['debt'] / df['income']

# Not interpretable
df['feature_123'] = df['x'] ** 2.3 + df['y'] * 0.1234
```

---

### 10.4 Not Testing Feature Value

**Wrong:** Create feature, assume it helps

**Right:** Cross-validate with/without feature

```python
# Model without feature
model1 = RandomForestClassifier()
score1 = cross_val_score(model1, X_without, y).mean()

# Model with feature
model2 = RandomForestClassifier()
score2 = cross_val_score(model2, X_with, y).mean()

# Feature valuable only if score2 > score1
```

---

## 11. Advanced Feature Engineering

### 11.1 Polynomial Features Smart Selection

```python
from sklearn.preprocessing import PolynomialFeatures

# Full: All combos (can explode dimensions)
poly_full = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_full.fit_transform(X)
# 10 features → 55 features!

# Better: Select top interactions
from itertools import combinations

top_interactions = []
for f1, f2 in combinations(X.columns, 2):
    interaction = X[f1] * X[f2]
    corr = abs(interaction.corr(y))
    if corr > 0.1:  # Keep if correlated with target
        top_interactions.append(interaction)

X_interactions = pd.concat([X] + top_interactions, axis=1)
```

---

### 11.2 Stacking Features (Meta-features)

**Idea:** Use predictions from one model as features for another

```python
# Level 0 models
model1 = RandomForestClassifier()
model2 = SVC(probability=True)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Create meta-features
meta_train = np.column_stack([
    model1.predict_proba(X_train)[:, 1],
    model2.predict_proba(X_train)[:, 1]
])

meta_test = np.column_stack([
    model1.predict_proba(X_test)[:, 1],
    model2.predict_proba(X_test)[:, 1]
])

# Level 1 model
meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)
final_pred = meta_model.predict(meta_test)
```

---

## 12. Tools & Libraries

**Automated Feature Engineering:**
```python
# Featuretools: Automated feature generation
from featuretools import dfs

es = EntitySet(id="data")
es.entity_from_dataframe(entity_id="transactions", dataframe=df)
feature_matrix, features = dfs(entityset=es, target_entity="transactions")

# Tsfresh: Time series features
from tsfresh import extract_features

extracted_features = extract_features(df, column_id="id", column_sort="time")
```

---

## 13. Key Takeaways

1. **Foundation:** Quality features > Complex model
2. **Process:** Create → Transform → Select → Scale
3. **Domain:** Leverage domain knowledge, not just statistics
4. **Avoid leakage:** Split before preprocessing
5. **Interactions:** Create multiplicative features for relationships
6. **Encoding:** One-hot for nominal, label for ordinal, target for high-cardinality
7. **Selection:** Remove low-variance, correlated, irrelevant features
8. **Scaling:** Critical for distance-based and gradient descent models
9. **Validation:** Test feature value with cross-validation
10. **Iteration:** Feature engineering is experimental, iterate!

---

## 14. Feature Engineering Checklist

```
□ Understand problem and data
□ Handle missing values
□ Detect and treat outliers
□ Create domain-specific features
□ Create interaction features
□ Transform skewed features
□ Encode categorical features
□ Remove constant/low-variance features
□ Remove highly correlated features
□ Scale numerical features (if needed)
□ Select relevant features
□ Validate improvements on test set
□ Document all transformations
□ Ensure no data leakage
```

---

**Congratulations! You've completed all 13 topics! 🎉**

You now have comprehensive notes on:
1. Core Statistics & Probability
2. Linear Regression
3. Logistic Regression
4. Decision Trees
5. k-Nearest Neighbors
6. Naive Bayes
7. Support Vector Machines
8. Random Forest
9. AdaBoost
10. Gradient Boosting
11. Model Evaluation & Metrics
12. Imbalanced Classification
13. Feature Engineering

**Next Steps:**
- Practice implementations with real datasets
- Work on Kaggle competitions
- Build end-to-end projects
- Deepen knowledge in specific areas (Deep Learning, NLP, Computer Vision)
- Contribute to open-source ML projects