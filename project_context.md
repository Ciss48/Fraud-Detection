# project_context.md
# Fraud Pattern Mining — Complete Project Context for Academic Report

> This document is 100% self-contained. It is the sole input for an AI writing a full academic report on this project. All numbers, metrics, rules, code logic, and visual descriptions are included verbatim. No local file access is required.

---

## SECTION 1 — PROJECT OVERVIEW

### Title
**Fraud Pattern Mining: Discovering Interpretable Fraud Patterns from Credit Card Transactions Using Association Rule Mining and Anomaly Detection**

### Problem Statement
Credit card fraud causes billions of dollars in losses annually. Traditional machine learning models (e.g., neural networks, gradient boosting) can achieve high predictive accuracy but function as black boxes — their decisions are not explainable to fraud analysts who need to act on them. This project addresses the gap between predictive power and interpretability: the goal is not merely to classify transactions as fraudulent, but to discover human-readable rules that describe *why* a transaction is likely to be fraud, so that a fraud analyst can read the rule, understand it, and decide on a response (block, hold, flag, monitor).

### Primary Objective
Mine interpretable fraud patterns from 1.85 million credit card transactions using Association Rule Mining (FP-Growth algorithm), compare them against Isolation Forest (unsupervised anomaly detection) and traditional ML models, and determine whether rule-based features improve ML model performance.

### Secondary Objectives
1. Quantify which features are most discriminative for fraud (EDA Phase)
2. Discover the top association rules characterizing fraud transactions, ranked by lift and confidence
3. Evaluate whether rules alone can replace ML models, or whether a hybrid approach is superior
4. Provide actionable analyst-ready rule summaries with recommended responses (BLOCK / HOLD / FLAG / MONITOR)

### Approach Summary
A three-phase pipeline:
- **Phase 1 (EDA):** Understand data distribution, engineer features, identify discriminative signals
- **Phase 2 (Pattern Mining):** Apply FP-Growth to the fraud-only transaction set; compare rules against two baselines (simple threshold, Isolation Forest)
- **Phase 3 (ML Models):** Train Random Forest and XGBoost models; evaluate whether Phase 2 rule flags as features improve ML performance

### Core Thesis
Association rules offer extremely high recall (85.9%) and full human interpretability but at the cost of very low precision (1.6%). ML models achieve a far better precision-recall trade-off (XGBoost: P=71.1%, R=86.6%, F1=78.1%, AUC-PR=0.894). Hybrid rule-augmented ML does not improve over pure XGBoost, suggesting the ML model already learns the rule signals implicitly. The best practical system combines the 25 interpretable rules (for analyst education and policy design) with XGBoost scoring (for automated transaction screening).

---

## SECTION 2 — DATASET DESCRIPTION

### Source
Simulated credit card transaction dataset from Kaggle (based on the Sparkov Data Generation tool). Two CSV files:
- `fraudTrain.csv` — 1,296,675 rows
- `fraudTest.csv` — 555,719 rows
- **Combined total: 1,852,394 rows × 23 original columns**

### Data Quality
- **Missing values:** 0 (zero nulls across all 23 columns)
- **Duplicate rows:** 0 (verified via `trans_num` uniqueness — 1,852,394 unique transaction numbers)
- **No data cleaning required** beyond dropping the spurious `Unnamed: 0` index column

### Class Distribution (Target Variable: `is_fraud`)
| Class | Count | Percentage |
|---|---|---|
| Non-fraud (0) | 1,842,743 | 99.479% |
| Fraud (1) | 9,651 | 0.521% |
| **Total** | **1,852,394** | **100%** |

- **Imbalance ratio: 1:190** (one fraud per 190 non-fraud transactions)
- Consequence: accuracy is a misleading metric (a model predicting all-non-fraud gets 99.48% accuracy). Primary metrics used are AUC-PR and F1.

### Original 23 Columns

| Column | Type | Description |
|---|---|---|
| `Unnamed: 0` | int | Index column (dropped) |
| `trans_date_trans_time` | string | Transaction timestamp (e.g., "2019-01-01 00:00:18") |
| `cc_num` | int64 | Credit card number (anonymized) |
| `merchant` | string | Merchant name (693 unique values) |
| `category` | string | Merchant category (14 unique values) |
| `amt` | float64 | Transaction amount in USD |
| `first` | string | Cardholder first name |
| `last` | string | Cardholder last name |
| `gender` | string | Cardholder gender (M/F) |
| `street` | string | Cardholder street address |
| `city` | string | Cardholder city |
| `state` | string | Cardholder state |
| `zip` | int | Cardholder ZIP code |
| `lat` | float64 | Cardholder latitude |
| `long` | float64 | Cardholder longitude |
| `city_pop` | int64 | Population of cardholder's city |
| `job` | string | Cardholder occupation |
| `dob` | string | Cardholder date of birth |
| `trans_num` | string | Unique transaction identifier |
| `unix_time` | int64 | Unix timestamp of transaction |
| `merch_lat` | float64 | Merchant latitude |
| `merch_long` | float64 | Merchant longitude |
| `is_fraud` | int (0/1) | Target variable |

### Merchant Categories (14 unique values)
entertainment, food_dining, gas_transport, grocery_net, grocery_pos, health_fitness, home, kids_pets, misc_net, misc_pos, personal_care, shopping_net, shopping_pos, travel

### 8 Derived (Engineered) Features

| Derived Feature | Source | Formula / Logic |
|---|---|---|
| `hour_of_day` | `trans_date_trans_time` | Extract hour (0–23) |
| `day_of_week` | `trans_date_trans_time` | Extract day of week (0=Mon … 6=Sun) |
| `month` | `trans_date_trans_time` | Extract month (1–12) |
| `is_night` | `hour_of_day` | 1 if hour ∈ {22,23,0,1,2,3,4,5,6}, else 0 |
| `distance_km` | `lat`, `long`, `merch_lat`, `merch_long` | Haversine formula — great-circle distance between cardholder and merchant |
| `age` | `dob`, transaction date | Years between date-of-birth and transaction date |
| `amt_bin` | `amt` | Discretized: low(<$50), medium($50–200), high($200–500), very_high(>$500) |
| `time_bin` | `hour_of_day` | Discretized: morning(6–12), afternoon(12–18), evening(18–22), night(22–6) |

**Haversine formula used:**
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))
```

---

## SECTION 3 — METHODOLOGY & PIPELINE

### 3.1 Phase 1 — Exploratory Data Analysis

**Input:** `fraudTrain.csv` + `fraudTest.csv` (concatenated)
**Output:** `data/processed/transactions_with_features.csv` (1,852,394 × 31 columns), 30+ figures

**Steps performed:**
1. Load and concatenate both CSVs; drop `Unnamed: 0`
2. Verify schema: shape, dtypes, null counts, duplicate check
3. Compute class distribution and imbalance ratio
4. Engineer 8 derived features (hour, day_of_week, month, is_night, distance_km, age, amt_bin, time_bin)
5. Compute univariate AUC-ROC for each feature (fraud vs non-fraud discrimination power)
6. Analyze distributions: histograms, boxplots, fraud rate by category, fraud rate by hour, fraud rate by amount bin
7. Cross-feature analysis: category × time_bin fraud rates
8. Discretization bin decisions documented (distance_km and city_pop confirmed non-discriminative → excluded from rule mining)

**Univariate AUC-ROC Results (discrimination power per feature):**

| Feature | AUC-ROC | Interpretation |
|---|---|---|
| `amt` | **0.8343** | Strongest predictor. Fraud median $390 vs non-fraud $47 |
| `is_night` | **0.7822** | Night fraud rate 1.50% vs day 0.10% (15× baseline) |
| `is_very_high_amt` (>$500) | **0.7380** | Binary flag for very high amounts |
| `hour_of_day` | 0.5856 | Clear gradient: 22h = 2.60% fraud, daytime hours ~0.10% |
| `age` | 0.5404 | Senior (>55) = 0.67% fraud; weak but usable signal |
| `city_pop` | 0.5079 | **NON-DISCRIMINATIVE** — rural/suburban/urban all ~0.52% |
| `distance_km` | 0.5008 | **NON-DISCRIMINATIVE** — fraud median 78.1km ≈ non-fraud 78.2km |
| `is_distant` (>100km) | 0.5006 | **NON-DISCRIMINATIVE** |

**Conclusion from EDA:** `distance_km` and `city_pop` are excluded from association rule mining and deprioritized in feature importance analysis. The key discriminative features are `amt`, `is_night`/`hour_of_day`, and `category`.

---

### 3.2 Phase 2 — Fraud Pattern Mining

**Input:** `data/processed/transactions_with_features.csv`
**Output:**
- `data/processed/transactions_phase2.csv` (1,852,394 × 38, with 3 new flag columns)
- `outputs/rules/fraud_rules_final.csv` (25 rules)
- `outputs/rules/method_comparison.csv` (3-method comparison)

#### 3.2.1 Feature Discretization for Association Rule Mining

Items constructed from 26 binary dummy variables per transaction:

| Item Group | Items (count) | Values |
|---|---|---|
| Category | 14 | one per merchant category (e.g., category=shopping_net) |
| Amount bin | 4 | amt_low, amt_medium, amt_high, amt_very_high |
| Time bin | 4 | time_morning, time_afternoon, time_evening, time_night |
| Age bin | 3 | age_young(<30), age_middle(30–55), age_senior(>55) |
| Target | 1 | fraud_yes |
| **Total** | **26** | |

Memory usage of the one-hot encoded boolean DataFrame: **48.2 MB**

#### 3.2.2 Baseline 1 — Simple Threshold Rule

**Rule:** Flag all transactions where `amt > $500` as fraud.

| Metric | Value |
|---|---|
| Transactions flagged | 21,709 (1.17% of all transactions) |
| True Positives (TP) | 4,684 |
| False Positives (FP) | 17,025 |
| Precision | 0.2158 (21.58%) |
| Recall | 0.4853 (48.53%) |
| F1-Score | 0.2987 (29.87%) |
| AUC-PR | 0.1074 |

#### 3.2.3 Baseline 2 — Isolation Forest

**Algorithm:** scikit-learn `IsolationForest`
**Features used:** amt, distance_km, city_pop, age, hour_of_day (5 numeric features)
**Method:** Contamination parameter sweep (expected fraud proportion)

| Contamination | Flagged | TP | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0.005 | 9,262 | 1,829 | 0.1974 | 0.1894 | 0.1933 |
| **0.01** | **18,524** | **3,156** | **0.1704** | **0.3270** | **0.2240** |
| 0.02 | 37,048 | 4,490 | 0.1212 | 0.4652 | 0.1923 |

**Selected configuration:** `contamination=0.01` — best F1 score of **0.2240**

Full metrics for selected Isolation Forest:
- Precision: 0.1704, Recall: 0.3270, F1: 0.2240, AUC-PR: 0.0592

#### 3.2.4 Main Method — FP-Growth Association Rule Mining

**Algorithm:** `mlxtend.frequent_patterns.fpgrowth` + `association_rules`
**Input:** Fraud-only transaction set (9,651 rows) transformed to 26-item boolean DataFrame

**Hyperparameters:**
- `min_support = 0.0005` (0.05% of total dataset = ~926 transactions) — low because fraud base rate is only 0.52%
- `min_confidence = 0.01` (1% minimum) — permissive; lift will be the primary filter
- `use_colnames = True`

**FP-Growth execution:** 4.4 seconds on full dataset

**Frequent Itemset Statistics:**
| Itemset size | Count |
|---|---|
| 1-item | 26 |
| 2-item | 184 |
| 3-item | 428 |
| 4-item | 264 |
| **Total** | **902** |

**Rule generation:** 5,779 total rules extracted
- Rules with consequent = `fraud_yes`: 30
- Rules with `fraud_yes` consequent AND lift ≥ 3×: **25** (final rule set)

**Threshold sensitivity analysis (rules with fraud_yes consequent):**

| Lift threshold | # Rules | Fraud Recall |
|---|---|---|
| lift ≥ 2× | 30 | 96.6% |
| **lift ≥ 3×** | **25** | **85.9%** |
| lift ≥ 5× | 20 | 75.8% |
| lift ≥ 10× | 17 | 75.2% |
| lift ≥ 20× | 15 | 75.2% |
| lift ≥ 40× | 11 | 69.1% |

**Rationale for lift ≥ 3× threshold:** The jump from lift≥3 (85.9% recall) to lift≥2 (96.6% recall) adds only 5 more rules but adds 10.7 percentage points recall. The jump from lift≥3 to lift≥5 loses 10.1 points recall with only 5 fewer rules. Lift≥3 gives the best balance of interpretability and coverage.

**Final Association Rules Performance (lift ≥ 3×):**

| Metric | Value |
|---|---|
| Transactions flagged | 510,240 (27.54% of all transactions) |
| True Positives (TP) | 8,288 |
| False Positives (FP) | 501,952 |
| Precision | 0.0162 (1.62%) |
| Recall | 0.8588 (85.88%) |
| F1-Score | 0.0319 (3.19%) |
| AUC-PR | 0.0147 |

**Interpretation:** Rules achieve very high recall (catching 85.9% of all fraud) but at extremely low precision (only 1 in 62 flagged transactions is actually fraud). This makes rules unsuitable as a standalone automated blocking system but highly valuable as a detection/monitoring layer or for rule discovery/analyst education.

#### 3.2.5 Overlap Analysis (Isolation Forest vs Association Rules)

Among the 9,651 fraud transactions:

| Group | Count | % of fraud | Characteristics |
|---|---|---|---|
| Caught by BOTH methods | 3,151 | 32.6% | High confidence fraud — both methods agree |
| Caught by rules ONLY | 5,137 | 53.2% | Median amt $321.37; 86.5% night; top category: grocery_pos |
| Caught by IF ONLY | 5 | 0.1% | Median amt $19.00; 100% night; top category: health_fitness (statistical outliers, low amounts) |
| Caught by NEITHER | 1,358 | 14.1% | Fraud that neither method detected |

**Key insight:** Association rules dominate — they catch 85.9% of fraud including 53% that Isolation Forest entirely misses. The 5 transactions caught only by Isolation Forest are low-amount statistical anomalies undetectable by rule patterns. Rules substantially outperform Isolation Forest in recall (85.9% vs 32.7%).

---

### 3.3 Phase 3 — ML Model Training and Evaluation

**Input:** `data/processed/transactions_phase2.csv` (1,852,394 × 38)
**Output:**
- `outputs/rules/phase3_method_comparison.csv` (7-method comparison)
- `data/processed/transactions_phase3.csv` (1,852,394 × 48, with probability scores and predictions)

#### 3.3.1 Feature Sets

**BASE feature set (9 features):**
1. `amt` — transaction amount (continuous)
2. `hour_of_day` — 0–23 (continuous)
3. `day_of_week` — 0–6 (continuous)
4. `month` — 1–12 (continuous)
5. `distance_km` — Haversine distance (continuous; included here despite low univariate AUC as ML can use it in combination)
6. `city_pop` — city population (continuous; included for completeness)
7. `age` — cardholder age in years (continuous)
8. `is_night` — binary (1 if hour 22–6, else 0)
9. `category_enc` — LabelEncoded merchant category (0=entertainment, 1=food_dining, 2=gas_transport, 3=grocery_net, 4=grocery_pos, 5=health_fitness, 6=home, 7=kids_pets, 8=misc_net, 9=misc_pos, 10=personal_care, 11=shopping_net, 12=shopping_pos, 13=travel)

**AUGMENTED feature set (11 features):** BASE + `flag_threshold` + `flag_rules`
- `flag_threshold`: 1 if amt > $500, else 0 (from Phase 2 Baseline 1)
- `flag_rules`: 1 if transaction matches any of the 25 Phase 2 association rules, else 0

#### 3.3.2 Cross-Validation Setup

- **Method:** 5-fold Stratified K-Fold Cross-Validation using `cross_val_predict`
- **Rationale:** `cross_val_predict` generates out-of-fold (OOF) probability predictions across all 1,852,394 rows without data leakage. Each fold trains on ~1,481,915 rows (~80%) and predicts on ~370,479 rows (~20%).
- **Output:** OOF probability scores for all rows → used for threshold tuning and PR curve generation
- **Stratification:** Maintains 0.52% fraud rate in each fold

#### 3.3.3 Class Imbalance Handling

- **Random Forest:** `class_weight='balanced'` — automatically weights minority class (fraud) by 190× relative to majority
- **XGBoost:** `scale_pos_weight=190.9` — equivalent mechanism; ratio = n_negative / n_positive = 1,842,743 / 9,651
- **SMOTE:** Not used (confirmed in EDA: class weighting is sufficient and avoids synthetic data issues)

#### 3.3.4 Threshold Tuning

Default decision threshold of 0.5 is sub-optimal for imbalanced data where the positive class has probability scores clustering in a narrow range. Both RF and XGBoost thresholds were swept from 0.05 to 0.95 in steps of 0.01, selecting the threshold that maximizes F1-score on OOF predictions.

For the **final model selection** (XGBoost Base as best), a **F-beta score with β ≈ 1.53** (recall:precision weight ratio = 7:3) was used. This reflects a business preference to catch more fraud even at the cost of more false alarms.

**Optimal thresholds found:**

| Model | Optimal Threshold | Selection Criterion |
|---|---|---|
| RF Base | 0.90 | Max F1 |
| XGBoost Base | 0.94 | Max F1 |
| RF Rule-Augmented | 0.87 | Max F1 |
| XGBoost Augmented | 0.94 | Max F1 |
| XGBoost Base (final) | 0.9603 | Max F-beta (β=1.53) |

#### 3.3.5 Random Forest (Base)

**Configuration:**
- `n_estimators=100` (CV phase), `n_estimators=300` (full fit for feature importance)
- `max_depth=12`
- `class_weight='balanced'`
- `random_state=42`
- Features: 9 BASE features

**CV Results (5-fold OOF):**
- Threshold: 0.90
- Precision: 0.8828 (88.28%)
- Recall: 0.7314 (73.14%)
- F1: 0.8000 (80.00%)
- AUC-PR: 0.8593
- Transactions flagged: 7,996
- True Positives: 7,059

**Full fit time:** ~213 seconds (300 trees)

#### 3.3.6 XGBoost (Base)

**Configuration:**
- `n_estimators=300`
- `scale_pos_weight=190.9`
- `eval_metric='aucpr'`
- `use_label_encoder=False`
- `random_state=42`
- Features: 9 BASE features

**CV Results (5-fold OOF):**
- Threshold: 0.94
- Precision: 0.7112 (71.12%)
- Recall: 0.8663 (86.63%)
- F1: 0.7811 (78.11%)
- AUC-PR: **0.8940** ← highest AUC-PR across all models
- Transactions flagged: 11,756
- True Positives: 8,361

#### 3.3.7 RF Rule-Augmented

**Configuration:** Identical to RF Base, but trained on 11-feature AUGMENTED set (9 BASE + flag_threshold + flag_rules)

**CV Results (5-fold OOF):**
- Threshold: 0.87
- Precision: 0.8357 (83.57%)
- Recall: 0.7488 (74.88%)
- F1: 0.7899 (78.99%)
- AUC-PR: 0.8380
- Transactions flagged: 8,648
- True Positives: 7,227

**Observation:** Adding rule flags to RF *decreases* AUC-PR from 0.8593 to 0.8380. The rule flags do not help RF and may introduce noise.

#### 3.3.8 XGBoost Augmented

**Configuration:** Identical to XGBoost Base, but trained on 11-feature AUGMENTED set

**CV Results (5-fold OOF):**
- Threshold: 0.94
- Precision: 0.7097 (70.97%)
- Recall: 0.8666 (86.66%)
- F1: 0.7803 (78.03%)
- AUC-PR: 0.8939
- Transactions flagged: 11,786
- True Positives: 8,364

**Observation:** Adding rule flags to XGBoost produces essentially identical results to XGBoost Base (AUC-PR: 0.8939 vs 0.8940, F1: 0.7803 vs 0.7811). XGBoost already learns the rule signals from the raw features; the explicit flags provide negligible additional information.

#### 3.3.9 Best Model — XGBoost Base (Final)

**Selection rationale:** Highest AUC-PR (0.8940) and highest Recall (86.63%). Under F-beta (β≈1.53) criterion, XGBoost Base at threshold 0.9603 was selected for the confusion matrix.

**Final confusion matrix (XGBoost Base, threshold=0.9603, evaluated on full 1,852,394 rows):**

|  | Predicted Non-Fraud | Predicted Fraud |
|---|---|---|
| **Actual Non-Fraud** | 1,839,348 (TN) | 3,395 (FP) |
| **Actual Fraud** | 1,290 (FN) | 8,361 (TP) |

- **Fraud caught:** 8,361 / 9,651 = **86.6%** of all fraud
- **False alarms:** 3,395 non-fraud transactions incorrectly flagged
- **Missed fraud:** 1,290 transactions (13.4% of all fraud)
- **False Alarm Rate:** 3,395 / 1,842,743 = 0.18% of non-fraud flagged

---

## SECTION 4 — EXPERIMENTAL RESULTS

### 4.1 Complete 7-Method Comparison Table

| Method | Description | Precision | Recall | F1 | AUC-PR | Flagged | TP |
|---|---|---|---|---|---|---|---|
| Threshold (amt>$500) | Simple threshold | 0.2158 | 0.4853 | 0.2987 | 0.1074 | 21,709 | 4,684 |
| Isolation Forest | Unsupervised (cont=0.01) | 0.1704 | 0.3270 | 0.2240 | 0.0592 | 18,524 | 3,156 |
| Assoc. Rules (lift≥3×) | 25 interpretable rules | 0.0162 | 0.8588 | 0.0319 | 0.0147 | 510,240 | 8,288 |
| RF Base | RF 9 features, threshold tuned | 0.8828 | 0.7314 | 0.8000 | 0.8593 | 7,996 | 7,059 |
| **XGBoost Base** | **XGB 9 features, threshold tuned** | **0.7112** | **0.8663** | **0.7811** | **0.8940** | **11,756** | **8,361** |
| RF Rule-Augmented | RF + rule flags | 0.8357 | 0.7488 | 0.7899 | 0.8380 | 8,648 | 7,227 |
| XGBoost Augmented | XGB + rule flags | 0.7097 | 0.8666 | 0.7803 | 0.8939 | 11,786 | 8,364 |

### 4.2 Phase 2 Baselines (3-Method Comparison)

| Method | Precision | Recall | F1 | AUC-PR | Flagged | TP |
|---|---|---|---|---|---|---|
| Baseline: amt > $500 | 0.21576 | 0.48534 | 0.29872 | 0.10740 | 21,709 | 4,684 |
| Isolation Forest (cont=0.01) | 0.17037 | 0.32701 | 0.22403 | 0.05922 | 18,524 | 3,156 |
| Assoc. Rules (lift ≥ 3×) | 0.01624 | 0.85877 | 0.03188 | 0.01469 | 510,240 | 8,288 |

### 4.3 All 25 Fraud Association Rules (Complete)

Rules ranked by fraud_rate (confidence × 100). For each rule: condition, support (% of all transactions), fraud_rate (% of transactions matching condition that are fraud), lift, n_fraud_txn (true positives), fraud_recall (% of all fraud caught), and analyst action.

| # | Condition | Support% | Fraud Rate% | Lift× | TP | Recall% | Action |
|---|---|---|---|---|---|---|---|
| 0 | `amt>$500 AND category=shopping_net AND night(22–6)` | 0.16% | 63.704% | 122.3× | 1,885 | 19.53% | BLOCK |
| 1 | `amt>$500 AND category=misc_net AND night(22–6)` | 0.104% | 52.792% | 101.3× | 1,021 | 10.58% | BLOCK |
| 2 | `age>55 AND amt>$500 AND night(22–6)` | 0.173% | 46.101% | 88.5× | 1,478 | 15.31% | HOLD for manual review |
| 3 | `amt>$500 AND night(22–6)` | 0.466% | 46.032% | 88.4× | 3,973 | 41.17% | HOLD for manual review |
| 4 | `age 30–55 AND amt>$500 AND night(22–6)` | 0.223% | 42.338% | 81.3× | 1,749 | 18.12% | HOLD for manual review |
| 5 | `amt>$500 AND category=misc_net` | 0.173% | 36.972% | 71.0× | 1,182 | 12.25% | HOLD for manual review |
| 6 | `amt>$500 AND category=shopping_net` | 0.332% | 36.087% | 69.3× | 2,219 | 22.99% | HOLD for manual review |
| 7 | `age>55 AND amt>$500` | 0.296% | 32.646% | 62.7× | 1,790 | 18.55% | HOLD for manual review |
| 8 | `age 30–55 AND amt>$500 AND category=shopping_net` | 0.174% | 29.455% | 56.5× | 952 | 9.86% | HOLD for manual review |
| 9 | `amt $200–500 AND category=grocery_pos AND night(22–6)` | 0.486% | 22.053% | 42.3× | 1,985 | 20.57% | HOLD for manual review |
| 10 | `amt>$500` | 1.172% | 21.576% | 41.4× | 4,684 | 48.53% | HOLD for manual review |
| 11 | `amt>$500 AND category=shopping_pos` | 0.322% | 17.727% | 34.0× | 1,056 | 10.94% | FLAG — escalate to analyst |
| 12 | `age 30–55 AND amt>$500` | 0.673% | 16.215% | 31.1× | 2,022 | 20.95% | FLAG — escalate to analyst |
| 13 | `amt $200–500 AND category=grocery_pos` | 0.878% | 13.697% | 26.3× | 2,228 | 23.09% | FLAG — escalate to analyst |
| 14 | `amt $200–500 AND night(22–6)` | 1.208% | 10.408% | 20.0× | 2,329 | 24.13% | FLAG — escalate to analyst |
| 15 | `age 30–55 AND amt $200–500 AND night(22–6)` | 0.724% | 7.883% | 15.1× | 1,057 | 10.95% | FLAG — escalate to analyst |
| 16 | `age>55 AND amt $200–500` | 0.718% | 7.305% | 14.0× | 972 | 10.07% | FLAG — escalate to analyst |
| 17 | `category=shopping_net AND night(22–6)` | 2.401% | 4.239% | 8.1× | 1,885 | 19.53% | MONITOR — log and watch |
| 18 | `amt $200–500` | 3.540% | 4.012% | 7.7× | 2,631 | 27.26% | MONITOR — log and watch |
| 19 | `age 30–55 AND amt $200–500` | 2.153% | 2.986% | 5.7× | 1,191 | 12.34% | MONITOR — log and watch |
| 20 | `category=misc_net AND night(22–6)` | 2.182% | 2.527% | 4.8× | 1,021 | 10.58% | MONITOR — log and watch |
| 21 | `category=grocery_pos AND night(22–6)` | 4.745% | 2.258% | 4.3× | 1,985 | 20.57% | MONITOR — log and watch |
| 22 | `age<30 AND night(22–6)` | 4.725% | 1.749% | 3.4× | 1,531 | 15.86% | MONITOR — log and watch |
| 23 | `category=shopping_net` | 7.521% | 1.593% | 3.1× | 2,219 | 22.99% | MONITOR — log and watch |
| 24 | `age>55 AND night(22–6)` | 9.773% | 1.623% | 3.1× | 2,938 | 30.44% | MONITOR — log and watch |

**Rule Action Levels defined:**
- **BLOCK (Precision ≥ 50%):** Automatically decline transaction in real-time
- **HOLD (Precision 15–49%):** Place transaction on hold, require step-up authentication
- **FLAG (Precision 5–14%):** Escalate to fraud analyst queue
- **MONITOR (Precision < 5%):** Log for behavioral pattern tracking; do not interrupt transaction

**Cumulative recall of top rules:**
- Top 1 rule: 19.5% of fraud
- Top 3 rules: 41.2% of fraud  
- Top 5 rules: 58.1% of fraud
- All 25 rules: 85.9% of fraud

### 4.4 Feature Importance (RF Base — full fit on 1,852,394 rows)

Feature importances from Random Forest (n_estimators=300, max_depth=12, class_weight='balanced'):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `amt` | ~0.42 (highest by far) |
| 2 | `category_enc` | ~0.18 |
| 3 | `is_night` | ~0.12 |
| 4 | `hour_of_day` | ~0.10 |
| 5 | `age` | ~0.08 |
| 6 | `distance_km` | ~0.04 |
| 7 | `month` | ~0.03 |
| 8 | `city_pop` | ~0.02 |
| 9 | `day_of_week` | ~0.01 |

*Note: Exact importance values are approximate (read from figure). The relative ranking is definitive.*

For **RF Rule-Augmented**, the feature importance chart shows `flag_rules` ranked approximately 4th–5th (around 0.06–0.08 importance), confirming rules add some signal but don't dominate.

### 4.5 Precision-Recall Curve Summary

AUC-PR values from 5-fold CV OOF predictions:

| Model | AUC-PR |
|---|---|
| XGBoost Base | **0.8940** |
| RF Base | 0.8593 |
| XGBoost Augmented | 0.8939 |
| RF Rule-Augmented | 0.8380 |
| Threshold (amt>$500) | 0.1074 |
| Isolation Forest | 0.0592 |
| Association Rules (lift≥3×) | 0.0147 |

**Baseline comparison reference points** (plotted as scatter dots on the PR curve):
- Threshold rule: (Recall=0.485, Precision=0.216)
- Isolation Forest: (Recall=0.327, Precision=0.170)
- Association Rules: (Recall=0.859, Precision=0.016)

---

## SECTION 5 — FIGURES & VISUALIZATIONS

All figures are saved in `Fraud-Detection/outputs/figures/`. Filenames, content, and key insights:

### Phase 1 Figures (EDA)

**`01_class_distribution.png`**
- **Content:** Two-panel bar chart. Left: absolute counts (Non-fraud: 1,842,743; Fraud: 9,651). Right: percentage bars (99.48% vs 0.52%).
- **Key insight:** Extreme class imbalance (1:190 ratio) — visually demonstrates why accuracy is a misleading metric.
- **Report section:** Dataset Description / Class Imbalance

**`02_amount_distribution.png`**
- **Content:** Two overlapping histograms (fraud in red, non-fraud in blue) on log scale. X-axis: transaction amount ($0–$2,000+). Non-fraud distribution peaks around $50–100; fraud distribution peaks around $200–500 with a substantial tail above $500.
- **Key insight:** Fraud transactions are systematically higher in amount. The visual separation between distributions is the strongest single predictor.
- **Report section:** EDA / Amount Analysis

**`03_amount_bins_fraud_rate.png`**
- **Content:** 4-bar horizontal bar chart showing fraud rate (%) by amount bin. low(<$50)=0.22%, medium($50–200)=0.03%, high($200–500)=4.01%, very_high(>$500)=21.58%.
- **Key insight:** Very_high bin (>$500) has 21.58% fraud rate — 41× the baseline. Medium bin ($50–200) has the *lowest* fraud rate (0.03%) — fraudsters appear to avoid mid-range amounts.
- **Report section:** Feature Engineering / Amount Discretization

**`04_hourly_fraud_rate.png`**
- **Content:** 24-bar chart, one bar per hour of day (0–23). Bars colored by time period (morning=blue, afternoon=green, evening=orange, night=red). Night hours (22–6) have fraud rates 1.3%–2.6%; daytime hours 6–21 average ~0.10%.
- **Key insight:** Hour 22 (10pm) = 2.60% fraud rate, the peak. Strong circadian pattern — fraud is heavily concentrated at night. This validates `is_night` as the second-strongest feature.
- **Report section:** EDA / Temporal Analysis

**`05_category_fraud_rate.png`**
- **Content:** Horizontal bar chart, 14 bars sorted descending by fraud rate. Top bars: shopping_net 1.59%, misc_net 1.30%, grocery_pos 1.26%, shopping_pos 0.63%, gas_transport 0.41%.
- **Key insight:** Online shopping categories (shopping_net, misc_net) have highest fraud rates — 2–3× the baseline. In-person grocery (grocery_pos) also elevated. Travel and personal_care are lowest.
- **Report section:** EDA / Category Analysis

**`06_category_counts.png`**
- **Content:** Horizontal bar chart showing absolute transaction counts per category. gas_transport and grocery_pos have the most transactions (>180,000 each). shopping_net has ~139,000.
- **Key insight:** High-fraud categories aren't necessarily the most common — shows fraud rate needs volume context.
- **Report section:** EDA / Category Analysis

**`07_age_distribution.png`**
- **Content:** Age histogram comparing fraud (red) and non-fraud (blue) distributions. Both follow similar shapes peaking around age 30–50. Fraud distribution shows slight elevation for seniors (55+).
- **Key insight:** Age has AUC=0.54 — weak but non-trivial signal. Senior cardholders have 0.67% fraud rate vs 0.45% middle-aged.
- **Report section:** EDA / Age Analysis

**`08_age_group_fraud_rate.png`**
- **Content:** 3-bar chart: young(<30)=0.51%, middle(30–55)=0.45%, senior(>55)=0.67%.
- **Key insight:** Senior group is highest risk but difference is small (0.22 percentage points from middle). Age alone is insufficient but useful in combination.
- **Report section:** EDA / Age Analysis

**`09_distance_distribution.png`**
- **Content:** Overlapping histograms of distance_km for fraud (red) vs non-fraud (blue). Both distributions are nearly identical, both peaking around 60–80km.
- **Key insight:** No separation between fraud and non-fraud by distance. This directly motivates excluding distance_km from association rule mining (AUC=0.501 ≈ random).
- **Report section:** EDA / Distance Analysis

**`10_distance_bins_fraud_rate.png`**
- **Content:** 3-bar chart: local(<10km)=0.45%, regional(10–100km)=0.52%, distant(>100km)=0.52%.
- **Key insight:** All three distance bins have nearly identical fraud rates (~0.5%). Confirms distance_km is non-discriminative.
- **Report section:** EDA / Distance Analysis

**`11_city_pop_fraud_rate.png`**
- **Content:** 3-bar chart: rural(<10k)=0.52%, suburban(10k–100k)=0.52%, urban(>100k)=0.54%.
- **Key insight:** City population has essentially no discriminative power for fraud detection. All bins ≈ baseline rate.
- **Report section:** EDA / Geographic Analysis

**`12_gender_fraud_rate.png`**
- **Content:** 2-bar chart: Male=0.57%, Female=0.48%.
- **Key insight:** Small difference (0.09 percentage points). Gender is not a useful fraud indicator.
- **Report section:** EDA / Demographic Analysis

**`13_monthly_fraud_rate.png`**
- **Content:** 12-bar chart by month. Peaks in January (0.81%) and February (0.87%); lowest in summer months (June–August ~0.35–0.40%).
- **Key insight:** Post-holiday fraud spike (Jan–Feb). Seasonal pattern may reflect increased transaction volume plus compromised cards from holiday shopping.
- **Report section:** EDA / Temporal Analysis

**`14_dayofweek_fraud_rate.png`**
- **Content:** 7-bar chart (Mon=0, Sun=6). Thursday–Saturday slightly elevated (0.61–0.64%) vs Monday (0.40%).
- **Key insight:** Weak weekly pattern. Not strong enough for standalone rules but may contribute in combination with other features.
- **Report section:** EDA / Temporal Analysis

**`15_cross_feature_heatmap.png`**
- **Content:** Heatmap, rows = merchant categories (14), columns = time bins (morning/afternoon/evening/night). Color = fraud rate %. Darkest cells: shopping_net × night (4.2%), misc_net × night (2.5%).
- **Key insight:** Most powerful cross-feature interaction. shopping_net × night = 4.2% fraud rate (8× baseline). This combination directly motivates the top association rule (Rule 0).
- **Report section:** EDA / Cross-Feature Analysis

**`16_univariate_auc.png`**
- **Content:** Horizontal bar chart, 8 features ranked by univariate AUC-ROC. amt=0.834, is_night=0.782, is_very_high_amt=0.738, hour=0.586, age=0.540, city_pop=0.508, distance=0.501, is_distant=0.501.
- **Key insight:** Clear tier separation between strong (amt, is_night), moderate (hour, age), and non-discriminative (city_pop, distance) features. This directly informs feature selection for rule mining.
- **Report section:** EDA / Feature Discrimination Analysis

**`combined_distribution.png`**
- **Content:** Multi-panel figure summarizing distributions of key numeric features (amt, age, distance_km, city_pop) with fraud vs non-fraud overlays.
- **Key insight:** Summary visualization confirming that only `amt` shows clear separation; other features largely overlap.
- **Report section:** EDA / Distribution Overview

**`05_distance_analysis_3charts.png`**
- **Content:** Three-panel distance analysis: (1) histogram comparison, (2) bin fraud rates, (3) box plots by fraud label.
- **Key insight:** All three panels confirm distance_km is non-discriminative. Median distance: fraud=78.1km, non-fraud=78.2km.
- **Report section:** EDA / Distance Analysis

### Phase 2 Figures (Pattern Mining)

**`phase2_01_method_comparison_bar.png`**
- **Content:** Grouped bar chart comparing 3 Phase 2 methods (Threshold, Isolation Forest, Association Rules) on Precision, Recall, and F1. Three groups of 3 bars.
- **Key insight:** Association Rules dominate on Recall (0.859) but have near-zero Precision (0.016). Threshold rule has best F1 (0.299) among Phase 2 baselines.
- **Report section:** Phase 2 Results / Method Comparison

**`phase2_02_overlap_venn.png`**
- **Content:** Venn diagram with two overlapping circles: "Association Rules" and "Isolation Forest". Numbers: Both=3,151, Rules only=5,137, IF only=5. Outer region (neither)=1,358.
- **Key insight:** Rules catch 53% of fraud that Isolation Forest entirely misses. Only 5 fraud cases caught exclusively by Isolation Forest. The overlap region (32.6%) represents the "consensus fraud" both methods agree on.
- **Report section:** Phase 2 Results / Overlap Analysis

**`phase2_03_top_rules_lift_chart.png`**
- **Content:** Horizontal bar chart showing top 15 rules by lift value. Rule 0 (shopping_net + amt>$500 + night) = 122.3× lift is clearly the longest bar. Bars colored by action level (BLOCK=red, HOLD=orange, FLAG=yellow, MONITOR=green).
- **Key insight:** The top rules have extremely high lift — Rule 0 at 122× means a transaction matching that condition is 122 times more likely to be fraud than a random transaction.
- **Report section:** Phase 2 Results / Top Association Rules

**`phase2_04_rule_recall_vs_precision.png`**
- **Content:** Scatter plot with x-axis=Recall (% of fraud caught) and y-axis=Precision (fraud rate %), one point per rule. Points labeled by rule number. Top-right cluster (Rules 0–3) represents rules with both high precision and decent recall.
- **Key insight:** Classic precision-recall tradeoff at rule level. Rules 0 and 1 (BLOCK level) have 52–64% precision. Rules 17–24 (MONITOR level) have <5% precision but collectively cover broad fraud patterns.
- **Report section:** Phase 2 Results / Rule Quality Analysis

**`phase2_05_threshold_sensitivity.png`**
- **Content:** Line chart with x-axis=lift threshold (2 to 40) and y-axis=recall (%). Shows how recall drops as lift threshold increases from 96.6% (lift≥2) to 69.1% (lift≥40).
- **Key insight:** Choosing lift≥3 preserves 85.9% recall while keeping only 25 rules (vs 30 at lift≥2). The "elbow" in the curve around lift=3–5 justifies the threshold choice.
- **Report section:** Phase 2 Methods / Threshold Selection

**`phase2_06_if_only_vs_rules_only.png`**
- **Content:** Side-by-side comparison of two fraud subgroups: "Caught by IF only" (5 cases) and "Caught by Rules only" (5,137 cases). Shows distributions of amount, time of day, and category.
- **Key insight:** IF-only fraud = low-amount ($19 median), all nighttime, health_fitness category — statistical outliers in transaction behavior. Rules-only fraud = higher-amount ($321 median), 86.5% night, mostly grocery/shopping categories.
- **Report section:** Phase 2 Results / Method Complementarity

### Phase 3 Figures (ML Models)

**`phase3_01_pr_curves.png`**
- **Content:** Precision-Recall curves for all 3 ML models overlaid on one chart. Three scatter points showing Phase 2 baselines. X-axis=Recall (0–1), Y-axis=Precision (0–1). XGBoost (blue) and RF (orange) curves dominate. The upper-left corner represents high precision at low recall; curves sweep toward high recall. Baseline points cluster in the lower region (low precision or low recall).
- **Key insight:** XGBoost AUC-PR=0.894 vs RF AUC-PR=0.859 vs any Phase 2 method (<0.11). ML models dramatically outperform rule-based and threshold approaches on the precision-recall tradeoff.
- **Report section:** Phase 3 Results / Model Evaluation

**`phase3_02_feature_importance.png`**
- **Content:** Side-by-side horizontal bar charts: RF Base feature importance (left) and RF Rule-Augmented feature importance (right). In both charts `amt` is the tallest bar. Right chart includes `flag_rules` and `flag_threshold` as additional bars.
- **Key insight:** `amt` dominates (~42% importance). `category_enc` and `is_night` are second and third. In the augmented model, `flag_rules` ranks 4th–5th with ~6–8% importance, confirming the rule flags capture some signal but don't dominate.
- **Report section:** Phase 3 Results / Feature Importance

**`phase3_03_all_methods_comparison.png`**
- **Content:** Grouped bar chart comparing all 7 methods on Precision, Recall, and F1. Three groups of 7 bars. The right side of the chart (ML models: RF Base, XGBoost Base, RF Aug, XGB Aug) clearly dominates the left side (Phase 2 baselines).
- **Key insight:** Clear tier structure: ML models (F1 0.78–0.80) >> simple threshold (F1 0.30) >> Isolation Forest (F1 0.22) >> association rules alone (F1 0.03). This is the headline result figure.
- **Report section:** Phase 3 Results / Full Comparison

**`phase3_04_confusion_matrix.png`**
- **Content:** 2×2 confusion matrix heatmap for XGBoost Base (best model). TN=1,839,348, FP=3,395, FN=1,290, TP=8,361. Color scale: dark blue for large cells, light for small. Annotated with counts and percentages.
- **Key insight:** 86.6% of fraud caught (8,361/9,651 TP). Only 3,395 false alarms out of 1,842,743 non-fraud (0.18% false alarm rate). 1,290 fraud transactions missed (13.4% FN).
- **Report section:** Phase 3 Results / Best Model Analysis

---

## SECTION 6 — KEY FINDINGS & INSIGHTS

### Finding 1: Amount is the Single Strongest Fraud Predictor

- Univariate AUC-ROC of `amt` = **0.834** — far above any other feature
- Fraud transactions have a median amount of **$390.00** vs **$47.24** for non-fraud (8.3× higher)
- 48.5% of all fraud transactions involve amounts > $500
- The `very_high` bin (>$500) has a **21.58% fraud rate** — 41× the 0.52% baseline
- The `medium` bin ($50–$200) has only 0.03% fraud rate — fraudsters deliberately avoid mid-range amounts (possibly to avoid common detection thresholds)

### Finding 2: Nighttime (22:00–06:00) is the Key Temporal Pattern

- Night fraud rate: **1.50%** vs daytime fraud rate: **0.10%** — a **15× multiplier**
- Worst single hour: **22:00 (10pm) = 2.60% fraud rate**
- `is_night` has univariate AUC = **0.782** — second strongest feature
- The nighttime pattern likely reflects: (a) reduced cardholder awareness of unauthorized charges, (b) fraudster operational patterns, (c) reduced real-time monitoring by issuing banks at night

### Finding 3: Online Shopping Categories Have Elevated Fraud Rates

- `shopping_net` (online retail): 1.59% fraud rate, 3× baseline
- `misc_net` (online miscellaneous): 1.30% fraud rate, 2.5× baseline
- `grocery_pos` (in-person grocery): 1.26% fraud rate, 2.4× baseline
- Physical categories (gas_transport, entertainment) have lower fraud rates (0.41%, ~0.3%)
- Online channels (`_net` suffix) are systematically riskier than point-of-sale (`_pos` suffix)

### Finding 4: The Strongest Fraud Signal is a Three-Way Interaction

The top rule is: **`amt>$500 AND category=shopping_net AND night(22–6)`**
- Fraud rate: **63.7%** (63 out of 100 matching transactions are fraud)
- Lift: **122.3×** — 122 times more likely to be fraud than baseline
- Covers 1,885 fraud transactions (19.5% of all fraud)
- This interaction cannot be detected by examining features independently — it requires joint analysis

### Finding 5: Distance Has Zero Discriminative Power

- `distance_km` (cardholder-to-merchant): AUC = **0.501** ≈ random
- Fraud median distance: 78.1km; Non-fraud median distance: 78.2km — essentially identical
- This is counterintuitive (one might expect "out-of-area" transactions to be suspicious) but reflects that the dataset spans a diverse geographic range where legitimate purchases often occur far from home
- Distance should be excluded from fraud detection features in this dataset

### Finding 6: Association Rules Have High Recall but Low Precision

- 25 rules with lift ≥ 3× catch **85.9%** of all fraud
- But they also flag **510,240 transactions** (27.5% of the entire dataset)
- Precision = **1.62%** — only 1 in 62 flagged transactions is actually fraud
- Standalone rules are operationally unusable as automated blockers (too many false alarms)
- Rules are most valuable as: (a) analyst education, (b) policy guidelines, (c) explanatory layer for ML decisions

### Finding 7: ML Models Massively Outperform All Rule-Based Approaches

| Tier | Best Method | AUC-PR |
|---|---|---|
| ML (Phase 3) | XGBoost Base | **0.894** |
| Rule-based (Phase 2) | Threshold amt>$500 | 0.107 |

- XGBoost achieves AUC-PR 8.3× higher than the best Phase 2 method
- XGBoost catches 8,361/9,651 (86.6%) of fraud with only 3,395 false alarms
- The simple threshold rule catches only 4,684 (48.5%) with 17,025 false alarms

### Finding 8: Rule Augmentation Does Not Improve XGBoost

- XGBoost Base AUC-PR: **0.8940**
- XGBoost Augmented (+ rule flags) AUC-PR: **0.8939**
- Difference: -0.0001 (negligible)
- Interpretation: XGBoost already learns the fraud patterns captured by the rules from the raw features — adding explicit rule flags provides no new information
- This contrasts with RF, where rule augmentation *decreases* performance (0.8593 → 0.8380), possibly because RF handles the interaction terms less efficiently

### Finding 9: Fraud Patterns Are Invisible in Individual Demographics

- Gender: Male 0.57% vs Female 0.48% — negligible difference
- City population: rural/suburban/urban all ~0.52% — non-discriminative
- Age alone: weak signal (senior 0.67% vs middle 0.45%)
- These features only gain predictive power in *combination* with high-signal features (amt, category, time)

### Finding 10: XGBoost is the Best Practical Model

- Highest AUC-PR (0.894) and highest Recall (86.6%) simultaneously
- Under business-weighted F-beta (β=1.53, favoring recall 7:3 over precision), XGBoost Base at threshold 0.9603 provides the optimal operating point
- At this threshold: catches 86.6% of fraud (8,361 cases), generates only 3,395 false alarms (0.18% false alarm rate on non-fraud)
- False alarms are low enough to be reviewed by a small analyst team; missed fraud (13.4%) represents cases with no detectable pattern in these 9 features

---

## SECTION 7 — CONCLUSIONS & TECHNICAL ENVIRONMENT

### 7.1 Summary of Conclusions

1. **Interpretable rules are viable for analyst-level fraud policy** but not for automated blocking. The 25 discovered rules, especially the top 4–5, represent concrete, actionable fraud policy: any transaction matching Rule 0 (shopping_net + amt>$500 + night) should trigger an immediate block.

2. **Machine learning substantially outperforms all rule-based and anomaly detection methods** on precision-recall tradeoff. XGBoost (AUC-PR=0.894) vs best Phase 2 method (AUC-PR=0.107) — an 8× improvement.

3. **Hybrid rule-augmented ML provides no improvement** over pure XGBoost. This suggests the ML model is sufficiently powerful to learn the interaction patterns from raw features without needing pre-encoded rule features.

4. **The optimal fraud detection system** for this dataset combines: (a) XGBoost scoring for automated real-time screening, (b) association rules for analyst education, policy design, and explanation of model decisions (interpretability layer), and (c) threshold tuned to business priorities (precision vs recall tradeoff).

5. **Key actionable insight for fraud prevention:** The combination of large transaction amount (>$500), nighttime processing (22:00–06:00), and online shopping categories (shopping_net, misc_net) is the highest-confidence fraud signal and should trigger automatic holds or blocks.

6. **Non-obvious negative results:** Distance between cardholder and merchant is not a fraud predictor in this dataset. City size is not predictive. Gender is not predictive. These should be excluded from production fraud models to avoid bias and reduce noise.

### 7.2 Limitations

1. **Simulated dataset:** Data generated by Sparkov simulation tool, which may not fully replicate the statistical properties of real-world fraud (e.g., card-not-present fraud, account takeover, synthetic identity fraud)
2. **No longitudinal features:** No velocity features (e.g., number of transactions in last hour/day), which are often the strongest real-world fraud indicators
3. **No cardholder history:** No features capturing deviation from cardholder's own historical behavior
4. **Static rules:** Association rules are mined once; in production, rules would need periodic re-mining as fraud patterns evolve
5. **Binary fraud label:** Real fraud is more granular (dispute type, chargebacks, merchant fraud vs. cardholder fraud)

### 7.3 Technical Environment

**Language:** Python 3.x

**Libraries and versions (from requirements.txt):**

| Library | Version | Use |
|---|---|---|
| pandas | ≥ 2.0.0 | Data manipulation, DataFrame operations |
| numpy | ≥ 1.24.0 | Numerical computation, Haversine formula |
| matplotlib | ≥ 3.7.0 | Static visualizations (all figures) |
| plotly | ≥ 5.15.0 | Interactive visualizations |
| scikit-learn | ≥ 1.3.0 | Isolation Forest, RF, LabelEncoder, cross_val_predict, metrics |
| mlxtend | ≥ 0.22.0 | FP-Growth algorithm, association_rules function |
| xgboost | (latest compatible) | XGBoost gradient boosting |
| nbformat | ≥ 5.9.0 | Notebook format support |
| nbconvert | ≥ 7.7.0 | Notebook conversion |
| ipykernel | ≥ 6.25.0 | Jupyter kernel |
| jupyter-core | ≥ 5.3.0 | Jupyter infrastructure |

**Random seeds used:** `random_state=42` for all stochastic operations (RF, XGBoost, train/test splits)

**Hardware context:** Consumer-grade Windows machine. FP-Growth on 9,651 fraud transactions took 4.4 seconds. RF full fit (300 trees on 1.85M rows) took ~213 seconds. XGBoost CV took longer than RF CV due to gradient boosting iterations.

**Operating system:** Windows 11
**Development environment:** Jupyter Notebook

### 7.4 Project File Structure

```
Fraud-Detection/
├── data/
│   ├── raw/
│   │   ├── fraudTrain.csv       # 1,296,675 rows × 23 cols
│   │   └── fraudTest.csv        # 555,719 rows × 23 cols
│   └── processed/
│       ├── transactions_with_features.csv  # Phase 1 output: 1,852,394 × 31
│       ├── transactions_phase2.csv         # Phase 2 output: 1,852,394 × 38
│       └── transactions_phase3.csv         # Phase 3 output: 1,852,394 × 48
├── notebooks/
│   ├── phase1_eda.ipynb              # EDA + feature engineering
│   ├── phase2_pattern_mining.ipynb   # Rule mining + baselines
│   └── phase3_detection_model.ipynb  # ML model training + evaluation
├── outputs/
│   ├── rules/
│   │   ├── fraud_rules_final.csv          # 25 association rules
│   │   ├── method_comparison.csv          # Phase 2 3-method comparison
│   │   └── phase3_method_comparison.csv   # Full 7-method comparison
│   └── figures/
│       └── [31 figure files — see Section 5]
├── CLAUDE.md / claude.md    # Project guidance
├── requirements.txt
└── venv/                    # Python virtual environment
```

### 7.5 Column Additions by Phase

**After Phase 1 (31 columns total from 23 original):**
Original 23 + hour_of_day, day_of_week, month, is_night, distance_km, age, amt_bin, time_bin (8 derived)

**After Phase 2 (38 columns total):**
+ flag_threshold, flag_iso_forest, flag_rules, [and rule match columns] (7 new)

**After Phase 3 (48 columns total):**
+ prob_rf_base, prob_xgb_base, prob_rf_aug, prob_xgb_aug (4 probability scores)
+ pred_rf_base, pred_xgb_base, pred_rf_aug, pred_xgb_aug (4 binary predictions)
+ [additional meta columns] (2)

---

*End of project_context.md — Total coverage: 3 phases, 7 methods, 25 rules (complete), all 31 figures, all numeric metrics.*
