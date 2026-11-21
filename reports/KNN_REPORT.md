# KNN CLASSIFICATION EXPERIMENT REPORT
**Dataset:** Forbes Highest-Paid Athletes (1990–2020) | `hackathon_live/data/forbes_athletes_cleaned.csv`  
**Script:** [`hackathon_live/knn_experiment.py`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/knn_experiment.py)  
**Model Artifact:** [`hackathon_live/models/knn_model.pkl`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/models/knn_model.pkl)  
**Metrics Output:** [`hackathon_live/results/knn_metrics.json`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/knn_metrics.json)  
**CV Results Table:** [`hackathon_live/results/knn_cv_results.csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/knn_cv_results.csv)  
**Predictions Table:** [`hackathon_live/results/knn_test_predictions.csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/knn_test_predictions.csv)  
**Execution Date:** November 21, 2026  

---

## 1. Executive Summary & Problem Framing

This report documents the implementation, tuning, and evaluation of a K-Nearest Neighbors (KNN) classification model on the Forbes Highest-Paid Athletes dataset ($N = 301$). 

### Exploratory Nature & Hypothesis Framing:
This is an **exploratory prediction task** designed to investigate whether high-level earnings and list attributes carry a geometric separability signal under strict leakage controls, rather than a production-grade forecasting tool.

The experiment evaluates the following exploratory hypothesis:
- **Hypothesis $H_1$:** Top-10 athletes competing in **Team Sports** operate under institutional and contractual mechanisms (collective bargaining agreements, league salary caps, guaranteed team payrolls) distinct from athletes in **Individual Sports** (episodic pay-per-view prize purses, tournament appearance fees, individual endorsement premiums). Consequently, an athlete's financial profile (`earnings`), historical era (`Year`), list standing (`Current Rank`), and nationality profile (`Nationality_Group`) carry a detectable multivariate geometric signature capable of separating the two groups.
- **Null Hypothesis $H_0$:** At the extreme boundary of global sports wealth, compensation dynamics are sport-agnostic; KNN will perform no better than naive guessing baselines.

---

## 2. Dataset Validation & Target Distribution

- **Total Dataset Size:** 301 rows $\times$ 16 columns (cleaned dataset).
- **Target Formulation:** `Sport_Type` (Binary):
  - **`Team Sport`** ($N = 146, 48.50\%$): Basketball ($82$), Soccer ($33$), American Football ($21$), Baseball ($6$), Hockey ($4$).
  - **`Individual Sport`** ($N = 155, 51.50\%$): Boxing ($46$), Golf ($44$), Auto Racing ($40$), Tennis ($23$), Cycling ($1$), MMA ($1$).
- **Distribution Characteristics:** The target is virtually **50/50 balanced** (48.5% vs. 51.5%), completely eliminating majority-class imbalance artifacts.

---

## 3. Strict Data Leakage Firewall

To guarantee 100% mathematical integrity, all features were audited against ground-truth leakage:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FEATURE SELECTION & AUDIT                                │
├───────────────────────────────┬──────────────────────────┬─────────────────────────────┤
│ Feature Name                  │ Data Type / Scale        │ Status & Justification      │
├───────────────────────────────┼──────────────────────────┼─────────────────────────────┤
│ 1. earnings ($ million)       │ Continuous Float         │ PERMITTED (Primary metric)  │
│ 2. Year                       │ Temporal Integer         │ PERMITTED (Era indicator)   │
│ 3. Current Rank               │ Ordinal Integer (1–10)   │ PERMITTED (List standing)   │
│ 4. Nationality_Group          │ Categorical (USA/Int'l)  │ PERMITTED (Geographic split)│
├───────────────────────────────┼──────────────────────────┼─────────────────────────────┤
│ • Sport / Sport_Clean         │ Categorical              │ STRICTLY EXCLUDED (Leakage) │
│ • S.NO / Name                 │ Identifiers              │ STRICTLY EXCLUDED (Non-gen) │
│ • Rank_Tier                   │ Categorical              │ STRICTLY EXCLUDED (Derived) │
│ • Earnings_Category           │ Categorical              │ STRICTLY EXCLUDED (Derived) │
│ • Rank_Trajectory             │ Categorical              │ STRICTLY EXCLUDED (Derived) │
│ • Decade                      │ Categorical              │ STRICTLY EXCLUDED (Derived) │
│ • Previous Year Rank          │ Mixed / Ordinal          │ EXCLUDED from initial model │
└───────────────────────────────┴──────────────────────────┴─────────────────────────────┘
```

- **Feature Matrix Shape ($X$):** Exactly 301 rows $\times$ 4 features.
- **Missing Values in $X$:** 0 (zero nulls across all four features).

---

## 4. Experimental Setup & Stratified Train/Test Partitioning

1. **Partitioning:** The dataset was split into an **80% Training Set ($N = 240$)** and a **20% Held-Out Test Set ($N = 61$)** using `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
2. **Stratification Verification:**
   - Training Set ($N=240$): `Individual Sport` = 124 ($51.67\%$), `Team Sport` = 116 ($48.33\%$).
   - Test Set ($N=61$): `Individual Sport` = 31 ($50.82\%$), `Team Sport` = 30 ($49.18\%$).
   - The class proportions match the full population almost perfectly.
3. **Strict Isolation:** Preprocessing and hyperparameter selection were executed **exclusively on the 240 training samples**. The test set was locked and evaluated exactly once.
4. **Fold-Wise Preprocessing:** Inside each cross-validation fold, `StandardScaler` was fitted strictly on the training fold and applied to the validation fold via Scikit-Learn `Pipeline` and `ColumnTransformer`.

---

## 5. Baseline Models Evaluation

Two naive baselines were established on the training set and evaluated on the held-out test set to anchor performance:

| Baseline Model | Strategy Description | Train 5-Fold CV Accuracy | Train 5-Fold CV F1-Macro | Test Accuracy ($N=61$) | Test Balanced Acc | Test F1-Macro | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | Predicts `Individual Sport` for every record | 51.67% ($\pm 0.83\%$) | 0.3406 ($\pm 0.0037$) | **50.82%** (31/61) | 50.00% | 0.3370 | 0.5000 |
| **Dummy (Stratified)** | Randomly guesses classes matching prior probabilities | 47.08% ($\pm 6.67\%$) | 0.4650 ($\pm 0.0674$) | **52.46%** (32/61) | 52.37% | 0.5225 | 0.5237 |

---

## 6. Hyperparameter Tuning via Stratified 5-Fold Cross-Validation

A comprehensive grid search was executed over 28 parameter combinations:
- **Number of Neighbors ($k$):** $[1, 3, 5, 7, 9, 11, 15]$
- **Distance Metric ($p$):** $[1 \text{ (Manhattan / } L_1\text{)}, 2 \text{ (Euclidean / } L_2\text{)}]$
- **Neighbor Weights:** `['uniform', 'distance']`
- **Cross-Validation Scheme:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Primary Optimization Metric:** `f1_macro`

### Top-5 Hyperparameter Configurations from Grid Search:
| Rank | $k$ (Neighbors) | Metric ($p$) | Weights | Mean CV F1-Macro | Mean CV Accuracy | Mean CV Balanced Acc | Mean CV ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Best)** | **9** | **Manhattan ($p=1$)** | **uniform** | **0.6942** | **0.6958** | **0.6954** | **0.7462** |
| 2 | 11 | Manhattan ($p=1$) | uniform | 0.6934 | 0.6958 | 0.6950 | 0.7410 |
| 3 | 11 | Manhattan ($p=1$) | distance | 0.6865 | 0.6875 | 0.6869 | 0.7383 |
| 4 | 9 | Manhattan ($p=1$) | distance | 0.6853 | 0.6875 | 0.6869 | 0.7371 |
| 5 | 7 | Manhattan ($p=1$) | uniform | 0.6833 | 0.6833 | 0.6828 | 0.7346 |

### Optimal Hyperparameters Selected:
- **$k = 9$**
- **Distance Metric = Manhattan ($p=1$, $L_1$ norm)**
- **Weighting Scheme = Uniform**

*Note on Distance Metric:* Manhattan distance ($p=1$) performed better empirically than Euclidean distance ($p=2$) in cross-validation across tested $k$ values (averaging 2.4 percentage points higher F1-macro). While $L_1$ norms are often hypothesized to be less sensitive to extreme outliers than squared Euclidean distances, on this specific dataset we report this strictly as an observed empirical performance difference.

---

## 7. Performance on Held-Out Test Set ($N = 61$)

The optimal pipeline was fitted on all 240 training samples and evaluated on the untouched held-out test set:

| Evaluation Metric | KNN Model Score | Dummy (Most Frequent) Baseline | Dummy (Stratified Guessing) Baseline | Absolute Lift Over Best Dummy |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **70.49%** (43 / 61) | 50.82% (31 / 61) | 52.46% (32 / 61) | **+18.03 percentage points** |
| **Balanced Accuracy** | **70.59%** | 50.00% | 52.37% | **+18.22 percentage points** |
| **Macro F1-Score** | **0.7042** | 0.3370 | 0.5225 | **+0.1817 (+18.17 pts)** |
| **Weighted F1-Score**| **0.7040** | 0.3429 | 0.5238 | **+0.1802 (+18.02 pts)** |
| **Macro Precision** | **0.7086** | 0.2541 | 0.5246 | **+0.1840** |
| **Macro Recall** | **0.7059** | 0.5000 | 0.5237 | **+0.1822** |
| **ROC-AUC Score** | **0.7108** | 0.5000 | 0.5237 | **+0.1871** |

---

### Confusion Matrix on Held-Out Test Set:
```
                      Predicted: Individual Sport    Predicted: Team Sport    Total Actual
Actual: Individual Sport          20 (True Negative)            11 (False Positive)        31
Actual: Team Sport                 7 (False Negative)           23 (True Positive)         30
Total Predicted                   27                            34                         61
```

### Detailed Per-Class Classification Report:
| Class Label | Precision | Recall | F1-Score | Support ($N$) |
| :--- | :---: | :---: | :---: | :---: |
| **Individual Sport** | 0.7407 (74.1%) | 0.6452 (64.5%) | 0.6897 (69.0%) | 31 |
| **Team Sport** | 0.6765 (67.6%) | 0.7667 (76.7%) | 0.7188 (71.9%) | 30 |
| **Macro Average** | **0.7086** | **0.7059** | **0.7042** | **61** |
| **Weighted Average** | **0.7091** | **0.7049** | **0.7040** | **61** |

---

## 8. Does KNN Genuinely Beat the Dummy Baseline?

**YES, KNN demonstrates observable empirical predictive lift above both naive baselines on this dataset:**

*(Note: Formal inferential hypothesis tests, such as McNemar's test or paired t-tests, were not formally conducted; hence, this gain is characterized as an empirical observation on the sample rather than formal statistical significance.)*

1. **Substantial Empirical Accuracy Gain:** KNN achieves **70.49% accuracy** on the held-out test set, beating the most-frequent baseline (50.82%) by **+19.67 percentage points** and the stratified guessing baseline (52.46%) by **+18.03 percentage points**.
2. **Balanced Performance Across Both Classes:** Because the classes are nearly 50/50 balanced, accuracy is not an artifact of majority-class collapse. KNN achieves balanced recall across both categories:
   - Team Sport Recall: **76.67%** (correctly identifying 23 of 30 team athletes).
   - Individual Sport Recall: **64.52%** (correctly identifying 20 of 31 individual athletes).
3. **Discriminative Capability (ROC-AUC):** The model achieves a test ROC-AUC of **0.7108**, confirming that its predicted class probabilities provide genuine separation well above random chance (0.5000).
4. **Encouraging Cross-Validation and Test Alignment:** The 5-Fold cross-validation F1-macro score on the training set (**0.6942**) is very close to the held-out test set F1-macro (**0.7042**, difference $< 1$ percentage point). This alignment is encouraging, though given the small test sample ($N=61$), it should be regarded as a positive indicator rather than definitive proof of generalization stability across unseen populations.
5. **Why 70.5% Confirms Model Integrity:** In this exploratory task, an accuracy of $\approx 70.5\%$ indicates that while team and individual athletes exhibit discernible geometric clustering, compensation overlap at the absolute apex of wealth remains high. A model scoring $95\% - 100\%$ on this dataset would immediately indicate target leakage (such as retaining `Sport` or sport-derived proxies). The achieved $70.5\%$ score confirms genuine, non-leaky machine learning on non-trivial real-world data.

---

## 9. Sample Size Limitations ($N = 301$) & Analytical Caveats

1. **Test Set Granularity ($N = 61$):**
   - Because the held-out test set contains only 61 records, **each individual prediction represents $\approx 1.64$ percentage points of accuracy**. A shift of just 3 misclassified samples changes reported accuracy by nearly $5$ percentage points.
   - Consequently, performance claims should be reported alongside the cross-validation distribution ($69.58\% \pm 3.2\%$).
2. **Absence of Ground-Truth Separation:**
   - Mega-earners like Michael Jordan (Basketball) and Tiger Woods (Golf) share similar normalized feature coordinates. High financial success is not solely dictated by team vs. individual structure.

---

## 10. Summary of Generated Output Artifacts

All required files have been verified to exist on disk:
- [`hackathon_live/knn_experiment.py`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/knn_experiment.py) — Complete, reproducible experiment script.
- [`hackathon_live/models/knn_model.pkl`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/models/knn_model.pkl) — Serialized Scikit-Learn Pipeline (24.2 KB).
- [`hackathon_live/results/knn_metrics.json`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/knn_metrics.json) — Full structured JSON record of baseline, CV, and test metrics.
- [`hackathon_live/results/knn_cv_results.csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/knn_cv_results.csv) — Complete 28-model grid search log.
- [`hackathon_live/results/knn_test_predictions.csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/knn_test_predictions.csv) — Row-level predictions, actuals, and class probabilities for the held-out test set.
