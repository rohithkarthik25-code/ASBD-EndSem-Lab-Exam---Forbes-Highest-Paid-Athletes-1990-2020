# DATASET ACQUISITION, INSPECTION & SOLUTION PLANNING REPORT
**Hackathon / Practical Exam — Forbes Highest-Paid Athletes (1990–2020)**
**Date & Time:** November 21, 2026 | **Workspace:** `hackathon_live/`

---

## 1. Executive Summary & Problem Understanding

The objective defined in `problem_statement_updated.pdf` requires an end-to-end analytical pipeline applied to the **Forbes Highest-Paid Athletes (1990–2020)** dataset:
1. **Data Preprocessing:** Cleaning noisy data, feature selection, and data transformation.
2. **Descriptive Analytics & Visualizations:** At least 10 unique, course-aligned plots with detailed interpretations.
3. **Predictive Analytics:**
   - (i) Association Rule Mining using the **DIC (Dynamic Itemset Counting)** algorithm.
   - (ii) **K-Nearest Neighbors (KNN)** classification / **DIANA (Divisive Analysis)** hierarchical clustering.
   - (iii) Closed Association Rules (CLICS) using the **COFI (Co-occurrence Frequent Itemset)** algorithm.
4. **PySpark Big Data Test Drive:** Implementing two predictive analytics techniques in PySpark, establishing performance enhancement and scalability compared to single-threaded Python.
5. **Deliverables:** Final report (PDF) and commented source code.

---

## 2. Dataset Acquisition & Storage

- **Source:** Kaggle (`parulpandey/forbes-highest-paid-athletes-19902019`)
- **Acquisition Tool:** `kagglehub.dataset_download`
- **Stored File Paths:**
  - Original: [`hackathon_live/data/Forbes Richest Atheletes (Forbes Richest Athletes 1990-2020).csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/data/Forbes%20Richest%20Atheletes%20(Forbes%20Richest%20Athletes%201990-2020).csv) (14,230 bytes)
  - Standardized Copy: [`hackathon_live/data/forbes_athletes.csv`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/data/forbes_athletes.csv) (14,230 bytes)

---

## 3. Exhaustive Dataset Profile & Inspection Findings

### 3.1 Dimensions & Schema
- **Shape:** 301 rows $\times$ 8 columns
- **Duplicate Rows:** 0 (zero identical rows)

| Column Name | Inferred Type | Python Dtype | Missing Count | Missing % | Unique Count | Sample Values |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `S.NO` | Identifier | `int64` | 0 | 0.0% | 301 | 1, 2, 3, 4, 5 |
| `Name` | Categorical | `str / object` | 0 | 0.0% | 82 | Michael Jordan, Tiger Woods, Kobe Bryant |
| `Nationality` | Categorical | `str / object` | 0 | 0.0% | 22 | USA, UK, Germany, Switzerland, Portugal |
| `Current Rank` | Discrete Ordinal | `int64` | 0 | 0.0% | 10 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `Previous Year Rank`| Mixed / Ordinal | `str / object` | 24 | 7.97% | 36 | `1`, `>10`, `>40`, `?`, `none`, `not ranked` |
| `Sport` | Categorical | `str / object` | 0 | 0.0% | 29 | Basketball, boxing, Golf, F1 racing, NBA |
| `Year` | Temporal | `int64` | 0 | 0.0% | 30 | 1990, 1991 ... 2020 (Excluding 2001) |
| `earnings ($ million)`| Continuous Numeric | `float64` | 0 | 0.0% | 198 | 8.1, 28.6, 69.0, 115.0, 300.0 |

### 3.2 Year Coverage & Anomalies
- **Time Span:** 1990 to 2020 (31 calendar years).
- **Missing Year Anomaly:** **Year 2001 is completely absent** from Forbes publications.
- **Year 2002 Anomaly:** Year 2002 contains **11 records** instead of 10. Both Jacques Villeneuve and Jeff Gordon tied for Rank 10 with exactly \$20.0M earnings. All other 29 years contain exactly 10 records ($29 \times 10 + 11 = 301$ rows).

### 3.3 Numerical Distribution of Earnings (`earnings ($ million)`)
- **Count:** 301
- **Mean:** \$45.52M | **Standard Deviation:** \$33.53M
- **Min:** \$8.10M (Evander Holyfield / Arnold Palmer, 1990)
- **25th Percentile (Q1):** \$24.00M
- **50th Percentile (Median):** \$39.00M
- **75th Percentile (Q3):** \$59.40M
- **90th Percentile:** \$85.50M | **95th Percentile:** \$100.00M | **99th Percentile:** \$127.00M
- **Max:** \$300.00M (Floyd Mayweather, 2015)
- **Distribution Shape:**
  - **Skewness:** `+3.10` (Strong positive/right skew; vast majority earn \$15M–\$50M, with extreme multi-hundred-million outliers).
  - **Kurtosis:** `+18.62` (Extreme leptokurtic distribution with heavy tails).

### 3.4 Categorical Values & Data Quality Issues
1. **`Sport` (29 raw categories $\to$ 11 true categories):**
   - High inconsistency due to casing (`basketball` vs `Basketball`, `boxing` vs `Boxing`, `golf` vs `Golf`, `soccer` vs `Soccer`, `tennis` vs `Tennis`).
   - Synonyms and sub-disciplines: `F1 racing`, `F1 Motorsports`, `Auto Racing`, `Auto racing`, `Auto Racing (Nascar)`, `NASCAR`, and `motorcycle gp` all belong to **Auto Racing**.
   - League abbreviations: `NBA` is Basketball; `NFL` is American Football.
   - Multi-sport entries: `American Football / Baseball` (Deion Sanders).
   - *Cleaned Distribution:* Basketball (82), Boxing (46), Golf (44), Auto Racing (40), Soccer (33), Tennis (23), American Football (21), Baseball (6), Hockey (4), Cycling (1), MMA (1).
2. **`Nationality` (22 raw $\to$ 21 standardized):**
   - Demonym vs Country mismatch: `Filipino` (Manny Pacquiao) vs `Philippines`.
   - Strong class imbalance: **USA represents 68.4% (206 / 301 rows)**, followed by UK (13), Germany (13), Switzerland (12), Portugal (10), Brazil (9), Argentina (9).
3. **`Previous Year Rank`:**
   - 24 missing (`NaN`) values representing first-time entrants or athletes not ranked.
   - Text values: `not ranked` (4), `none` (1), `?` (3), `??` (2).
   - Range markers: `>10` (36), `>14` (1), `>20` (4), `>30` (2), `>40` (6), `>100` (6).
   - Pure numeric strings: `1` to `40`.

---

## 4. Attribute Mapping for Analytic Tasks

| Task | Primary Attributes | Role & Transformation Needed |
| :--- | :--- | :--- |
| **Descriptive Analytics** | `earnings`, `Current Rank`, `Year`, `Sport`, `Nationality` | Direct aggregation, career totals, inflation adjustments, distributions. |
| **Data Visualizations** | All attributes | 10+ plots capturing temporal evolution, distributions, correlations, and clusters. |
| **Association Rule Mining (DIC / COFI)** | Discretized `earnings`, `Sport`, `Nationality`, `Era`, `Rank_Tier`, `Rank_Trajectory` | Transaction transformation (one basket per row or per athlete career). |
| **KNN Classification** | Features: `earnings`, `Year`, `Current Rank`, `Rank_Diff`, `Nation_US` $\to$ Target: `Sport_Group` or `Rank_Tier` | Normalization/StandardScaler, distance metrics (Euclidean, Manhattan). |
| **DIANA Clustering** | Continuous features (`earnings`, `Year`, `Current Rank`) or Aggregated Athlete Profiles | Standardization, Divisive top-down hierarchical splitting, dendrogram. |

---

## 5. Converting Continuous Attributes into Transactions (for DIC & COFI)

Both **DIC** (Dynamic Itemset Counting) and **COFI** (Co-occurrence Frequent Itemset trees) require categorical market-basket transactions $\{i_1, i_2, \dots, i_m\}$. Continuous attributes cannot be mined directly without discretization.

### 5.1 Discretization Strategy
1. **Earnings (`earnings ($ million)`):**
   - Domain/Quantile Bins:
     - `Earnings_Moderate`: $< \$25M$ (Bottom 25% of Top-10)
     - `Earnings_High`: $\$25M - \$60M$ (Interquartile Top-10 range)
     - `Earnings_Ultra`: $> \$60M$ (Top 25% elite megastars)
2. **Temporal Era (`Year`):**
   - `Era_1990s`: $1990 - 1999$ (Jordan, Tyson, Senna era)
   - `Era_2000s`: $2000 - 2009$ (Tiger Woods, Schumacher era)
   - `Era_2010s_2020`: $2010 - 2020$ (Mayweather, Ronaldo, Messi, LeBron era)
3. **Current Rank (`Current Rank`):**
   - `Rank_Podium`: Ranks 1 to 3
   - `Rank_MidTop10`: Ranks 4 to 7
   - `Rank_LowerTop10`: Ranks 8 to 10
4. **Rank Trajectory (`Previous Year Rank` vs `Current Rank`):**
   - `Trajectory_Improved`: Current Rank < Previous Year Rank
   - `Trajectory_Maintained`: Current Rank == Previous Year Rank
   - `Trajectory_Dropped`: Current Rank > Previous Year Rank
   - `Trajectory_NewEntry`: Unranked or Previous Rank `>10`
5. **Nationality Grouping (`Nationality`):**
   - `Nation_USA` vs `Nation_International` (mitigates high-cardinality noise).

### 5.2 Transaction Formulation
Each Forbes ranking row becomes a basket of discrete items:
$$\text{Transaction}_i = \{\text{Sport: Basketball}, \text{Nation: USA}, \text{Earnings: High}, \text{Era: 1990s}, \text{Rank: Podium}\}$$
This enables DIC and COFI to discover rules such as:
$$\{\text{Sport: Boxing}, \text{Rank: Podium}\} \implies \{\text{Earnings: Ultra}\} \quad (\text{conf}=0.85, \text{lift}=2.4)$$
$$\{\text{Sport: Soccer}\} \implies \{\text{Nation: International}, \text{Era: 2010s_2020}\}$$

---

## 6. Recommended 10+ Meaningful Visualizations

| # | Visualization Type | Variables / Features Plotted | Analytical Purpose | Expected Inference |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Histogram with KDE** | `earnings ($ million)` | Examine earnings distribution shape and skewness. | Severe right-skew with peak at \$20M–\$40M and long tail up to \$300M (log-normal). |
| **2** | **Categorical Boxplot** | `Sport` (Cleaned) vs `earnings` | Compare median earnings and detect sport-specific outliers. | Boxing has the highest outliers (Mayweather); Basketball has the highest consistent volume. |
| **3** | **Time-Series Line Plot** | `Year` vs Mean, Median, and Max `earnings` | Track historical wage inflation of top athletes across 30 years. | Sharp exponential rise post-2005; massive wage spikes in 2015 and 2018 (mega-fights). |
| **4** | **Horizontal Bar Chart** | Athlete `Name` vs Cumulative Career Earnings | Identify all-time highest-earning athletes in Forbes records. | Tiger Woods leads cumulative earnings (\$1.37B+), followed by LeBron, Mayweather, Ronaldo. |
| **5** | **100% Stacked Area Chart** | `Year` / Decades vs `Sport` Market Share | Visualize shifting sport dominance over 3 decades. | 1990s dominated by Boxing/Auto Racing; 2010s dominated by Soccer and Basketball. |
| **6** | **Heatmap / Correlation Matrix** | `Year`, `Current Rank`, `earnings`, `Rank_Diff`, `Nation_Encoded` | Quantify linear and monotonic relationships between attributes. | Negative correlation between Current Rank and earnings; positive trend between Year and earnings. |
| **7** | **Scatter Plot with Regression Line** | `Current Rank` vs `earnings` (by Era) | Analyze pay gap between Rank 1 and Rank 10 across eras. | Rank 1 commands an exponential premium over Ranks 2–10, expanding significantly over time. |
| **8** | **Violin Plot** | Decades (1990s, 2000s, 2010s) vs `earnings` | Show multi-modal density and spread of earnings across eras. | 1990s density was tightly clustered (\$10M–\$30M); 2010s shows high dispersion (\$30M–\$300M). |
| **9** | **Donut / Treemap Chart** | `Nationality` Breakdown (USA vs Rest of World) | Assess globalization of elite sports wealth. | USA holds ~68% of list positions, but European soccer stars rapidly captured top ranks in 2010s. |
| **10** | **DIANA Dendrogram** | Top 25 Multi-Year Athletes (Career Earnings, Appearances, Peak Rank) | Visualize hierarchical divisive clustering of athlete dominance tiers. | Clean separation of All-Time Icons (Tiger, Jordan, Floyd) from short-tenured elite athletes. |
| **11** | **Bubble Chart (Bonus)** | X=`Year`, Y=`earnings`, Size=`Appearances`, Color=`Sport` | Multi-dimensional overview of individual athlete points over time. | Highlights longevity vs transient peak earnings across different sporting careers. |
| **12** | **Pairplot (Bonus)** | Numerical features across Cleaned Sports | Multi-feature scatter and distribution matrix. | Visual validation of feature separability for subsequent KNN classification. |

---

## 7. PySpark Implementation Selection

### Selected Techniques for PySpark:
1. **Technique 1: Parallel K-Nearest Neighbors (KNN) Distance Engine**
2. **Technique 2: Distributed Frequent Itemset & Association Rule Mining (DIC / Parallel Itemset Counting / FP-Growth)**

### Rationale:
- **KNN in PySpark:** Standard KNN has $O(N \cdot M \cdot d)$ computational complexity. By distributing test queries across Spark executors and broadcasting the training matrix (or using Spark RDD `cartesian` / DataFrame cross-joins with vectorized UDFs), distance calculations and nearest-neighbor sorting scale horizontally across CPU cores.
- **Association Mining in PySpark:** Frequent itemset mining exhibits exponential combinatorial complexity $O(2^k)$. DIC’s step-based counting intervals and candidate pruning map directly to PySpark’s distributed MapReduce paradigm (`mapPartitions`, `reduceByKey`, and distributed prefix trees via `pyspark.ml.fpm.FPGrowth`).

---

## 8. Strategy for Fair Python vs. PySpark Benchmarking

### The "Small Data Overhead" Challenge:
- The Forbes dataset has only **301 rows (~14 KB)**.
- Single-threaded Python (vectorized NumPy / Pandas) processes 301 rows in **$\approx 2$ to $5$ milliseconds**.
- PySpark incurs JVM initialization, task serialization (Py4J), network socket setup, and Catalyst query planning, taking **$\approx 1,500$ to $3,500$ milliseconds**.
- Simply comparing execution time on 301 rows would yield a misleading result ("Spark is 500x slower"), failing to reflect big data realities.

### Proposed Fair Benchmarking Framework:
1. **Multi-Scale Benchmark (Inflection Point Analysis):**
   - Benchmark both systems across 4 scale levels:
     - Level 1: $N = 301$ (Original dataset)
     - Level 2: $N = 10,000$ rows (Bootstrap replicated)
     - Level 3: $N = 100,000$ rows
     - Level 4: $N = 1,000,000$ rows (Big data scale)
2. **Key Metrics to Measure:**
   - **Warm Execution Latency:** Measure steady-state computation excluding SparkSession boot time.
   - **Throughput:** Records processed per second ($\text{rows} / \text{sec}$).
   - **Scaling Curve:** Execution time vs $N$ to prove the exact **crossover point** where Spark overtakes Python.
   - **Resource Utilization:** Single-core CPU bottleneck (Python) vs Multi-core / Executor saturation (PySpark).
   - **Memory Stability:** Peak RAM consumption under scaled loads.

---

## 9. Dataset Limitations, Assumptions & Algorithm Caveats

1. **Small Sample Size ($N = 301$):**
   - High risk of overfitting with complex models. Regularization and cross-validation are strictly required.
   - Large-$k$ KNN ($k > 15$) is inappropriate given small per-sport sample sizes.
2. **Nominal vs. Real USD (Inflation Distortion):**
   - A \$28.6M paycheck in 1990 has equivalent purchasing power to $\approx \$56M$ in 2020. Comparing raw nominal dollars over a 30-year span introduces temporal distortion unless an inflation adjustment index is noted or era-stratified analysis is used.
3. **Severe Class Imbalance:**
   - USA accounts for 68.4% of rows; Basketball accounts for 27.2%. Algorithms must use stratified splits and macro-averaged metrics.
4. **Data Discretization Sensitivity:**
   - In DIC and COFI, support and confidence thresholds must be chosen carefully (e.g., $\text{min\_support} \in [0.05, 0.10]$, $\text{min\_confidence} \in [0.5, 0.7]$) to prevent either empty rule sets or combinatorial rule flooding.
5. **Unsuitable Algorithms:**
   - Deep Neural Networks, Large GBDTs with deep trees, or density clustering with high min-points (DBSCAN min_samples > 10).

---

## 10. Proposed Next Steps (Upon User Approval)

- **Phase A:** Execute complete Data Cleaning and Preprocessing (Sport taxonomy standardization, Nationality unification, Missing rank imputation/binning).
- **Phase B:** Generate and export all 10+ high-resolution visualizations to `hackathon_live/results/`.
- **Phase C:** Implement DIC and COFI Association Rule Mining on discretized transactions.
- **Phase D:** Implement KNN Classification and DIANA Hierarchical Clustering (with dendrogram).
- **Phase E:** Implement PySpark distributed versions of KNN and ARM, executing the multi-scale performance benchmark.
- **Phase F:** Compile the comprehensive Final Submission Report (`hackathon_live/reports/FINAL_SUBMISSION_REPORT.md` / PDF).
