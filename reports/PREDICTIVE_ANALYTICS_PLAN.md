# PREDICTIVE ANALYTICS ARCHITECTURAL & EXPERIMENTAL PLAN (REVISED)
**Dataset:** Forbes Highest-Paid Athletes (1990–2020) | `hackathon_live/data/forbes_athletes_cleaned.csv`  
**Problem Statement Reference:** [`hackathon_live/problem_statement_updated.pdf`](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/problem_statement_updated.pdf)  
**Output Document:** `hackathon_live/reports/PREDICTIVE_ANALYTICS_PLAN.md`  
**Revision Date:** November 21, 2026  

---

## 1. Executive Summary & Analytical Scope

The hackathon problem statement mandates four specific predictive and pattern-mining techniques over the cleaned Forbes Highest-Paid Athletes dataset ($N = 301$ rows):
1. **Dynamic Itemset Counting (DIC)** for Association Rule Mining (ARM).
2. **K-Nearest Neighbors (KNN)** Classification.
3. **Divisive Analysis (DIANA)** Hierarchical Clustering.
4. **Co-occurrence Frequent Itemsets (COFI)** for Closed Association Rules (CLICS).
5. **PySpark Big Data Test Drive:** Implementation of two predictive techniques on the PySpark platform with empirical scalability and performance benchmarking.

This revised plan establishes the formal mathematical, data-engineering, and experimental design for all four algorithms before writing code. It enforces strict data-leakage controls, frames KNN as a formal hypothesis evaluation, details the exact DIC state machine and COFI closed-rule mechanisms, specifies the true DIANA divisive procedure, and defines a scientifically sound Python-vs-PySpark benchmarking framework without pre-judging performance.

---

## 2. Core Differences Across Analytic Paradigms

To prevent conflation across machine learning paradigms, the following distinctions are formally enforced:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               ANALYTICAL PARADIGMS                                      │
├─────────────────────┬──────────────────────────┬───────────────────────────────────────┤
│ Paradigm            │ Mathematical Goal        │ Target Variable Requirement           │
├─────────────────────┼──────────────────────────┼───────────────────────────────────────┤
│ Association Mining  │ Find co-occurrence rules │ No target; unsupervised itemsets      │
│ (DIC & COFI)        │ X => Y with sup/conf/lift│ (transactions of discrete tokens)     │
├─────────────────────┼──────────────────────────┼───────────────────────────────────────┤
│ Classification      │ Map feature vector X to  │ Explicit categorical label Y;         │
│ (KNN)               │ a discrete class label Y │ trained on train set; evaluated on    │
│                     │ via distance metrics     │ untouched test set                    │
├─────────────────────┼──────────────────────────┼───────────────────────────────────────┤
│ Clustering          │ Partition N objects into │ No target; unsupervised distance      │
│ (DIANA)             │ hierarchy based on       │ matrix dissimilarity minimization     │
│                     │ pairwise dissimilarity   │                                       │
└─────────────────────┴──────────────────────────┴───────────────────────────────────────┘
```

---

## 3. Deep-Dive Design for the Four Required Techniques

---

### Technique 1: Dynamic Itemset Counting (DIC) Association Rule Mining

#### 1.1 Objective & Suitability
- **Objective:** Discover frequent itemset combinations and high-confidence association rules characterizing the commercial profiles of elite athletes across eras, sports, compensation brackets, and performance tiers.
- **Why DIC (Not Apriori or FP-Growth):**
  - Standard Apriori requires $k$ full sequential passes over the database to find itemsets of size $k$ (counting 1-itemsets in pass 1, 2-itemsets in pass 2, etc.).
  - **DIC (Brin, Motwani, Ullman, Tsur, SIGMOD 1997)** relaxes this strict synchronization by partitioning the database into intervals of size $M$ (e.g. $M = 50$ transactions).
  - At every interval checkpoint $M$, DIC checks if all subsets of size $k-1$ for an itemset are suspected/confirmed frequent; if so, it begins counting that $k$-itemset immediately without waiting for a full pass to complete.
  - This allows DIC to discover 2-itemsets, 3-itemsets, and 4-itemsets in typically **1.5 to 2 passes** over the database.
  - **STRICT DIRECTIVE:** DIC must be implemented as the genuine Dynamic Itemset Counting algorithm with interval-based candidate introduction and its 4-state machine. FP-Growth, Apriori, or generic counting will not be substituted or mislabeled as DIC.

#### 1.2 Transaction Representation & Preprocessing
Continuous variables cannot be mined directly. Every row of the 301 records is converted into a categorical transaction basket of discrete items:
$$\text{Transaction}_i = \{\text{Sport: } s, \text{Nation: } n, \text{Era: } e, \text{EarningsTier: } t, \text{RankTier: } r, \text{Trajectory: } j\}$$

**Itemset Vocabulary ($D$):**
- `Sport`: 11 items (`Sport_Basketball`, `Sport_Boxing`, `Sport_Golf`, `Sport_Soccer`, `Sport_AutoRacing`, `Sport_Tennis`, `Sport_AmericanFootball`, `Sport_Baseball`, `Sport_Hockey`, `Sport_Cycling`, `Sport_MMA`)
- `Nation`: 2 items (`Nation_USA`, `Nation_International`)
- `Era`: 3 items (`Era_1990s`, `Era_2000s`, `Era_2010s_2020`)
- `EarningsTier`: 3 items (`Earnings_Moderate` [$\le \$25M$], `Earnings_High` [$\$25M-\$60M$], `Earnings_Ultra` [$> \$60M$])
- `RankTier`: 3 items (`Rank_Podium` [1-3], `Rank_MidTop10` [4-7], `Rank_LowerTop10` [8-10])
- `Trajectory`: 5 items (`Traj_Improved`, `Traj_Maintained`, `Traj_Dropped`, `Traj_PromotedTop10`, `Traj_NewEntry`)

*Each transaction contains exactly 6 items; average transaction length = 6.0.*

#### 1.3 Detailed Algorithm Steps (DIC State Machine)
DIC maintains all tracked itemsets in a state machine consisting of 4 distinct categories:
1. **Dashed Square ($\square$):** Suspected infrequent (actively counting, current support $< \text{minsup}$, not yet completed full pass of $N$ transactions).
2. **Dashed Circle ($\bigcirc$):** Suspected frequent (actively counting, current support $\ge \text{minsup}$, not yet completed full pass of $N$ transactions).
3. **Solid Square ($\blacksquare$):** Confirmed infrequent (has been counted through all $N$ transactions in the database, final support $< \text{minsup}$).
4. **Solid Circle ($\CIRCLE$):** Confirmed frequent (has been counted through all $N$ transactions in the database, final support $\ge \text{minsup}$).

```
[Start] --> All 1-itemsets initialized to Dashed Square (count=0, seen=0)
                │
                ▼ Process interval of M transactions
   Update (count, seen) for active itemsets in Dashed states
                │
   Evaluate support threshold (count / seen >= minsup)?
        ├── YES ──> Transition from Dashed Square to Dashed Circle
        └── NO  ──> Remain in Dashed Square (or revert if support dropped)
                │
   [Dynamic Candidate Introduction]:
   For any k-itemset whose immediate (k-1) subsets are ALL in Circle state (Dashed or Solid):
        └── If not already tracked, introduce into Dashed Square (count=0, seen=0)
            and begin counting immediately at this checkpoint!
                │
   [Full Cycle Check]:
   Has an itemset accumulated seen == N transactions?
        ├── If final support (count / N) >= minsup ──> Move to Solid Circle (Confirmed Frequent!)
        └── If final support (count / N) < minsup  ──> Move to Solid Square (Confirmed Infrequent!)
                │
   [Termination Condition]:
   Halt when NO itemsets remain in Dashed states (all candidates confirmed Solid).
```

#### 1.4 Parameters to Test
- Interval Step Size: $M \in \{30, 50, 75\}$ (for $N=301$, $M=50$ provides ~6 checkpoints per pass).
- Minimum Support: $\text{minsup} \in \{0.05, 0.08, 0.10, 0.15\}$ (i.e. occurring in 15 to 45 transactions).
- Minimum Confidence: $\text{minconf} \in \{0.50, 0.60, 0.70, 0.80\}$.
- Minimum Lift threshold: $\text{lift} > 1.2$ (to filter out uninformative associations).

#### 1.5 Validation & Expected Outputs
- **Ground-Truth Validation:** Discovered frequent itemsets and exact support counts will be validated against an exhaustive brute-force counter on the same transactions, mathematically confirming that DIC finds the exact same frequent itemsets with fewer database passes.
- Association rules tabular output: Antecedent $\implies$ Consequent, Support, Confidence, Lift.
- Passes counter: Record exact number of database passes required to reach termination.

---

### Technique 2: K-Nearest Neighbors (KNN) Classification

#### 2.1 Problem Framing & Hypothesis Evaluation
- **Target Selection:** `Sport_Type` (Binary: `Team Sport` vs. `Individual Sport`).
  - `Team Sport` ($N = 146, 48.51\%$): Basketball, Soccer, American Football, Baseball, Hockey.
  - `Individual Sport` ($N = 155, 51.49\%$): Boxing, Golf, Tennis, Auto Racing, Cycling, MMA.
- **Epistemological Framing:** Predicting whether an elite athlete plays a team or individual sport from high-level annual earnings lists is treated as a **hypothesis to evaluate**, not an inherently meaningful operational prediction.
  - **Hypothesis $H_1$:** Top-10 athletes in team sports operate under different compensation and rank dynamics (e.g. collective bargaining, salary caps, stable multi-year team contracts) compared to individual sport athletes (e.g. episodic pay-per-view prize purses, tournament prize money, performance bonuses), creating a multivariate signature detectable by KNN.
  - **Null Hypothesis $H_0$:** At the extreme Top-10 boundary, superstar earnings dynamics are sport-agnostic; KNN will perform no better than a naive baseline.

#### 2.2 Strict Data Leakage Firewall
To guarantee mathematical integrity, features are strictly partitioned:
- **PERMITTED Features ($X$):**
  1. `earnings ($ million)` (Continuous float)
  2. `Year` (Temporal integer)
  3. `Current Rank` (Ordinal integer 1–10)
  4. `Nationality_Group` (Binary encoded: USA = 1, International = 0)
  5. `Previous_Year_Rank_Numeric` (Ordinal rank mapped to numeric with Unranked = 50, and binary indicator `Is_Prior_Unranked` $\in \{0, 1\}$)
- **STRICTLY EXCLUDED Features (Leakage & Identification):**
  - `Sport` and `Sport_Clean` (Direct ground-truth leakage of target!).
  - `S.NO` and `Name` (Arbitrary identifiers).
  - `Rank_Tier` (Derived directly from Current Rank).
  - `Earnings_Category` (Derived directly from earnings).
  - `Rank_Trajectory` (Derived from Current Rank and Previous Rank).
  - `Decade` (Derived directly from Year).

#### 2.3 Experimental Setup & Baselines
- **Baseline Models:**
  - `DummyClassifier(strategy='most_frequent')`: Predicts majority class (`Individual Sport`, expected baseline accuracy $\approx 51.5\%$).
  - `DummyClassifier(strategy='stratified', random_state=42)`: Random guessing matching class priors.
- **Validation Strategy:** Stratified 5-Fold Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- **Fold-Wise Preprocessing:** `StandardScaler` is fitted strictly on $X_{\text{train}}$ of each fold and used to transform $X_{\text{train}}$ and $X_{\text{val}}$ independently. No test data is seen during scaling.

#### 2.4 Parameters to Tune
- Number of Neighbors: $k \in \{1, 3, 5, 7, 9, 11, 15\}$ (Odd values to break ties).
- Distance Metrics: Euclidean ($p=2$) vs. Manhattan ($p=1$).
- Neighbor Weights: `uniform` vs. `distance` ($w_i = 1/d_i$).

#### 2.5 Evaluation Metrics
- Stratified 5-Fold Cross-Validation Accuracy & Balanced Accuracy.
- Precision, Recall, Macro F1-score, and Weighted F1-score.
- Out-of-fold Confusion Matrix and ROC-AUC score.

#### 2.6 Simplest Defensible PySpark Implementation
- In PySpark MLlib, there is no native `KNeighborsClassifier` (Spark MLlib provides K-Means, Logistic Regression, Random Forest, etc., but not KNN).
- **Proposed Defensible Implementation:**
  1. Establish the Scikit-Learn implementation as the gold-standard reference.
  2. In PySpark, implement distributed distance calculation:
     - Broadcast the standardized training matrix ($X_{\text{train}}$) to all Spark executors using `sc.broadcast()`.
     - In parallel across Spark partitions (`mapPartitions` or vectorized UDF), compute pairwise Euclidean distances between each test vector and all broadcast training vectors.
     - Sort distances locally per test vector, select the $k$ smallest, and apply majority voting.
  3. **Strict Validation Requirement:** The PySpark implementation must be validated against Scikit-Learn on the **identical data and train/test split**. Predictions must match 100%.
  4. **Performance Reality:** On 301 rows, do not claim Spark is faster without measured evidence; acknowledge that single-threaded Python is expected to be substantially faster due to zero JVM/RPC overhead.

---

### Technique 3: Divisive Hierarchical Clustering (DIANA)

#### 3.1 Objective & Suitability
- **Objective:** Partition athletes into a hierarchical tree (dendrogram) using a **top-down divisive approach**, identifying macroscopic clusters of career performance and compensation archetypes.
- **Why DIANA (Not AGNES / Agglomerative):**
  - Standard hierarchical clustering is agglomerative (bottom-up: starts with $N$ clusters, merges closest).
  - **DIANA (Kaufman & Rousseeuw, 1990)** is divisive (top-down: starts with 1 grand cluster containing all $N$ objects and recursively splits).
  - Early splits in DIANA identify major high-level clusters with higher stability than agglomerative methods, which can suffer from early errant merges (chaining effect).
  - **STRICT DIRECTIVE:** DIANA must follow the actual Kaufman & Rousseeuw divisive procedure with the Macnaughton-Smith splinter group. Agglomerative clustering will not be substituted.

#### 3.2 Dual-Level Representation Design
DIANA will be executed at two complementary levels:
- **Level A (Row-Level, $N=301$):** Continuous attributes (`earnings`, `Year`, `Current Rank`). Standardized via z-score. Explores temporal-rank-earnings clusters across appearances.
- **Level B (Athlete-Career Level, $N=82$ Unique Athletes):**
  Aggregate career profile vector for each athlete:
  $$\mathbf{x}_{\text{athlete}} = [\text{Total Earnings}, \text{Appearances Count}, \text{Peak Single-Year Earnings}, \text{Best Rank}, \text{Mean Rank}]$$
  *Standardized with StandardScaler.* Produces a clean, highly interpretable 82-leaf dendrogram cleanly separating All-Time GOATs (Tiger, Jordan, Floyd), Sustained Superstars (LeBron, Federer, Ronaldo), and Episodic Top-10 Entrants.

#### 3.3 Detailed Algorithm Steps (Macnaughton-Smith Splinter Technique)
1. Initialize with one single cluster containing all objects: $\mathcal{C}_1 = \{1, 2, \dots, N\}$.
2. At each iteration, select the cluster $\mathcal{C}_m$ with the **largest diameter**:
   $$\text{diam}(\mathcal{C}_m) = \max_{i, j \in \mathcal{C}_m} d(i, j)$$
3. **Splinter Group Initialization:**
   For each object $i \in \mathcal{C}_m$, compute its average dissimilarity to all other objects in $\mathcal{C}_m$:
   $$D(i, \mathcal{C}_m \setminus \{i\}) = \frac{1}{|\mathcal{C}_m| - 1} \sum_{j \in \mathcal{C}_m, j \ne i} d(i, j)$$
   The object with the maximum average dissimilarity initiates the "splinter group" $\mathcal{S} = \{i^*\}$. The remainder is $\mathcal{R} = \mathcal{C}_m \setminus \{i^*\}$.
4. **Iterative Reassignment Loop:**
   For each object $k \in \mathcal{R}$, compute:
   $$\Delta(k) = D(k, \mathcal{R} \setminus \{k\}) - D(k, \mathcal{S})$$
   Find $k^* = \arg\max_{k \in \mathcal{R}} \Delta(k)$. If $\Delta(k^*) > 0$, move $k^*$ from $\mathcal{R}$ to $\mathcal{S}$.
   Repeat until $\Delta(k^*) \le 0$ for all remaining objects in $\mathcal{R}$.
5. Replace $\mathcal{C}_m$ with the two new clusters $\mathcal{R}$ and $\mathcal{S}$.
6. Repeat steps 2–5 until all objects form singleton clusters or desired $k$ clusters are reached. Record divisive heights to generate the dendrogram.

#### 3.4 Parameters to Test & Evaluation
- Distance Metrics: Euclidean distance vs. Manhattan distance.
- Number of Clusters to Cut: $k \in \{2, 3, 4, 5, 6\}$.
- Evaluation: Silhouette Coefficient, Davies-Bouldin Index, and visual validation against SciPy hierarchical baselines.

---

### Technique 4: COFI Closed Association Rule Mining (CLICS)

#### 4.1 Objective & Mathematical Definition
- **Objective:** Mine non-redundant, lossless **Closed Association Rules** (CLICS) from co-occurrence frequent itemset trees (COFI-trees) without combinatorial candidate explosion.
- **Closed Itemset Definition:** An itemset $X$ is **Closed** if no proper superset $Y \supset X$ has the exact same support:
  $$\forall Y \supset X, \quad \text{support}(Y) < \text{support}(X)$$
- **Why COFI (Zaïane & El-Hajj, 2005):**
  - Ordinary association mining generates massive rule redundancy. For example, if $\{\text{Soccer}\} \implies \{\text{Int'l}\}$ has $100\%$ confidence, then any superset $\{\text{Soccer}, X\} \implies \{\text{Int'l}\}$ also has $100\%$ confidence, flooding the output with redundant subsets.
  - COFI builds independent, compact local trees for each frequent item, directly mining closed frequent itemsets without generating non-closed sub-patterns.
  - **STRICT DIRECTIVE:** Generic association mining or FP-Growth must not be labeled as COFI. The implementation must explicitly perform closure checking and produce non-redundant closed rules.

#### 4.2 Algorithm Steps (COFI & CLICS Framework)
1. **First Pass:** Count frequency of all individual items; prune items below $\text{minsup}$ to obtain sorted frequent 1-itemsets $F_1$.
2. **Second Pass:** Build a compact prefix-tree from transactions containing only $F_1$ items (ordered by descending support).
3. **Local COFI-Tree Construction:**
   For each item $i \in F_1$, construct an independent `COFI-tree(i)` containing only paths that contain item $i$ and co-occurring frequent items.
4. **Mining Closed Itemsets:**
   Traverse each `COFI-tree(i)` locally. If a pattern has support equal to one of its supersets, discard it (subsumption checking); retain only maximal support patterns (Closed Frequent Itemsets).
5. **Rule Derivation (CLICS):**
   From the closed itemset collection, derive rules $A \implies B$ where confidence $\ge \text{minconf}$ and no simpler rule $A' \subset A$ provides equal confidence.

#### 4.3 Parameters & Evaluation
- Minimum Support: $\text{minsup} \in \{0.05, 0.08, 0.10, 0.15\}$.
- Minimum Confidence: $\text{minconf} \in \{0.50, 0.60, 0.70, 0.80\}$.
- **Compression Metric:** Calculate the exact compression ratio:
  $$\text{Compression Ratio} = 1 - \frac{|\text{Closed Rules}|}{|\text{Ordinary Rules}|}$$

---

## 4. PySpark Implementation & Synthetic Scaling Benchmark Design

The problem statement mandates:
> *"Test drive any two of the above predictive analytics technique in pyspark platform and establish the performance enhancement achieved by the big data support from Pyspark."*

### 4.1 Selection of the Two PySpark Techniques
1. **PySpark Predictive Technique 1: Distributed K-Nearest Neighbors (KNN)**
   - RDD/DataFrame parallel distance calculation against broadcast train matrix.
   - Evaluated and validated against Scikit-Learn on identical data.
2. **PySpark Predictive Technique 2: Distributed Association Rule Mining**
   - Native PySpark `pyspark.ml.fpm.FPGrowth` on discretized transactions.
   - Distributed MapReduce itemset counter mirroring the interval-based counting mechanism of DIC (`mapPartitions` + `reduceByKey`).

### 4.2 Separation of Actual Model Execution vs. Scalability Experiments
- **Stage A (Real Dataset Execution, $N = 301$):**
  - Run both PySpark KNN and PySpark Association Mining on the authentic 301-row dataset.
  - Record genuine wall-clock latency, output predictions, and metrics.
  - Objectively report timings without attempting to force Spark to outperform Python.
- **Stage B (Synthetic Scalability Experiment, $N = 10\text{K}, 100\text{K}, 1\text{M}$):**
  - A separate, clearly labeled scalability stress test to evaluate distributed scaling behavior.

### 4.3 Synthetic Scaling Methodology & Limitations
- **Expansion Method:** Parametric bootstrap resampling with replacement from the 301-row dataset, augmented with controlled Gaussian jitter ($\mu = 0, \sigma = 0.02 \times \sigma_{\text{feature}}$) applied to continuous variables (`earnings`, `Year`, `Current Rank`).
- **Random Seed:** Strictly pinned to `random_state = 42` for 100% reproducibility.
- **Why Jitter is Necessary:** Pure duplication of identical rows creates artificial nearest neighbors at distance $d = 0$ (trivially corrupting KNN distance distributions) and artificially inflates duplicate transactions in ARM. Controlled perturbation maintains realistic density without changing macro-level distributions.
- **Explicit Distinction:** Benchmark results on synthetic data will be explicitly segregated from findings on the authentic Forbes dataset.
- **Measured Metrics:** Warm execution time (excluding one-time SparkSession boot), throughput (rows/sec), and execution time scaling curve ($\text{Time}$ vs. $N$).

---

## 5. Revised Implementation Order

Prioritizing a correct baseline, working reference implementations, validation, and then benchmarking:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        REVISED IMPLEMENTATION PIPELINE                                 │
├───────┬──────────────────────────────────┬──────────┬──────────────────────────────────┤
│ Phase │ Milestone                        │ Time Est.│ Deliverable Output               │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 1│ Baseline & Data Structuring      │ 10 min   │ data/transactions.json &         │
│       │ (Feature matrices & Dummy cls)   │          │ feature_matrix.csv               │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 2│ Scikit-Learn KNN Reference       │ 15 min   │ models/knn_classifier.py         │
│       │ (Stratified 5-Fold CV + Tuning)  │          │ results/knn_metrics.csv          │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 3│ DIANA Hierarchical Clustering    │ 20 min   │ models/diana_clustering.py       │
│       │ (Macnaughton-Smith + Dendrogram) │          │ visualizations/diana_dendrogram  │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 4│ DIC Association Rule Mining      │ 20 min   │ models/dic_association.py        │
│       │ (Interval-based state machine)   │          │ results/dic_rules.csv            │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 5│ COFI Closed Association Mining   │ 20 min   │ models/cofi_closed_rules.py      │
│       │ (Local trees + CLICS pruning)    │          │ results/cofi_closed_rules.csv    │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 6│ PySpark Implementations          │ 20 min   │ pyspark_implementation.py        │
│       │ (PySpark KNN & PySpark ARM)      │          │ results/pyspark_validation.csv   │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 7│ Synthetic Scalability Benchmark  │ 15 min   │ results/scalability_benchmark.csv│
│       │ (10K / 100K / 1M stress tests)   │          │ visualizations/scaling_curve.png │
├───────┼──────────────────────────────────┼──────────┼──────────────────────────────────┤
│ Step 8│ Final Synthesis Report           │ 15 min   │ reports/FINAL_EXAM_REPORT.md     │
└───────┴──────────────────────────────────┴──────────┴──────────────────────────────────┘
```

---

## 6. Risk Matrix & Quality Gates

| Risk Identified | Severity | Mitigation Strategy in Execution |
| :--- | :---: | :--- |
| **Data Leakage into KNN** | **HIGH** | Strict feature firewall excluding `Sport`, `Sport_Clean`, `S.NO`, `Name`, and target-derived columns. Preprocessing fitted strictly on train folds. |
| **Singleton Class Crash in CV** | **HIGH** | Avoid full 11-class sport classification. Use balanced `Team Sport vs Individual Sport` ($48.5\% / 51.5\%$). |
| **Algorithm Substitution (DIC/COFI/DIANA)** | **CRITICAL**| Implement the exact state machine for DIC, exact local co-occurrence trees for COFI, and exact Macnaughton-Smith splinter group for DIANA. |
| **Pre-judged Spark Superiority** | **HIGH** | Objectively report real measured latency on $N=301$; explain JVM/RPC startup overhead; use synthetic scaling only to demonstrate theoretical crossover. |
| **Discrepancy Between PySpark & Python** | **HIGH** | Quality gate: PySpark KNN predictions must match Scikit-Learn 100% on identical test data before proceeding to scalability benchmarks. |

---

### Plan Approval Readiness

This revised plan incorporates all 8 directives, enforces strict methodological controls, and is ready for implementation.
