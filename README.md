# ASBD EndSem Lab Exam - Forbes Highest Paid Athletes 1990-2020

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-4.3.0-orange.svg)](https://spark.apache.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0-green.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red.svg)](https://jupyter.org/)

Comprehensive analytics, unsupervised pattern mining, and distributed machine learning project for the **Analytics and Systems of Big Data (ASBD)** Practice End Semester Lab Examination.

---

## Academic Submission Metadata

| Attribute | Details |
| :--- | :--- |
| **Exam Title** | Analytics and Systems of Big Data Practice End Semester Lab Exam |
| **Student Name** | O Rohith Karthik |
| **Roll Number** | CS23I1011 |
| **Course Name** | Analytics and Systems of Big Data |
| **Course Instructor** | Prof. Sivaselvan B |
| **Submission Date** | 21-November-2026 |
| **Dataset Source** | [Forbes Highest-Paid Athletes 1990–2020 (Kaggle)](https://www.kaggle.com/datasets/parulpandey/forbes-highest-paid-athletes-19902019) |
| **Primary Report** | [`ASBD_EndSem_Final_Report.ipynb`](./ASBD_EndSem_Final_Report.ipynb) |

---

## Executive Summary

This repository presents an end-to-end investigation of athletic compensation, athlete longevity, and market-basket transaction behavior using historical Forbes data ($N = 301$ records from 1990 to 2020). The study covers:
1. **Data Preprocessing & Cleaning:** Resolving text inconsistencies (28 raw sport variants standardized to 10 canonical disciplines), cleaning numeric earnings, handling nationality imbalances (`USA` vs. `Other`), and formatting discretized market-basket transaction files.
2. **Exploratory Data Analytics (EDA):** Generating and interpreting 11 high-resolution statistical visualizations capturing earnings skewness, sport-wise distributions, rank trajectories, and longitudinal decade shifts.
3. **Classical Machine Learning:**
   - **K-Nearest Neighbors (KNN):** Supervised classification of athletic discipline (`Team Sport` vs. `Individual Sport`) with $K=9$, Manhattan distance, achieving **70.49% test accuracy**.
   - **DIANA (Divisive Analysis):** Top-down hierarchical clustering isolating 2 natural macroeconomic athlete cohorts with an optimal **Silhouette Score of 0.4042** at $K=2$.
   - **Dynamic Itemset Counting (DIC):** Step-wise checkpoint itemset mining ($M = 50$), discovering **87 frequent itemsets** and **61 association rules** in 4 passes.
   - **COFI (Co-occurrence Frequent Itemsets):** Prefix-tree-based non-redundant closed pattern mining, generating **79 closed itemsets** (**9.2% redundancy reduction**) and **56 rules** in 2 passes.
4. **Distributed Big Data with Apache PySpark:**
   - **Distributed Broadcast KNN:** Test point distribution via RDDs with broadcast training sets and `mapPartitions` Manhattan distance scanning, achieving **100.0% exact parity** with Scikit-Learn.
   - **Distributed Spark MLlib FP-Growth:** Parallel prefix-tree projection mining identical **87 frequent itemsets** and **71 rules**.
   - **Comparative Rule Overlap Validation:** Canonical signature audit proving exact set-theoretic partition of **80 unique rules** ($47 + 9 + 5 + 19 = 80$).
5. **Big Data Scalability Benchmarking:** Controlled synthetic scaling from $1\times$ to $1,000\times$ (301 to 301,000 records) quantifying the "small data distributed penalty" ($661\times$ to $1,691\times$ overhead on 301 rows) vs. large data amortization ($100\times$ transaction growth yielded only $1.30\times$ runtime increase in Spark FP-Growth).

---

## Repository Directory Architecture

```
hackathon_live/
│
├── ASBD_EndSem_Final_Report.ipynb        # Submission-ready, fully executed Jupyter Notebook
├── README.md                             # Academic repository documentation and guidelines
├── .gitignore                            # Git exclusion rules (bytecode, cache, virtualenvs)
│
├── data/
│   ├── forbes_athletes.csv               # Original raw Kaggle dataset (301 records)
│   ├── forbes_athletes_cleaned.csv       # Standardized, cleaned tabular dataset
│   └── data_dictionary.md                # Attribute definitions, types, and categorical levels
│
├── models/
│   ├── knn_best_model.joblib             # Trained Scikit-Learn KNN Pipeline (K=9, Manhattan)
│   └── diana_model.joblib                # Fitted DIANA divisive clustering model (K=2)
│
├── results/
│   ├── dic_transactions.csv              # Discretized market-basket transaction records
│   ├── dic_frequent_itemsets.csv         # DIC mined frequent itemsets (87 itemsets)
│   ├── dic_association_rules.csv         # DIC generated association rules (61 rules)
│   ├── cofi_closed_itemsets.csv          # COFI mined closed itemsets (79 itemsets)
│   ├── cofi_closed_rules.csv             # COFI generated closed rules (56 rules)
│   ├── pyspark_frequent_itemsets.csv     # Spark MLlib FP-Growth frequent itemsets (87 itemsets)
│   ├── pyspark_association_rules.csv     # Spark MLlib FP-Growth association rules (71 rules)
│   ├── pyspark_knn_test_predictions.csv  # Spark distributed KNN test predictions (61 rows)
│   ├── benchmark_summary.json            # Verified raw benchmark timing & scalability data
│   └── benchmark_scalability.csv         # Scalability benchmark runs across 1x to 1,000x scales
│
├── visualizations/                       # 11 Exploratory Data Analytics plots
│   ├── plot_01_earnings_distribution.png
│   ├── plot_02_earnings_by_sport.png
│   ├── plot_03_top_athletes_frequency.png
│   ├── plot_04_earnings_trend.png
│   ├── plot_05_nationality_distribution.png
│   ├── plot_06_rank_vs_earnings.png
│   ├── plot_07_sport_type_comparison.png
│   ├── plot_08_earnings_by_decade.png
│   ├── plot_09_correlation_heatmap.png
│   ├── plot_10_top_athletes_cumulative_earnings.png
│   ├── plot_11_sport_composition_by_decade.png
│   │
│   ├── diana_dendrogram.png              # DIANA divisive hierarchical tree dendrogram
│   ├── diana_clusters_k2.png             # DIANA 2D PCA cluster projection
│   ├── dic_rules_scatter.png             # DIC support vs. confidence vs. lift scatter
│   ├── dic_top_rules_bar.png             # DIC top rules ranked by lift
│   ├── cofi_closed_itemsets_dist.png     # COFI closed itemset cardinality distribution
│   ├── cofi_rules_scatter.png            # COFI closed rules scatter
│   ├── cofi_top_rules_bar.png            # COFI top rules ranked by lift
│   │
│   ├── association_comparison/           # Comparative analysis visualizations
│   │   ├── rule_overlap_venn.png         # Canonical rule overlap Venn / Euler diagram
│   │   ├── itemset_cardinality_comparison.png
│   │   └── support_confidence_lift_distributions.png
│   │
│   └── benchmarking/                     # Scalability suite plots
│       ├── knn_runtime_breakdown.png
│       ├── spark_overhead_breakdown.png
│       ├── association_runtime_comparison.png
│       ├── scalability_runtime_curves.png
│       ├── scalability_speedup_ratios.png
│       └── scalability_memory_usage.png
│
├── reports/
│   ├── PREPROCESSING_REPORT.md           # Data cleaning & feature transformation audit
│   ├── DESCRIPTIVE_ANALYTICS_REPORT.md   # Detailed observations across 11 EDA plots
│   ├── KNN_EXPERIMENT_REPORT.md          # KNN methodology, validation & classification metrics
│   ├── DIANA_CLUSTERING_REPORT.md        # DIANA divisive clustering math & silhouette evaluation
│   ├── DIC_ASSOCIATION_REPORT.md         # DIC checkpoint lattice transitions & rule mining
│   ├── COFI_ASSOCIATION_REPORT.md        # COFI tree projection, closed pruning & rule metrics
│   ├── PYSPARK_KNN_REPORT.md             # Spark broadcast KNN implementation & parity validation
│   ├── PYSPARK_FPGROWTH_REPORT.md        # Spark MLlib FP-Growth architecture & verification
│   ├── ASSOCIATION_COMPARISON_REPORT.md  # 3-way comparative audit & canonical rule partition
│   └── BIG_DATA_PERFORMANCE_BENCHMARK_REPORT.md # Scalability benchmarking across 1x to 1,000x
│
├── preprocessing.py                      # Standalone data cleaning & transaction generator
├── generate_visualizations.py            # Standalone generator for the 11 EDA plots
├── knn_experiment.py                     # Standalone Scikit-Learn KNN training & evaluation
├── diana_clustering.py                   # Standalone DIANA divisive clustering implementation
├── dic_association_mining.py             # Standalone Dynamic Itemset Counting implementation
├── cofi_closed_mining.py                 # Standalone COFI closed itemset mining implementation
├── pyspark_knn.py                        # Standalone PySpark broadcast KNN implementation
├── pyspark_fpgrowth.py                   # Standalone PySpark MLlib FP-Growth implementation
├── compare_association_algorithms.py     # Standalone comparative rule & itemset audit script
├── benchmark_knn.py                      # KNN cold/warm/session-reset benchmarking script
├── benchmark_association_algorithms.py   # Association mining benchmark suite
└── benchmark_scalability.py              # Synthetic scalability suite (1x to 1,000x)
```

---

## Core Findings and Experimental Results

### 1. Classification & Clustering Performance
- **KNN Classification ($K=9$, Manhattan distance):**
  - Test Accuracy: **70.49%** (43 / 61 correct)
  - Balanced Accuracy: **70.77%**
  - Macro F1-Score: **0.7024** | Weighted F1-Score: **0.7049**
  - PySpark Broadcast KNN yielded **100.0% exact prediction parity** (61/61 identical classifications).
- **DIANA Clustering:**
  - Optimal Silhouette: **0.4042 at $K=2$**.
  - Cluster 0 ($N = 104$, 34.6%): Ultra-high earning, global individual athletes (Soccer, Golf, Boxing).
  - Cluster 1 ($N = 197$, 65.4%): Domestic team sports athletes (Basketball, American Football, Baseball).

### 2. Association Pattern Mining Comparison
- **Frequent Itemset Equivalence:** Both DIC and Spark FP-Growth mined identical **87 frequent itemsets** ($\text{minsup} = 0.0831$, support $\ge 25$).
- **Redundancy Reduction via Closed Itemsets:** COFI mined **79 closed frequent itemsets**, successfully pruning 8 redundant itemsets without information loss (**9.2% compression ratio**).
- **Canonical Rule Overlap Audit:** Evaluating directional implication signatures ($A \Rightarrow B$) across all three algorithms verified exactly **80 unique rules**:
  - $DIC \cap COFI \cap Spark$: **47 rules** (core fundamental relationships)
  - $DIC \cap COFI \setminus Spark$: **9 rules** (multi-consequent rules omitted by Spark MLlib)
  - $DIC \cap Spark \setminus COFI$: **5 rules** (derived from non-closed frequent itemsets)
  - $Spark \setminus [DIC \cup COFI]$: **19 rules** (single-consequent permutations)
  - $DIC \text{ only} = 0$, $COFI \text{ only} = 0$
  - Partition Sum: $47 + 9 + 5 + 19 = \mathbf{80 \text{ unique rules}}$.

### 3. Scalability Benchmarking & Distributed Overhead
- **The Small Data Distributed Penalty ($N = 301$):**
  - Scikit-Learn KNN executed pure compute in **4.24 ms** vs. PySpark warm compute at **2,804.30 ms** (**$661.2\times$ faster**). Including cold JVM bootstrap (8,606.00 ms), Scikit-Learn is **$992.6\times$ faster**.
  - COFI closed mining executed pure compute in **4.29 ms** vs. Spark FP-Growth warm compute at **4,464.10 ms** (**$1,040.6\times$ faster**). Including cold JVM bootstrap (10,038.00 ms), COFI is **$1,691.3\times$ faster**.
- **Big Data Scalability Amortization:**
  - As transaction volume expanded $100\times$ (from 3,010 to 301,000 transactions), Spark FP-Growth compute increased from $5.27\text{ s}$ to only $6.87\text{ s}$ (**$1.30\times$ scaling factor**), proving distributed prefix projection amortizes coordination costs.
  - KNN brute-force broadcast scaling displayed quadratic complexity ($O(N^2)$), requiring $1,540.04\text{ s}$ at $N = 301,000$ (extrapolated from 500 measured queries), whereas Scikit-Learn's KD-Tree branch-pruning completed in $2.57\text{ s}$.

---

## Instructions for Execution & Reproduction

### Prerequisites
- **Python:** Version 3.10+ (tested on Python 3.11.9)
- **Java:** OpenJDK 17 or higher (`JAVA_HOME` configured)
- **Apache PySpark:** Version 3.5.0 or 4.3.0

### Environment Setup
```bash
# Clone the repository
git clone <REPOSITORY_URL>
cd "hackathon_live"

# Install required Python dependencies
pip install numpy pandas matplotlib seaborn scikit-learn pyspark joblib psutil jupyter nbclient
```

### Running the Complete Project via Jupyter
The primary deliverable is the fully executed Jupyter Notebook:
```bash
jupyter notebook ASBD_EndSem_Final_Report.ipynb
```
Or execute headlessly:
```bash
jupyter execute --inplace ASBD_EndSem_Final_Report.ipynb
```

### Running Standalone Pipeline Scripts
```bash
# 1. Clean data and generate transactions
python preprocessing.py

# 2. Generate all 11 EDA plots
python generate_visualizations.py

# 3. Run classical ML models
python knn_experiment.py
python diana_clustering.py
python dic_association_mining.py
python cofi_closed_mining.py

# 4. Run PySpark distributed implementations
python pyspark_knn.py
python pyspark_fpgrowth.py

# 5. Run comparative association audit
python compare_association_algorithms.py

# 6. Run performance benchmarks
python benchmark_knn.py
python benchmark_association_algorithms.py
python benchmark_scalability.py
```

---

## Technical & Experimental Limitations
1. **Local Mode Emulation:** PySpark experiments ran in `local[2]` mode on a dual-core consumer laptop. Real-world cluster execution would introduce physical network socket latency and multi-rack shuffle overhead.
2. **Exact vs. Approximate KNN:** PySpark KNN used exact brute-force Manhattan distance scans. Production-scale high-dimensional search typically utilizes Approximate Nearest Neighbors (ANN) via Locality-Sensitive Hashing (LSH).
3. **Synthetic Scalability Tiers:** Scalability datasets were synthesized via joint empirical resampling (`seed = 42 + scale`), preserving marginal probabilities without introducing novel real-world dimensional sparsity.

---

## License & Academic Integrity
This project is developed solely for academic evaluation in the **Analytics and Systems of Big Data (ASBD)** course at the Indian Institute of Information Technology, Design and Manufacturing (IIITDM), Kancheepuram. All algorithmic source code, benchmarking scripts, and documentation reflect original, rigorously verified research.
