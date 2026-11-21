# Technical Report: Comparative Validation of Association Mining Algorithms
## Dynamic Itemset Counting (DIC), Closed Itemset Mining (COFI), and Apache Spark MLlib FP-Growth

**Project:** Forbes Richest Athletes (1990–2020) Analytics  
**Phase:** Phase 4 — Big Data Analytics Platform (Step 4: Comparative Validation)  
**Dataset:** `results/dic_transactions.csv` ($N = 301$ transactions, 23 distinct items, transaction width = 5)  
**Execution Script:** [compare_association_algorithms.py](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/compare_association_algorithms.py)  
**Comparison Summary CSV:** [results/association_algorithm_comparison.csv](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/association_algorithm_comparison.csv)  
**Comparison Metrics JSON:** [results/association_algorithm_comparison.json](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/association_algorithm_comparison.json)  
**Visualizations Directory:** [visualizations/association_comparison/](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/)  

---

## 1. Executive Summary

This report delivers a rigorous, comparative validation of three distinct association pattern mining paradigms implemented across our project on the standardized 301-record Forbes Richest Athletes market-basket transaction dataset:

1. **Dynamic Itemset Counting (DIC)**: A stream-oriented relaxation of the Apriori lattice that dynamically adds candidate itemsets at intermediate checkpoints ($M = 50$) before completing full database passes ([Brin et al., SIGMOD 1997](https://doi.org/10.1145/253260.253325)).
2. **Co-occurrence Frequent Itemset (COFI) Mining**: A pattern-growth closed itemset miner that constructs compact, memory-resident item-conditional subtrees to extract the lossless condensed representation $\mathcal{C}$ and non-redundant association rules ([Zaïane & El-Hajj, DMKD 2005](https://doi.org/10.1007/s10618-005-0002-3)).
3. **Apache Spark MLlib FP-Growth**: A distributed implementation of the Parallel FP-Growth algorithm ([Li et al., PFP 2008](https://doi.org/10.1145/1454008.1454027)) utilizing Spark RDD transformations to parallelize prefix-tree projection across cluster executors.

### Core Quantitative Findings

- **Complete Itemset Parity ($100.0\%$)**: DIC and Apache Spark MLlib FP-Growth both discovered the exact same collection of **87 frequent itemsets** at $\text{minsup} = 0.08$ ($\text{count} \ge 25$). Across all 87 itemsets, the itemset compositions, cardinalities, and exact integer support counts are mathematically identical.
- **Lossless Closed Representation**: COFI successfully identified and extracted all **79 closed frequent itemsets**, pruning exactly **8 non-closed itemsets** ($9.20\%$ compression) with **zero loss of information**. Every pruned itemset was demonstrated to possess a proper closed superset with the exact same support count.
- **Exact Set-Theoretic Rule Partitioning**: Across the three algorithms, a total universe of **80 unique rule signatures** was identified:
  - **47 rules** form the mutual intersection common to **all three algorithms** ($58.75\%$).
  - **9 rules** are exclusive to **DIC and COFI** ($11.25\%$) because both implementations support multi-item consequents ($|B| \ge 2$), which Spark MLlib excludes by API design ($|B| = 1$).
  - **5 rules** are shared by **DIC and Spark FP-Growth** ($6.25\%$) but were **pruned by COFI** because their underlying itemset is non-closed; COFI replaced each with a strictly more informative closed rule possessing identical support and confidence.
  - **19 rules** are exclusive to **Spark FP-Growth** ($23.75\%$) due to threshold filtering differences: Spark MLlib lacks an internal `minLift` filter, generating 13 rules with weak association ($1.00 \le \text{lift} < 1.10$) and 6 rules with independent/negative correlation ($\text{lift} < 1.00$) that were filtered out by DIC and COFI's $\text{minlift} \ge 1.10$ threshold.
  - **0 rules** exist in DIC only, and **0 rules** exist in COFI only, confirming complete mathematical consistency across the entire mining pipeline.
  - **Mathematical Identity Verified**: $47 + 9 + 5 + 19 = 80$ mutually exclusive partition sum matches $|\text{DIC} \cup \text{COFI} \cup \text{Spark}| = 80$ exactly.

---

## 2. Algorithmic Architectures & Theoretical Paradigms

| Dimension | Dynamic Itemset Counting (DIC) | Co-occurrence Frequent Itemset (COFI) | Apache Spark MLlib FP-Growth |
| :--- | :--- | :--- | :--- |
| **Foundational Paradigm** | Relaxed Apriori Lattice (Candidate Generate & Test) | Pattern-Growth via Compressed Subtrees | Distributed Pattern-Growth via Parallel Projections |
| **Primary Data Structure** | Dynamic Itemset Hash Table / Trie with 4 states: `DS`, `DC`, `SS`, `SC` | Global Header Table + In-Memory Local COFI-Trees | Resilient Distributed Dataset (RDD) of Prefix Trees |
| **Candidate Generation** | Yes (dynamic additions at checkpoints $M$) | No (recursive tree traversal) | No (recursive conditional FP-Tree mining) |
| **Database I/O Passes** | 4 passes over data (with $M = 50$) | 2 passes (Pass 1: $F_1$, Pass 2: tree build) | 2 passes (Pass 1: count $F_1$, Pass 2: build RDD trees) |
| **Target Representation** | Complete frequent itemsets $\mathcal{F}$ | Lossless Closed Frequent Itemsets $\mathcal{C}$ | Complete frequent itemsets $\mathcal{F}$ |
| **Consequent Rule Cardinality** | Arbitrary ($|B| \ge 1$) | Arbitrary ($|B| \ge 1$) | Strictly Single-Item ($|B| = 1$) |
| **Redundancy Elimination** | No | Yes (subsumed rules pruned) | No |
| **Target Scale** | Medium-scale sequential / streaming data | Complex, dense categorical datasets | Massive horizontal clusters (TB/PB scale) |

```
Algorithmic Flow Comparison:

[ DIC ]
Transactions ---> [ Checkpoint M=50 ] ---> [ State Transitions: DS -> DC -> SS -> SC ] ---> Complete F (87)
                      ^                             |
                      +-- Candidate Additions <-----+

[ COFI ]
Transactions ---> [ Pass 1: F1 (16) ] ---> [ Pass 2: COFI-Trees ] ---> [ DFS Closed Pruning ] ---> Closed C (79)
                                                                               |
                                                                               +---> Non-redundant Rules (56)

[ Spark FP-Growth ]
Transactions ---> [ Pass 1: Global F1 RDD ] ---> [ Pass 2: Partitioned FP-Trees ] ---> Complete F (87)
                                                                 |
                                                                 +---> Single-Consequent Rules (71)
```

---

## 3. Frequent Itemset Discovery & Closedness Analysis

### 3.1 Cardinality Distribution Comparison

The transaction universe comprises 301 records across 23 distinct one-hot categorical items, with each transaction containing exactly 5 items. The frequent itemset mining results across cardinalities $k=1$ to $k=4$ are summarized below:

| Itemset Cardinality ($k$) | DIC Total | Spark FP-Growth Total | Exact Parity (DIC == Spark) | COFI Closed Total | Pruned Non-Closed Itemsets | Closedness Compression Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 1$** | 16 | 16 | 100.0% Match | 14 | 2 | 12.50% |
| **$k = 2$** | 46 | 46 | 100.0% Match | 42 | 4 | 8.70% |
| **$k = 3$** | 23 | 23 | 100.0% Match | 21 | 2 | 8.70% |
| **$k = 4$** | 2 | 2 | 100.0% Match | 2 | 0 | 0.00% |
| **Total** | **87** | **87** | **100.0% Match** | **79** | **8** | **9.20%** |

![Itemset Cardinality Comparison](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/itemset_cardinality_comparison.png)

### 3.2 Lossless Condensed Basis Verification

An itemset $X$ is defined as **closed** if and only if there exists no proper superset $Y \supset X$ such that $\text{support}(Y) = \text{support}(X)$. When an itemset is not closed, it introduces redundancy into the frequent pattern base because its exact support count can be completely inferred from its closed superset.

COFI pruned exactly 8 frequent itemsets. The table below details each pruned non-closed itemset, its observed support count, and the exact closed superset that subsumes it:

| Cardinality | Pruned Non-Closed Itemset | Support Count | Support (%) | Subsuming Closed Superset | Closed Superset Support | Information Loss |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| $k=1$ | `Sport=Basketball` | 82 | 27.24% | `{Nation=USA, Sport=Basketball}` | 82 | **Zero (0.0%)** |
| $k=1$ | `Sport=Soccer` | 33 | 10.96% | `{Nation=International, Sport=Soccer}` | 33 | **Zero (0.0%)** |
| $k=2$ | `{Decade=2000s, Sport=Basketball}` | 32 | 10.63% | `{Decade=2000s, Nation=USA, Sport=Basketball}` | 32 | **Zero (0.0%)** |
| $k=2$ | `{Decade=2010s, Sport=Basketball}` | 25 | 8.31% | `{Decade=2010s, Nation=USA, Sport=Basketball}` | 25 | **Zero (0.0%)** |
| $k=2$ | `{Earnings=High, Sport=Basketball}` | 47 | 15.61% | `{Earnings=High, Nation=USA, Sport=Basketball}` | 47 | **Zero (0.0%)** |
| $k=2$ | `{Rank=Mid-Tier, Sport=Basketball}` | 40 | 13.29% | `{Nation=USA, Rank=Mid-Tier, Sport=Basketball}` | 40 | **Zero (0.0%)** |
| $k=3$ | `{Decade=2000s, Earnings=High, Sport=Basketball}` | 26 | 8.64% | `{Decade=2000s, Earnings=High, Nation=USA, Sport=Basketball}` | 26 | **Zero (0.0%)** |
| $k=3$ | `{Earnings=High, Rank=Mid-Tier, Sport=Basketball}` | 25 | 8.31% | `{Earnings=High, Nation=USA, Rank=Mid-Tier, Sport=Basketball}` | 25 | **Zero (0.0%)** |

#### Domain Insight into Closedness Pruning
In the historical Forbes dataset:
1. Every single basketball player in the top-10 list is American ($100.0\%$ co-occurrence with `Nation=USA`). Thus, adding `Nation=USA` to any basketball itemset does not reduce support count ($82 \rightarrow 82$).
2. Every single soccer player in the top-10 list is non-American ($100.0\%$ co-occurrence with `Nation=International`). Thus, adding `Nation=International` to `Sport=Soccer` does not reduce support count ($33 \rightarrow 33$).

Because the support of the sub-itemset is identical to the support of the superset, retaining both represents unneeded duplication. COFI correctly identifies and eliminates these 8 unclosed patterns without losing a single unit of information.

---

## 4. Association Rule Mining & Set-Theoretic Overlap

### 4.1 Rule Count Discrepancies and Underlying Causes

The three algorithms yielded differing rule counts under their default approved execution configurations:
- **DIC**: 61 association rules ($\text{conf} \ge 0.50$, $\text{lift} \ge 1.10$, arbitrary $|B|$)
- **COFI**: 56 closed association rules ($\text{conf} \ge 0.50$, $\text{lift} \ge 1.10$, underlying closed itemset, arbitrary $|B|$)
- **Spark FP-Growth**: 71 association rules ($\text{conf} \ge 0.50$, unconstrained lift, single-item consequent $|B| = 1$)

![Rule Overlap Venn](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/rule_overlap_venn.png)

### 4.2 Exact Set-Theoretic Partition Breakdown

Across all three implementations, exactly **80 unique association rules** were generated. The complete Venn partition space is accounted for below:

```
                          Total Rule Universe: 80
                   +-----------------------------------+
                   |         Spark FP-Growth (71)       |
                   |   +---------------------------+   |
                   |   |        DIC (61)           |   |
                   |   |   +-------------------+   |   |
                   |   |   |     Intersection  |   |   |
                   |   |   |   All Three: 47   |   |   |
                   |   |   +-------------------+   |   |
                   |   |         |           |     |   |
                   |   |  DIC &  |           |     |   |
                   |   |  Spark  |           |     |   |
                   |   |  Only: 5|           |     |   |
                   |   +---------+-----------+-----+   |
                   |             |           |         |
                   |             |   DIC &   |         |
                   |             |   COFI    |         |
                   |             |  Only: 9  |         |
                   |             +-----------+         |
                   |                   |               |
                   |                   +---------------+--> COFI (56)
                   |                                   |
                   |   Spark Only: 19                  |
                   |   (Lift < 1.10)                   |
                   +-----------------------------------+
                   DIC Only: 0 | COFI Only: 0
```

#### Partition 1: Mutual Intersection (Common to All Three) — 47 Rules
These 47 rules satisfy all conditions across all three paradigms:
- Single-item consequent ($|B| = 1$).
- Derived from a closed frequent itemset.
- High confidence ($\text{conf} \ge 0.50$) and strong positive correlation ($\text{lift} \ge 1.10$).
- Examples include:
  - `Decade=2010s, Rank=Podium => Earnings=Ultra-High` ($\text{supp}=26, \text{conf}=0.8667, \text{lift}=3.6231$)
  - `Decade=1990s, Rank=Lower => Earnings=Moderate` ($\text{supp}=28, \text{conf}=0.9333, \text{lift}=3.4683$)
  - `Sport=Soccer => Nation=International` ($\text{supp}=33, \text{conf}=1.0000, \text{lift}=3.1684$)

#### Partition 2: DIC & COFI Only (Multi-Item Consequents) — 9 Rules
These 9 rules were generated by both DIC and COFI but omitted by Apache Spark MLlib:

| Antecedent | Consequent | Consequent Length ($|B|$) | Support Count | Confidence | Lift | Reason for Omission in Spark |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `Decade=1990s` | `Earnings=Moderate, Nation=USA` | 2 | 53 | 0.5824 | 2.1887 | MLlib API restricts $|B| = 1$ |
| `Decade=2000s` | `Earnings=High, Nation=USA` | 2 | 66 | 0.7253 | 1.8385 | MLlib API restricts $|B| = 1$ |
| `Decade=2000s, Sport=Basketball` | `Earnings=High, Nation=USA` | 2 | 26 | 0.8125 | 2.0596 | MLlib API restricts $|B| = 1$ |
| `Earnings=High, Sport=Basketball` | `Decade=2000s, Nation=USA` | 2 | 26 | 0.5532 | 1.8385 | MLlib API restricts $|B| = 1$ |
| `Earnings=High, Sport=Basketball` | `Nation=USA, Rank=Mid-Tier` | 2 | 25 | 0.5319 | 1.8378 | MLlib API restricts $|B| = 1$ |
| `Earnings=Moderate` | `Decade=1990s, Nation=USA` | 2 | 53 | 0.7910 | 2.1887 | MLlib API restricts $|B| = 1$ |
| `Rank=Mid-Tier, Sport=Basketball` | `Earnings=High, Nation=USA` | 2 | 25 | 0.6250 | 1.5843 | MLlib API restricts $|B| = 1$ |
| `Sport=Basketball` | `Earnings=High, Nation=USA` | 2 | 47 | 0.5732 | 1.4529 | MLlib API restricts $|B| = 1$ |
| `Sport=Boxing` | `Decade=1990s, Nation=USA` | 2 | 27 | 0.5870 | 1.6241 | MLlib API restricts $|B| = 1$ |

**Significance:** Apache Spark's `org.apache.spark.ml.fpm.FPGrowth` implementation restricts association rule generation to single-item consequents by architectural choice to avoid combinatorial explosions in distributed rule generation. DIC and COFI employ full lattice exploration, discovering these richer multi-item associations.

#### Partition 3: DIC & Spark Only (Pruned by COFI due to Redundancy) — 5 Rules
These 5 rules were generated by DIC and Spark, but were pruned by COFI because their underlying itemset is not closed:

| Rule in DIC & Spark | Support Count | Confidence | Lift | Subsuming Closed Rule in COFI | Closed Rule Support | Closed Rule Conf | Closed Rule Lift |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| `Decade=2000s, Sport=Basketball => Earnings=High` | 26 | 0.8125 | 1.6524 | `Decade=2000s, Sport=Basketball => Earnings=High, Nation=USA` | 26 | 0.8125 | 2.0596 |
| `Earnings=High, Sport=Basketball => Decade=2000s` | 26 | 0.5532 | 1.8300 | `Earnings=High, Sport=Basketball => Decade=2000s, Nation=USA` | 26 | 0.5532 | 1.8385 |
| `Earnings=High, Sport=Basketball => Rank=Mid-Tier` | 25 | 0.5319 | 1.3341 | `Earnings=High, Sport=Basketball => Nation=USA, Rank=Mid-Tier` | 25 | 0.5319 | 1.8378 |
| `Rank=Mid-Tier, Sport=Basketball => Earnings=High` | 25 | 0.6250 | 1.2711 | `Rank=Mid-Tier, Sport=Basketball => Earnings=High, Nation=USA` | 25 | 0.6250 | 1.5843 |
| `Sport=Basketball => Earnings=High` | 47 | 0.5732 | 1.1657 | `Sport=Basketball => Earnings=High, Nation=USA` | 47 | 0.5732 | 1.4529 |

**Redundancy Elimination Analysis:**
In all 5 cases, the antecedent and support count are identical between the non-closed rule and its closed counterpart. However, the closed COFI rule delivers strictly more informative insight: knowing that a player is in basketball and high earnings implies not only that they are in the 2000s, but that they are *an American in the 2000s*—with the exact same confidence ($55.32\%$) and higher lift ($1.8385$ vs $1.8300$). DIC and Spark retain both, producing semantic clutter.

#### Partition 4: Spark Only (Filtered by Lift Threshold) — 19 Rules
Spark MLlib generated 19 rules that DIC and COFI rejected. The breakdown across lift tiers explains this discrepancy:

| Sub-Tier | Count | Lift Range | Interpretation | Examples |
| :--- | :---: | :---: | :--- | :--- |
| **Weak Association** | 13 | $1.00 \le \text{lift} < 1.10$ | Statistically near-independent; very weak co-occurrence | `Earnings=High => Nation=USA` ($\text{lift}=1.0169, \text{conf}=0.6959$)<br>`Rank=Lower => Nation=USA` ($\text{lift}=1.0343, \text{conf}=0.7079$) |
| **Negative / Inverse** | 6 | $\text{lift} < 1.00$ | Negative correlation; antecedent actually suppresses consequent | `Decade=2010s => Nation=USA` ($\text{lift}=0.8183, \text{conf}=0.5600$)<br>`Earnings=Ultra-High => Nation=USA` ($\text{lift}=0.8321, \text{conf}=0.5694$) |

![Rule Count by Lift Category](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/rule_count_by_lift_category.png)

**Analytical Assessment:** A rule with $\text{lift} < 1.00$ indicates that the presence of the antecedent reduces the probability of the consequent relative to its base rate. For example, while $68.4\%$ of all athletes are American, only $56.0\%$ of 2010s athletes are American. While confidence exceeds $0.50$, the relationship is inversely correlated. DIC and COFI filter these out via $\text{minlift} \ge 1.10$, whereas Spark's default output retains them unless post-filtered.

### 4.3 Independent Mathematical Audit & Resolution of the Rule Universe Discrepancy

During independent audit, a discrepancy was investigated where mutually exclusive partition counts ($47 + 9 + 5 + 19 = 80$) conflicted with a previously drafted label stating 76 unique rules. Reconstructing the canonical rule signatures directly from the primary source CSV files (`dic_association_rules.csv`, `cofi_closed_rules.csv`, `pyspark_association_rules.csv`) demonstrates the following:

#### 1. Mutually Exclusive & Collectively Exhaustive Partition Space (Total = 80 Rules)
$$\begin{array}{l|l|c|c}
\hline
\textbf{Partition Identifier} & \textbf{Formal Set Definition} & \textbf{Rule Count} & \textbf{Percentage of Universe} \\
\hline
P_1\text{ (Intersection of All Three)} & \text{DIC} \cap \text{COFI} \cap \text{Spark} & 47 & 58.75\% \\
P_2\text{ (DIC \& COFI Only)} & (\text{DIC} \cap \text{COFI}) \setminus \text{Spark} & 9 & 11.25\% \\
P_3\text{ (DIC \& Spark Only)} & (\text{DIC} \cap \text{Spark}) \setminus \text{COFI} & 5 & 6.25\% \\
P_4\text{ (COFI \& Spark Only)} & (\text{COFI} \cap \text{Spark}) \setminus \text{DIC} & 0 & 0.00\% \\
P_5\text{ (DIC Only)} & \text{DIC} \setminus (\text{COFI} \cup \text{Spark}) & 0 & 0.00\% \\
P_6\text{ (COFI Only)} & \text{COFI} \setminus (\text{DIC} \cup \text{Spark}) & 0 & 0.00\% \\
P_7\text{ (Spark Only)} & \text{Spark} \setminus (\text{DIC} \cup \text{COFI}) & 19 & 23.75\% \\
\hline
\textbf{Total Universe Union} & |\text{DIC} \cup \text{COFI} \cup \text{Spark}| & \mathbf{80} & \mathbf{100.00\%} \\
\hline
\end{array}$$

#### 2. Verification of Set-Theoretic Identity
$$\sum_{i=1}^{7} |P_i| = 47 + 9 + 5 + 0 + 0 + 0 + 19 = 80 \equiv |\text{DIC} \cup \text{COFI} \cup \text{Spark}|$$

#### 3. Reconciliation of Individual Algorithm Totals
Every algorithm's source rule count is exactly reconstructed from these mutually exclusive partitions:
- **DIC Total:** $|P_1| + |P_2| + |P_3| + |P_5| = 47 + 9 + 5 + 0 = \mathbf{61}\text{ rules (100\% match)}$.
- **COFI Total:** $|P_1| + |P_2| + |P_4| + |P_6| = 47 + 9 + 0 + 0 = \mathbf{56}\text{ rules (100\% match)}$.
- **Spark FP-Growth Total:** $|P_1| + |P_3| + |P_4| + |P_7| = 47 + 5 + 0 + 19 = \mathbf{71}\text{ rules (100\% match)}$.

#### 4. Exact Source of the Previous "76" Discrepancy
The previous "76" count was an arithmetic drafting error resulting from evaluating Spark rules ($71$) plus the $5$ unclosed rules from DIC ($71 + 5 = 76$). This erroneously omitted the $9$ multi-item consequent rules shared by DIC and COFI ($71 + 9 = 80$). Alternatively, taking $|\text{DIC}| + |\text{Spark}| - |\text{COFI}| = 61 + 71 - 56 = 76$ inadvertently subtracted COFI's total ($56$) rather than the true pair intersection $|\text{DIC} \cap \text{Spark}| = 52$ ($61 + 71 - 52 = 80$). The partition numbers ($47, 9, 5, 19$) were verified to be completely correct; only the total union label was corrected from 76 to **80**.

---

## 5. Metric Distributions Across Rule Collections

The boxplots below illustrate the distribution of Support, Confidence, and Lift across the generated rule sets:

![Metric Distributions](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/support_confidence_lift_distributions.png)

### Distribution Observations:
1. **Support Distributions**: All three rule sets exhibit comparable support medians ($\approx 0.10$–$0.12$). The minimum support observed across all rules is $0.0831$ (25 records), perfectly matching our minimum threshold.
2. **Confidence Distributions**: All three rule sets strictly enforce $\text{confidence} \ge 0.50$. The median confidence is slightly higher in COFI ($0.730$) and DIC ($0.725$) compared to Spark ($0.667$), as Spark contains many lower-confidence marginal rules.
3. **Lift Distributions**:
   - DIC and COFI distributions are bounded below by $1.10$, with medians around $1.72$ and maxima reaching $3.623$.
   - Spark exhibits a longer downward tail extending to $0.818$, reflecting its unconstrained lift filtering.

---

## 6. Comprehensive Comparative Matrix

| Evaluation Dimension | Dynamic Itemset Counting (DIC) | COFI Closed Mining | Spark MLlib FP-Growth |
| :--- | :--- | :--- | :--- |
| **Total Frequent Itemsets Discovered** | 87 | 79 (Closed) | 87 |
| **Itemset Discovery Parity** | Baseline reference | Lossless compact basis | **100.0% Exact Match with DIC** |
| **Total Association Rules Generated** | 61 | 56 | 71 |
| **Total Unique Rules Union (|DIC ∪ COFI ∪ Spark|)** | 80 | 80 | 80 |
| **Rules Common to All Three** | 47 | 47 | 47 |
| **Multi-Item Consequent Rules ($|B| \ge 2$)** | 9 | 9 | 0 (Not supported by MLlib API) |
| **Subsumed / Redundant Rules** | 5 | 0 (Pruned by closedness) | 5 |
| **Rules with Lift < 1.10** | 0 (Filtered out) | 0 (Filtered out) | 19 (13 weak + 6 negative) |
| **Database I/O Scans** | 4 passes ($M = 50$) | 2 passes | 2 passes |
| **Memory Architecture** | Single-node dynamic trie/hash | Single-node compressed trees | Distributed executor JVM memory |
| **Execution Scaling** | Streaming / Incremental batches | In-memory pattern-growth | Horizontal cluster (Terabytes/Petabytes) |

---

## 7. Situational Recommendations

### When to Select Dynamic Itemset Counting (DIC)
- **Streaming / Batch Data Ingestion**: When data arrives in chunks or sequential intervals, DIC's checkpoint mechanism ($M$) allows itemset counting to start before complete batch ingestion.
- **Moderate Transaction Volumes**: Suitable for medium datasets where full database scans are costly, but the transaction universe fits within a single server's RAM.

### When to Select COFI Closed Mining
- **High Redundancy / Dense Categorical Domains**: When transactions feature high co-occurrence (e.g., medical diagnoses, sports analytics, demographic profiles) where standard frequent miners produce millions of redundant permutations.
- **Rule Interpretability & Business Reporting**: COFI delivers the highest information density per rule by replacing fragmented, lower-information rules with their complete closed counterparts.
- **Zero Loss of Support Counts**: Ideal when complete support reconstruction is required while saving memory and storage space.

### When to Select Apache Spark MLlib FP-Growth
- **Enterprise-Scale Big Data (GB/TB/PB)**: When transactions exceed single-machine memory capacity and require distributed clustering across multiple worker nodes.
- **Standardized Production Pipelines**: Where integration into existing Spark / Hadoop / Databricks enterprise data pipelines is essential.
- **Post-Processing Note**: When using Spark MLlib FP-Growth, practitioners should explicitly filter rule outputs by `lift >= 1.10` and be aware that multi-item consequents require custom implementation outside standard MLlib.

---

## 8. Reproducibility & Artifact Manifest

All artifacts, verification outputs, and visualization assets generated during this comparative validation phase are committed to the project workspace:

- **Comparison Pipeline Script**: [hackathon_live/compare_association_algorithms.py](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/compare_association_algorithms.py)
- **Comparison Summary Table**: [hackathon_live/results/association_algorithm_comparison.csv](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/association_algorithm_comparison.csv)
- **Comprehensive Comparison JSON**: [hackathon_live/results/association_algorithm_comparison.json](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/association_algorithm_comparison.json)
- **Visualizations**:
  - [itemset_cardinality_comparison.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/itemset_cardinality_comparison.png)
  - [rule_count_by_lift_category.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/rule_count_by_lift_category.png)
  - [rule_overlap_venn.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/rule_overlap_venn.png)
  - [support_confidence_lift_distributions.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/association_comparison/support_confidence_lift_distributions.png)
