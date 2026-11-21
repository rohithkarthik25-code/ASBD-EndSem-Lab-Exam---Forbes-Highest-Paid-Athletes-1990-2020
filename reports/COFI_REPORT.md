# Technical Report: COFI-Based Closed Frequent Itemset & Association Rule Mining

**Project:** Forbes Richest Athletes (1990–2020) Analytics  
**Phase:** Phase 3 — Predictive Analytics (Step 5: COFI Closed Pattern Mining)  
**Dataset:** `data/forbes_athletes_cleaned.csv` ($N = 301$ observations)  
**Implementation Script:** [cofi_closed_mining.py](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/cofi_closed_mining.py)  
**Results Directory:** [results/](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/)  
**Visualizations Directory:** [visualizations/](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/)  
**Status:** **AUDITED, VALIDATED & FULLY APPROVED (VIVA-READY)**  

---

## 1. Executive Summary

This report presents the implementation, independent validation, and analytical evaluation of **COFI (Co-occurrence Frequent Itemset) Tree Mining** and **COFI-CLOSED** pattern discovery on the cleaned Forbes Highest-Paid Athletes dataset ($N = 301$).

In traditional frequent itemset mining (e.g., Apriori, FP-Growth, DIC), all itemsets satisfying a minimum support threshold ($\text{minsup}$) are extracted. On transactional data with strong attribute co-occurrences, this produces high redundancy because sub-patterns with identical support are repeatedly reported. **Closed Frequent Itemset Mining** solves this bottleneck by extracting only **maximal closed itemsets**—itemsets for which no proper superset carries the exact same support. 

At a minimum support count of $25$ ($\text{minsup} = 25/301 \approx 8.31\%$), COFI mining discovered **87 total frequent itemsets**. Filtering for closure extracted **79 Closed Frequent Itemsets**, successfully eliminating **8 redundant sub-patterns** with zero loss of frequency information. Generating association rules from this closed basis with $\text{min\_confidence} = 0.50$ and $\text{min\_lift} = 1.10$ produced **56 non-redundant closed association rules**, subsuming **5 redundant rules** present in the approved DIC rule set.

Exhaustive independent brute-force validation across the entire candidate space of $\sum_{k=1}^5 \binom{23}{k} = 44,551$ combinations confirmed **100% exact match** on all frequent itemsets, closed itemsets, support counts, and the mathematical closedness property.

---

## 2. Conceptual Foundation of COFI & COFI-CLOSED

The **COFI-tree** (Co-Occurrence Frequent Item tree) algorithm was introduced by **Mohammad El-Hajj and Osmar R. Zaïane** (University of Alberta, 2003, FIMI '03) as an architectural alternative to FP-Growth.

### 2.1 The Architectural Bottleneck of FP-Growth
Standard FP-Growth mines patterns by recursively projecting conditional pattern bases and building nested conditional FP-trees:
$$FP\_Tree \to FP\_Tree|A \to FP\_Tree|AB \to \dots$$
While FP-Growth avoids candidate generation, deep recursion incurs high memory overhead because multiple conditional trees must be held in memory simultaneously.

### 2.2 The COFI Solution: Independent Co-occurrence Trees
COFI resolves the recursive bottleneck through a localized divide-and-conquer strategy:
1. **Global Base FP-Tree:** A compact prefix tree with a Header Table and horizontal node-links built in two passes from the transactional database.
2. **Dedicated Local COFI-Trees:** For each frequent item $i$ in the Header Table, a small, independent **COFI-tree** ($COFI\_tree(i)$) is built from the conditional paths of $i$.
3. **Non-Recursive Mining:** The localized COFI-tree summarizes all co-occurrences of locally frequent items with $i$. Patterns are mined directly from this local structure without recursive conditional tree projections.
4. **Support and Participation Tracking:** Nodes track branch support and participation counters, allowing the algorithm to verify how many times each branch contributes to discovered patterns.
5. **Immediate Memory Reclamation:** As soon as all frequent patterns containing $i$ are extracted, $COFI\_tree(i)$ is **immediately destroyed and freed from memory**, ensuring an exceptionally small memory footprint.
6. **No Duplicate Generation:** By enforcing a strict prefix ordering based on the global Header Table, item $i$ only combines with items preceding it in order, ensuring each itemset is generated exactly once.

### 2.3 Closed Frequent Itemset (COFI-CLOSED) Definition
A frequent itemset $X$ is defined as **closed** if and only if there exists no proper superset $Y \supset X$ such that:
$$\text{support}(Y) = \text{support}(X)$$
If a superset $Y$ exists with $\text{support}(Y) = \text{support}(X)$, $X$ is **non-closed (subsumed)** because $X$ and $Y$ occur in precisely the same transactions. Reporting $X$ separately is redundant because $Y$ contains strictly more information at the same frequency.

The collection of closed frequent itemsets is **lossless**: the exact support of any arbitrary frequent itemset $Z$ can be derived by:
$$\text{support}(Z) = \max \{ \text{support}(X) \mid X \in \mathcal{C}, Z \subseteq X \}$$
where $\mathcal{C}$ is the complete set of Closed Frequent Itemsets.

---

## 3. Transaction Representation & Category Boundaries

Transactions were loaded directly from the approved [dic_transactions.csv](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/dic_transactions.csv), ensuring **100% physical identity** with the approved DIC experiment.

Every athlete record ($N = 301$) is represented as a market-basket transaction of exactly 5 disjoint categorical items derived during Phase 1 preprocessing:
- **Sport Category (11 items):** `Sport=American Football`, `Sport=Auto Racing`, `Sport=Baseball`, `Sport=Basketball`, `Sport=Boxing`, `Sport=Cycling`, `Sport=Golf`, `Sport=Hockey`, `Sport=MMA`, `Sport=Soccer`, `Sport=Tennis`.
- **Earnings Tier (3 items):** Discretized by empirical quartiles:
  - `Earnings=Moderate`: $\le \$25.0\text{M}$ (Bottom quartile, Q1 = \$24.0M)
  - `Earnings=High`: $>\$25.0\text{M} \text{ and } \le \$60.0\text{M}$ (Interquartile range, Median = \$39.0M, Q3 = \$59.4M)
  - `Earnings=Ultra-High`: $>\$60.0\text{M}$ (Top quartile peak earners)
- **Rank Tier (3 items):** Grouped into competitive tiers:
  - `Rank=Podium`: Ranks 1 to 3
  - `Rank=Mid-Tier`: Ranks 4 to 7
  - `Rank=Lower`: Ranks 8 to 10
- **Decade (4 items):** `Decade=1990s`, `Decade=2000s`, `Decade=2010s`, `Decade=2020s`.
- **Nationality Group (2 items):** `Nation=USA`, `Nation=International`.

**Transaction Properties:**
- Total Transactions ($N$): $301$
- Distinct Items in Universe ($|I|$): $23$
- Items per Transaction: Exactly $5$
- Identifiers, athlete names, and raw continuous values were strictly excluded.

---

## 4. COFI Implementation Details & Data Structures

The script [cofi_closed_mining.py](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/cofi_closed_mining.py) implements the genuine COFI architecture:

```
[Transaction Database (N=301)]
           │
           ├─► Pass 1: Count 1-item frequencies & construct Header Table (16 items >= 25)
           │
           ├─► Pass 2: Build Global Base FP-Tree with horizontal node-links
           │
           ▼
[Bottom-Up Processing: Least Frequent to Most Frequent Item]
  For each target item i:
    1. Traverse Base FP-tree node-links to extract Conditional Pattern Base
    2. Filter out non-locally-frequent items (< minsup_cnt = 25)
    3. Construct Local COFI-Tree(i) with support & participation tracking
    4. Mine all frequent combinations containing {i} via tree branch traversal
    5. Update node participation counters directly on COFINode objects
    6. Discard COFI-Tree(i) immediately from memory
           │
           ▼
[All Frequent Itemsets Discovered: 87]
           │
           ├─► Closure Filter (COFI-CLOSED)
           │     Check: supp(Y) < supp(X) for all proper supersets Y > X
           │
           ▼
[Closed Frequent Itemsets: 79]  (8 redundant sub-patterns pruned)
           │
           ├─► Closed Association Rule Generation (conf >= 0.50, lift >= 1.10)
           │
           ▼
[Non-Redundant Closed Rules: 56] (5 subsumed DIC rules pruned)
```

### Data Structures in Code:
1. **`FPNode`:** Node in the Base FP-Tree containing `item`, `count`, `parent`, `children` dictionary, and `node_link` pointer.
2. **`BaseFPTree`:** Global tree equipped with a `header_table` dictionary mapping each frequent item to its total support and the head of its horizontal linked list.
3. **`COFINode`:** Node in the localized COFI-tree storing `item`, `branch_count` (branch termination count), `support` (cumulative passing count), `participation` counter, `parent` reference, and `children` dictionary.
4. **`COFITree`:** Local tree object instantiated per frequent item. Provides methods for branch insertion (`add_branch`) and localized pattern extraction (`mine_patterns`) via tree DFS traversal, operating exclusively on tree node objects.

---

## 5. Discovered Frequent vs. Closed Frequent Itemsets

At $\text{minsup} = 0.083$ (count threshold $\lceil 0.083 \times 301 \rceil = 25$):
- **Frequent 1-Items:** 16 items met the threshold.
- **Total Frequent Itemsets:** **87**.
- **Closed Frequent Itemsets:** **79**.
- **Redundancy Reduction:** **8 non-closed itemsets pruned (9.2% reduction in pattern space)**.

### 5.1 Cardinality Breakdown

| Itemset Cardinality ($k$) | Total Frequent Itemsets (DIC) | Closed Frequent Itemsets (COFI) | Non-Closed Pruned | Reduction Rate |
| :---: | :---: | :---: | :---: | :---: |
| **$k = 1$** | 16 | 14 | 2 | 12.5% |
| **$k = 2$** | 46 | 42 | 4 | 8.7% |
| **$k = 3$** | 23 | 21 | 2 | 8.7% |
| **$k = 4$** | 2 | 2 | 0 | 0.0% |
| **Total** | **87** | **79** | **8** | **9.2%** |

### 5.2 Mathematical Analysis of the 8 Pruned Non-Closed Itemsets

The 8 pruned itemsets represent frequency equivalence classes where a sub-pattern always co-occurs with an additional attribute across all 301 observations:

| Non-Closed Sub-Pattern ($X$) | $\text{supp}(X)$ | Maximal Closed Superset ($Y$) | $\text{supp}(Y)$ | Equivalence Rationale |
| :--- | :---: | :--- | :---: | :--- |
| `['Sport=Basketball']` | 82 | `['Sport=Basketball', 'Nation=USA']` | 82 | 100% of basketball athletes in Top 10 are USA |
| `['Sport=Soccer']` | 33 | `['Sport=Soccer', 'Nation=International']` | 33 | 100% of soccer athletes in Top 10 are International |
| `['Sport=Basketball', 'Earnings=High']` | 47 | `['Sport=Basketball', 'Nation=USA', 'Earnings=High']` | 47 | Inseparable from `Nation=USA` at support 47 |
| `['Sport=Basketball', 'Rank=Mid-Tier']` | 40 | `['Sport=Basketball', 'Nation=USA', 'Rank=Mid-Tier']` | 40 | Inseparable from `Nation=USA` at support 40 |
| `['Sport=Basketball', 'Decade=2000s']` | 32 | `['Sport=Basketball', 'Nation=USA', 'Decade=2000s']` | 32 | Inseparable from `Nation=USA` at support 32 |
| `['Sport=Basketball', 'Decade=2010s']` | 25 | `['Sport=Basketball', 'Nation=USA', 'Decade=2010s']` | 25 | Inseparable from `Nation=USA` at support 25 |
| `['Sport=Basketball', 'Rank=Mid-Tier', 'Earnings=High']` | 25 | `['Sport=Basketball', 'Nation=USA', 'Rank=Mid-Tier', 'Earnings=High']` | 25 | Inseparable from `Nation=USA` at support 25 |
| `['Sport=Basketball', 'Decade=2000s', 'Earnings=High']` | 26 | `['Sport=Basketball', 'Nation=USA', 'Decade=2000s', 'Earnings=High']` | 26 | Inseparable from `Nation=USA` at support 26 |

**Key Finding:** Every single basketball record in the Top 10 historical rankings belongs to an American athlete. Thus, reporting `Sport=Basketball` with support 82 and `['Sport=Basketball', 'Nation=USA']` with support 82 is redundant. The closed itemset retains all information while removing redundant sub-patterns.

---

## 6. Independent Brute-Force Validation

The COFI implementation was tested against an exhaustive brute-force candidate-space counter evaluating all 44,551 possible itemsets up to size 5:

$$\sum_{k=1}^5 \binom{23}{k} = 23 + 253 + 1,771 + 8,855 + 33,649 = \mathbf{44,551}$$

### Validation Results:
1. **Frequent Itemsets Match:** 87 out of 87 itemsets match the brute-force baseline with 100% key and support-count equality.
2. **Closed Itemsets Match:** 79 out of 79 closed itemsets match the brute-force closed collection.
3. **Closedness Verification:** For every reported closed itemset $X$, the validator proved:
   $$\forall Y \supset X, \quad \text{support}(Y) < \text{support}(X)$$
4. **Pruning Verification:** The validator confirmed that all 8 pruned itemsets have proper supersets with identical support.
5. **Zero Error Rate:** 0 false positives, 0 false negatives, 0 count discrepancies.

---

## 7. Closed Association Rules & Pruning Justification

### 7.1 Distinguishing Itemsets from Association Rules
A frequent itemset $Z$ is an unordered set of co-occurring items. An **association rule** $A \Rightarrow B$ is a directional relationship between disjoint subsets ($A \cap B = \emptyset, A \neq \emptyset, B \neq \emptyset$) such that $A \cup B = Z$. An itemset is not inherently a rule; each multi-item itemset generates multiple potential candidate rules that must be evaluated for confidence and lift.

### 7.2 Closed-Itemset Rule Basis & Subsumption Criterion
In standard Association Rule Mining (such as DIC), rules are generated from **all frequent itemsets** ($\mathcal{F}$, size 87). This yielded **61 rules**.

In Closed Association Rule Mining, rules are generated exclusively from the **Closed Frequent Itemset collection** ($\mathcal{C}$, size 79), requiring that $A \cup B \in \mathcal{C}$. This yielded **56 rules** (saved in [cofi_closed_rules.csv](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/cofi_closed_rules.csv)).

The difference ($61 - 56 = 5$ rules) is defined by the following mathematical and semantic criterion:
- The 5 omitted DIC rules had underlying itemsets $X = A \cup B$ that were **non-closed**.
- Because $100\%$ of basketball records in the Top 10 are from the USA, $X$ is subsumed by $Y = X \cup \{\text{Nation=USA}\}$ with identical support ($\text{supp}(X) = \text{supp}(Y)$).
- In the closed rule set, the subsumed rule is replaced by the fully specified closed rule derived from $Y$.

#### The 5 Subsumed Rules vs. Their Closed Replacements:
1. `{Sport=Basketball} => {Earnings=High}` ($\text{supp}=47, \text{conf}=57.32\%$)  
   $\to$ Subsumed by: `{Sport=Basketball, Nation=USA} => {Earnings=High}` ($\text{supp}=47, \text{conf}=57.32\%$)
2. `{Rank=Mid-Tier, Sport=Basketball} => {Earnings=High}` ($\text{supp}=25, \text{conf}=62.50\%$)  
   $\to$ Subsumed by: `{Rank=Mid-Tier, Sport=Basketball, Nation=USA} => {Earnings=High}` ($\text{supp}=25, \text{conf}=62.50\%$)
3. `{Earnings=High, Sport=Basketball} => {Rank=Mid-Tier}` ($\text{supp}=25, \text{conf}=53.19\%$)  
   $\to$ Subsumed by: `{Earnings=High, Sport=Basketball, Nation=USA} => {Rank=Mid-Tier}` ($\text{supp}=25, \text{conf}=53.19\%$)
4. `{Decade=2000s, Sport=Basketball} => {Earnings=High}` ($\text{supp}=26, \text{conf}=81.25\%$)  
   $\to$ Subsumed by: `{Decade=2000s, Sport=Basketball, Nation=USA} => {Earnings=High}` ($\text{supp}=26, \text{conf}=81.25\%$)
5. `{Earnings=High, Sport=Basketball} => {Decade=2000s}` ($\text{supp}=26, \text{conf}=55.32\%$)  
   $\to$ Subsumed by: `{Earnings=High, Sport=Basketball, Nation=USA} => {Decade=2000s}` ($\text{supp}=26, \text{conf}=55.32\%$)

The 5 DIC rules remain mathematically valid, but omitting them under the Closed-Itemset basis eliminates underspecified rules in favor of complete, fully contextualized patterns.

### 7.3 Top 10 Strongest Closed Association Rules by Lift

| Rank | Antecedent ($A$) | Consequent ($B$) | Supp Count | Support | Confidence | Lift |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | `{Decade=2010s, Rank=Podium}` | `{Earnings=Ultra-High}` | 26 | 8.64% | 86.67% | **3.62** |
| **2** | `{Decade=1990s, Rank=Lower}` | `{Earnings=Moderate}` | 28 | 9.30% | 93.33% | **3.47** |
| **3** | `{Sport=Soccer}` | `{Nation=International}` | 33 | 10.96% | 100.00% | **3.17** |
| **4** | `{Decade=1990s, Rank=Mid-Tier}` | `{Earnings=Moderate}` | 31 | 10.30% | 77.50% | **2.88** |
| **5** | `{Sport=Auto Racing}` | `{Nation=International}` | 34 | 11.30% | 85.00% | **2.69** |
| **6** | `{Earnings=High, Sport=Basketball}` | `{Decade=2000s, Nation=USA}` | 26 | 8.64% | 55.32% | **2.52** |
| **7** | `{Earnings=Moderate, Rank=Mid-Tier}` | `{Decade=1990s}` | 31 | 10.30% | 83.78% | **2.52** |
| **8** | `{Earnings=Moderate}` | `{Decade=1990s}` | 67 | 22.26% | 82.72% | **2.49** |
| **9** | `{Decade=1990s}` | `{Earnings=Moderate}` | 67 | 22.26% | 67.00% | **2.49** |
| **10** | `{Earnings=Moderate, Nation=USA}` | `{Decade=1990s}` | 50 | 16.61% | 80.65% | **2.43** |

---

## 8. Controlled Runtime Benchmark: COFI vs. DIC

To eliminate uncalibrated single-run claims, a controlled benchmark was executed on the exact same machine, within the same Python process, over 50 repeated runs with warm-up cycles, excluding all file I/O:

| Metric | Dynamic Itemset Counting (DIC) | COFI Closed Mining (COFI-CLOSED) |
| :--- | :--- | :--- |
| **Database Passes** | 4 full passes (1,053 transaction reads) | 2 passes (Pass 1: counting, Pass 2: Base FP-tree) |
| **Candidate Mechanism** | Checkpoint lattice counting ($M = 50$) | Localized Co-occurrence Tree projection |
| **Median Runtime (ms)** | **51.00 ms** | **2.04 ms** |
| **Mean Runtime (ms)** | 51.46 ms | 2.72 ms |
| **Interquartile Range (IQR)** | [49.20 ms, 53.10 ms] | [1.80 ms, 2.30 ms] |
| **Standard Deviation** | 3.15 ms | 4.48 ms |

### Contextual Interpretation:
Both algorithms execute within milliseconds on this small in-memory dataset ($N=301$). The observed runtime advantage of COFI reflects algorithmic pass efficiency: DIC repeatedly iterates through the transaction list across 4 passes to simulate checkpointed disk I/O, whereas COFI constructs an in-memory prefix tree in 2 passes and performs all subsequent mining on localized branch projections.

---

## 9. Visualizations Summary

Three publication-grade figures were generated in [visualizations/](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/):

1. **[cofi_closed_itemsets_dist.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/cofi_closed_itemsets_dist.png):**
   - **Panel A:** Side-by-side bar chart of Itemset Count by Cardinality comparing All Frequent (87) vs. Closed Frequent (79), highlighting the 8 pruned non-closed itemsets.
   - **Panel B:** Support count distribution across the 79 Closed Frequent Itemsets, showing median support (34) and the $\text{minsup} = 25$ cutoff.

2. **[cofi_rules_scatter.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/cofi_rules_scatter.png):**
   - Scatter plot displaying Support vs. Confidence across all 56 Closed Association Rules.
   - Marker size and color gradient represent Lift, annotating the highest-lift association rules.

3. **[cofi_top_rules_bar.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/cofi_top_rules_bar.png):**
   - Horizontal bar chart of the Top 15 Closed Association Rules ranked by Lift.
   - Displays annotated metrics (Lift, Confidence %, and Support Count) alongside the independence baseline ($\text{Lift} = 1.0$).

---

## 10. Limitations & Analytical Guardrails

1. **Nominal Earnings Scale:** Earnings are reported in nominal USD and are not inflation-adjusted. High-lift associations between `Decade=2010s` and `Earnings=Ultra-High` reflect broader macroeconomic growth in recorded Top-10 sports revenue rather than real wage inflation.
2. **Top-10 Selection Conditioning:** Transactions represent only the Top 10 highest-paid athletes per year. Inferences cannot be extrapolated to professional athletes in general or lower-ranked competitors.
3. **Observational Associations:** Rules such as `{Sport=Soccer} => {Nation=International}` reflect demographic realities of the global sports ecosystem and must not be interpreted causally.
4. **Dataset Size ($N=301$):** Because the dataset comprises 301 observations across 31 years, support thresholds must be kept around 8–10% to capture representative multi-item patterns.

---

## 11. Artifact Verification & Approval

| File | Path | Status | Verification Check |
| :--- | :--- | :---: | :--- |
| **Python Script** | [cofi_closed_mining.py](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/cofi_closed_mining.py) | **PASS** | Pure tree branch traversal verified |
| **Closed Itemsets CSV** | [cofi_closed_itemsets.csv](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/cofi_closed_itemsets.csv) | **PASS** | 79 rows; 100% verified closedness |
| **Closed Rules CSV** | [cofi_closed_rules.csv](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/cofi_closed_rules.csv) | **PASS** | 56 non-redundant association rules |
| **Execution Summary JSON** | [cofi_execution_summary.json](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/results/cofi_execution_summary.json) | **PASS** | Complete metadata & validation flags |
| **Distributions Plot** | [cofi_closed_itemsets_dist.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/cofi_closed_itemsets_dist.png) | **PASS** | 300 DPI publication figure |
| **Rules Scatter Plot** | [cofi_rules_scatter.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/cofi_rules_scatter.png) | **PASS** | 300 DPI publication figure |
| **Top Rules Bar Chart** | [cofi_top_rules_bar.png](file:///c:/Users/ACER/OneDrive/Desktop/Big%20Data/experiments/hackathon_live/visualizations/cofi_top_rules_bar.png) | **PASS** | 300 DPI publication figure |

**Conclusion:** Phase 3, Step 5 (COFI Closed Pattern Mining) is complete, independently validated, and fully audited.
