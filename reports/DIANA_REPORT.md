# DIANA (Divisive Analysis) Hierarchical Clustering Report
## Forbes Highest-Paid Athletes (1990–2020)

**Dataset:** `hackathon_live/data/forbes_athletes_cleaned.csv` ($N = 301$ rows, 16 columns)  
**Clustering Features:** `['earnings ($ million)', 'Year', 'Current Rank']`  
**Execution Script:** `hackathon_live/diana_clustering.py`  
**Generated Visualizations:** `visualizations/diana_dendrogram.png`, `visualizations/diana_clusters_k2.png` through `k6.png`  
**Generated Results:** `results/diana_cluster_metrics.csv`, `results/diana_cluster_profiles.csv`, `results/diana_cluster_assignments.csv`  

---

## 1. Executive Summary

This report documents the independent verification, implementation audit, and empirical evaluation of **DIANA (Divisive Analysis)** hierarchical clustering applied to the Forbes Highest-Paid Athletes dataset.

1. **Algorithmic Integrity:** The implementation adheres strictly to the classic divisive hierarchical paradigm formulated by Kaufman & Rousseeuw (1990), using the exact Macnaughton-Smith (1964) splinter group reassignment criterion. No agglomerative, Ward's linkage, or K-Means approximations were substituted.
2. **Feature Audit Compliance:** Clustering was performed strictly on standardized continuous and ranking attributes: `earnings ($ million)`, `Year`, and `Current Rank`. All identifier columns (`Name`, `S.NO`), raw/clean categorical targets (`Sport`, `Sport_Clean`, `Sport_Type`, `Nationality`, `Nationality_Clean`), and target-derived engineered columns (`Rank_Tier`, `Earnings_Category`, `Rank_Trajectory`, `Decade`) were strictly excluded from feature space.
3. **Metric Evaluation vs. Practical Utility:**
   - **Metric-Supported Solution ($k = 2$):** Statistically, $k = 2$ yields the highest Silhouette score ($0.6713$) and lowest Davies-Bouldin index ($0.5113$). However, this solution is an artifact of **extreme outlier isolation** (3 records vs. 298 records) and fails to reveal meaningful sub-structure within the athlete population.
   - **Interpretation-Oriented Segmentation ($k = 5$):** The 5-cluster partition ($N = [162, 2, 1, 67, 69]$) is not claimed as objectively optimal, but serves as a more informative, interpretation-oriented segmentation. It isolates the singular mega-fight outliers ($N = 2$ and $N = 1$) into distinct outlier bins while segmenting the mainstream population into three substantive groups: an Early/Mid-Period Moderate-Earnings Cohort ($N = 162$), a Multi-Era Elite Podium Cohort ($N = 69$), and a Late-Period Lower Top-10 Cohort ($N = 67$).

---

## 2. Mathematical Formulation & Algorithm Architecture

### 2.1 The DIANA Divisive Paradigm

Unlike agglomerative hierarchical clustering which starts with $N$ singleton clusters and iteratively merges the closest pairs ($O(N^3)$ or $O(N^2 \log N)$), DIANA operates **top-down (divisively)**:
- It begins with a single cluster containing all $N = 301$ observations: $\mathcal{C}_1 = \{1, 2, \dots, N\}$.
- At each step $k$, it identifies the cluster with the largest **diameter** $\mathcal{D}(C)$ and divides it into two subclusters: a **Splinter Group** $\mathcal{S}$ and a **Remainder Group** $\mathcal{R}$.
- Divisive methods establish the primary global structure of the dataset at the earliest splits before local variations are partitioned.

### 2.2 Feature Normalization & Dissimilarity Metric

Features are standardized using `StandardScaler` to ensure zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}, \quad j \in \{\text{Earnings}, \text{Year}, \text{Rank}\}$$

Pairwise dissimilarity $d(i, j)$ is the Euclidean distance in standardized 3D space:
$$d(i, j) = \| \mathbf{z}_i - \mathbf{z}_j \|_2 = \sqrt{\sum_{p=1}^3 (z_{ip} - z_{jp})^2}$$

The diameter of any cluster $C$ is defined as the maximum distance between any two objects within it:
$$\mathcal{D}(C) = \max_{i, j \in C} d(i, j)$$

### 2.3 Macnaughton-Smith Splinter Group Criterion

To split a selected cluster $C$ ($|C| = n$):

1. **Splinter Group Initialization:**
   For each object $i \in C$, compute its average dissimilarity to all other objects in $C$:
   $$D(i, C \setminus \{i\}) = \frac{1}{n - 1} \sum_{j \in C, j \neq i} d(i, j)$$
   The object with the maximum average dissimilarity initiates the splinter group:
   $$i^* = \arg\max_{i \in C} D(i, C \setminus \{i\})$$
   $$\mathcal{S} = \{i^*\}, \quad \mathcal{R} = C \setminus \{i^*\}$$

2. **Iterative Reassignment Loop:**
   For each object $k \in \mathcal{R}$, evaluate whether it is closer to the splinter group $\mathcal{S}$ than to the remainder $\mathcal{R} \setminus \{k\}$. The dissimilarity difference is:
   $$\Delta(k) = D(k, \mathcal{R} \setminus \{k\}) - D(k, \mathcal{S})$$
   where:
   $$D(k, \mathcal{R} \setminus \{k\}) = \frac{1}{|\mathcal{R}| - 1} \sum_{r \in \mathcal{R}, r \neq k} d(k, r)$$
   $$D(k, \mathcal{S}) = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} d(k, s)$$
   
   Identify the candidate with the maximum positive delta:
   $$k^* = \arg\max_{k \in \mathcal{R}} \Delta(k)$$
   - If $\Delta(k^*) > 0$: Object $k^*$ is reallocated from $\mathcal{R}$ to $\mathcal{S}$ ($\mathcal{S} \leftarrow \mathcal{S} \cup \{k^*\}$, $\mathcal{R} \leftarrow \mathcal{R} \setminus \{k^*\}$). Recompute dissimilarities and repeat.
   - If $\Delta(k^*) \le 0$: The iteration terminates. The subclusters $\mathcal{R}$ and $\mathcal{S}$ form the new cluster partition.
   - Non-empty guarantee: If $|\mathcal{R}| = 1$, $D(k, \mathcal{R} \setminus \{k\}) = 0$, so $\Delta(k) \le 0$, ensuring $|\mathcal{R}| \ge 1$ and $|\mathcal{S}| \ge 1$ at all times.

---

## 3. Step-by-Step Divisive Execution Log

DIANA was executed across the 301 athlete observations. The sequence of splits occurred as follows:

| Divisive Step | Parent Size | Split Diameter $\mathcal{D}(C)$ | Remainder Size ($|\mathcal{R}|$) | Splinter Size ($|\mathcal{S}|$) | Resulting $k$ | Structural Nature of Split |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Split 1** | $N = 301$ | **9.638** | $N = 298$ | $N = 3$ | $k = 2$ | Isolates mega-purse boxing anomalies (Mayweather 2015/2018, Pacquiao 2015) |
| **Split 2** | $N = 298$ | **5.667** | $N = 162$ | $N = 136$ | $k = 3$ | Partitions mainstream data into moderate-earnings cohort vs. higher-earnings cohort |
| **Split 3** | $N = 3$ | **4.198** | $N = 2$ | $N = 1$ | $k = 4$ | Subdivides outlier group: Mayweather (\$300M, \$285M) vs. Pacquiao (\$160M) |
| **Split 4** | $N = 136$ | **4.154** | $N = 67$ | $N = 69$ | $k = 5$ | Splits higher-earnings cohort into lower top-10 tier vs. elite podium tier |
| **Split 5** | $N = 162$ | **3.949** | $N = 86$ | $N = 76$ | $k = 6$ | Splits moderate-earnings cohort into lower tier (ranks 6–10) vs. upper tier (ranks 1–5) |

### Key Observations from the Split Hierarchy:
1. **Outlier Monopolization in Early Splits:** Split 1 exhibits a massive diameter ($9.638$), more than $70\%$ larger than Split 2 ($5.667$). The algorithm immediately isolates the singular boxing paydays.
2. **Topological Linkage Alignment:** In `build_diana_linkage`, internal nodes are topologically ordered by split diameter and post-order index. This guarantees that:
   - Every child cluster is merged before its parent.
   - The merge heights $Z[:, 2]$ are strictly non-decreasing across rows ($Z[i, 2] \le Z[i+1, 2]$).
   - Cutting the linkage tree with SciPy's `fcluster` matches the exact DIANA partition at every level $k \in \{2, 3, 4, 5, 6\}$.

---

## 4. Quantitative Cluster Evaluation Metrics ($k = 2$ to $6$)

| Number of Clusters ($k$) | Divisive Split Diameter | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | Cluster Sizes ($[n_1, \dots, n_k]$) | Smallest Cluster ($n_{min}$) | Largest Cluster ($n_{max}$) |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **$k = 2$** | 9.638 | **0.6713** | **0.5113** | $[298, 3]$ | 3 | 298 |
| **$k = 3$** | 5.667 | 0.3593 | 0.9280 | $[3, 162, 136]$ | 3 | 162 |
| **$k = 4$** | 4.198 | 0.3215 | 0.7185 | $[162, 136, 2, 1]$ | 1 | 162 |
| **$k = 5$** | 4.154 | 0.3193 | **0.6608** | $[162, 2, 1, 67, 69]$ | 1 | 162 |
| **$k = 6$** | 3.949 | 0.3337 | 0.7146 | $[2, 1, 67, 69, 86, 76]$ | 1 | 86 |

### Metric Reassessment & Solution Comparison:

1. **Why $k = 2$ is Metric-Supported but Analytically Uninformative:**
   - Statistically, $k = 2$ maximizes Silhouette ($0.6713$) and minimizes Davies-Bouldin ($0.5113$).
   - However, this is driven entirely by extreme geometric detachment: 3 outlier points ($1.0\%$) separated from 298 points ($99.0\%$).
   - It provides no visibility into how compensation, era, or competitive standing structure the remaining $99\%$ of the dataset.
2. **Reassessing $k = 5$ as an Interpretation-Oriented Segmentation:**
   - **Not Objectively Optimal:** $k = 5$ is not objectively optimal in a global metric sense; its Silhouette score ($0.3193$) is lower than $k = 2$ and $k = 3$.
   - **Local Davies-Bouldin Improvement:** The drop in Davies-Bouldin index from $0.9280$ ($k=3$) to $0.6608$ ($k=5$) reflects improved within-cluster compactness when the 136 higher earners are split into ranks 1–5 vs. ranks 6–10.
   - **Small-Cluster / Outlier Awareness:** $k = 5$ explicitly contains two tiny outlier clusters: $N = 2$ (Mayweather 2015, 2018) and $N = 1$ (Pacquiao 2015). These must be recognized as isolated data points, not generalizable segments.
   - **Substantive Clusters:** The three core population clusters at $k=5$ are $N = 162$ ($53.8\%$), $N = 69$ ($22.9\%$), and $N = 67$ ($22.3\%$).

---

## 5. Comprehensive Cluster Profiles ($k = 5$ Solution)

To prevent mischaracterization, cluster labels have been revised to reflect actual numerical distributions rather than overlapping chronological labels:

| Metric / Dimension | Cluster 0: Moderate-Earnings Baseline Cohort | Cluster 1: Extreme PPV Outliers (Mayweather) | Cluster 2: PPV Co-Headliner Outlier (Pacquiao) | Cluster 3: Late-Period Lower Top-10 Tier | Cluster 4: Multi-Era Elite Podium Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cluster Size ($N$)** | **162** (53.82%) | **2** (0.66%) | **1** (0.33%) | **67** (22.26%) | **69** (22.92%) |
| **Mean Earnings (\$M)** | **\$25.04M** | **\$292.50M** | **\$160.00M** | **\$56.00M** | **\$74.60M** |
| **Median Earnings (\$M)** | \$25.20M | \$292.50M | \$160.00M | \$51.00M | \$75.00M |
| **Earnings Range (\$M)** | \$8.1M – \$60.5M | \$285.0M – \$300.0M | \$160.0M | \$35.0M – \$93.4M | \$40.0M – \$127.0M |
| **Earnings Std Dev (\$M)**| \$11.07M | \$10.61M | \$0.00M | \$15.09M | \$22.22M |
| **Mean Year** | **1998.46** | **2016.50** | **2015.00** | **2014.81** | **2010.88** |
| **Year Median** | 1998.0 | 2016.5 | 2015.0 | 2015.0 | 2011.0 |
| **Year Span** | **1990 – 2011** | 2015 – 2018 | 2015 – 2015 | **2008 – 2020** | **1996 – 2020** |
| **Mean Current Rank** | **6.14** | **1.00** | **2.00** | **7.22** | **2.29** |
| **Median Current Rank** | 6.0 | 1.0 | 2.0 | 7.0 | 2.0 |
| **% USA Athletes** | **77.16%** | **100.00%** | **0.00%** | **64.18%** | **52.17%** |
| **Top 3 Sports** | Basketball (48), Boxing (33), Auto Racing (30) | Boxing (2) | Boxing (1) | Basketball (20), Football (12), Golf (12) | Soccer (17), Golf (16), Basketball (14) |
| **Notable Athletes** | Michael Jordan, Shaquille O'Neal, Mike Tyson | Floyd Mayweather | Manny Pacquiao | Phil Mickelson, Roger Federer, LeBron James | Tiger Woods, Lionel Messi, Cristiano Ronaldo |

### Addressing Overlapping Year Distributions:
A critical finding from audit inspection is that **Cluster 0 (1990–2011) and Cluster 4 (1996–2020) overlap across a 16-year period (1996 to 2011)**:
- High-earning podium athletes from the late 1990s and early 2000s (e.g., Michael Jordan 1997 \$78.3M, Tiger Woods 2001 \$78.0M, Michael Schumacher 2004 \$80.0M) are clustered into **Cluster 4** alongside 2010s soccer stars (Messi, Ronaldo) due to their elevated earnings and top-3 ranks.
- Conversely, athletes from that same era occupying lower ranks (ranks 6–10, earning \$15M–\$30M) are placed in **Cluster 0**.
- Therefore, Cluster 0 and Cluster 4 do not represent mutually exclusive chronological eras; rather, Cluster 4 represents **Multi-Era Elite Podium Performers**, while Cluster 0 represents the **Moderate-Earnings Baseline Cohort** of the 1990s and 2000s.
- Cluster 3 represents the **Late-Period Lower Top-10 Tier (2008–2020)**, reflecting modern athletes whose baseline compensation elevated above \$35M even while occupying ranks 6 through 10.

---

## 6. Granular Cluster Profiles ($k = 6$ Solution)

At $k = 6$, Split 5 partitions Cluster 0 into two rank-based sub-tiers:

| Cluster | $N$ | Size % | Mean Earnings | Mean Year | Mean Rank | % USA | Dominant Sports | Representative Athletes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Cluster 0** | 2 | 0.66% | \$292.50M | 2016.5 | 1.00 | 100.0% | Boxing (2) | Floyd Mayweather |
| **Cluster 1** | 1 | 0.33% | \$160.00M | 2015.0 | 2.00 | 0.0% | Boxing (1) | Manny Pacquiao |
| **Cluster 2** | 67 | 22.26% | \$55.00M | 2014.8 | 7.22 | 64.2% | Basketball (20), NFL (12), Golf (12) | Phil Mickelson, Roger Federer |
| **Cluster 3** | 69 | 22.92% | \$74.60M | 2010.9 | 2.29 | 52.2% | Soccer (17), Golf (16), Basketball (14) | Tiger Woods, Lionel Messi, Cristiano Ronaldo |
| **Cluster 4** | 86 | 28.57% | \$22.41M | 2000.2 | 8.27 | 75.6% | Basketball (25), Auto Racing (15), Boxing (10) | Kobe Bryant, Andre Agassi, Arnold Palmer |
| **Cluster 5** | 76 | 25.25% | \$28.02M | 1996.5 | 3.72 | 78.9% | Boxing (23), Basketball (23), Auto Racing (15) | Michael Jordan, Mike Tyson, Shaquille O'Neal |

---

## 7. Visualization Suite Overview

All visualization artifacts are rendered at $300\text{ DPI}$ with publication styling in `hackathon_live/visualizations/`:

1. **`visualizations/diana_dendrogram.png`:**
   - **Top Panel:** Complete DIANA divisive dendrogram generated from the topologically sorted linkage matrix $Z$. Displayed with $p = 25$ representative subclusters and leaf counts. Y-axis represents the exact divisive split diameter. Horizontal guide lines indicate cuts for $k = 3$ (diameter $5.667$) and $k = 5$ (diameter $4.154$).
   - **Bottom Panel:** Dual-axis metric curves plotting Silhouette Score (blue line) and Davies-Bouldin Index (red dashed line) across $k = 2$ through $k = 6$.
2. **`visualizations/diana_clusters_k2.png` through `k6.png`:**
   - 2-panel scatterplots for each cut level:
     - Left: Year vs. Earnings ($M) colored by cluster assignment.
     - Right: Current Rank vs. Earnings ($M) colored by cluster assignment.

---

## 8. Limitations & Methodological Constraints

1. **Low Feature Dimensionality ($D = 3$):**
   - Clustering 301 records using only three continuous features (`earnings`, `Year`, `Current Rank`) captures macro compensation and rank trends, but omits critical sport-specific compensation mechanics (e.g. salary caps in the NBA/NFL vs uncapped European soccer contracts vs individual endorsement deals).
2. **Dataset Size Constraint ($N = 301$):**
   - The dataset contains only the top 10 athletes per year over 30 years (plus one tie year). This represents an extreme right-tail truncation of all professional athletes, meaning clusters reflect nuances within the global top 10 rather than the broader athletic population.
3. **Outlier Sensitivity in Euclidean Space:**
   - DIANA splits by maximum cluster diameter $\max d(i, j)$. Because Floyd Mayweather's earnings ($z \approx +7.5$) create extreme Euclidean distances, Splits 1 and 3 are consumed solely by isolating individual prize-fight paydays rather than partitioning general population structure.
4. **Computational Scaling ($O(N^2)$ Dissimilarity):**
   - DIANA requires an $N \times N$ pairwise distance matrix and iterative reassignment per split ($O(N^2)$). While negligible for $N = 301$, it does not scale to Big Data without partitioning or sampling.
5. **Absence of Ground Truth:**
   - Unsupervised clustering has no external ground-truth labels. The choice of $k$ represents an analytical trade-off between mathematical cluster separation ($k = 2$) and actionable, granular segmentation ($k = 5$).

---

## 9. Audit Summary & Conclusion

- **Algorithm Correctness:** PASS. Genuine DIANA algorithm with Macnaughton-Smith criterion verified. All splits produce non-empty partitions.
- **Metric Verification:** PASS. Silhouette scores, Davies-Bouldin indices, and cluster sizes verified to exact numerical precision.
- **Linkage Matrix Audit:** PASS. Topologically sorted linkage matrix guarantees row-monotonicity and exact `fcluster` cut alignment.
- **Interpretation Correction:** Completed. Overlapping chronological labels replaced with neutral descriptive profiles; $k = 5$ framed as an interpretation-oriented segmentation with explicitly documented outlier clusters.
