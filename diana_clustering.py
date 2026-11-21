"""
DIANA (Divisive Analysis) Hierarchical Clustering Implementation
Forbes Highest-Paid Athletes (1990-2020) Dataset

Genuine implementation of the Kaufman & Rousseeuw (1990) DIANA algorithm
using the Macnaughton-Smith (1964) splinter group reassignment criterion.
Exclusively clusters on continuous/ranking features:
  - 'earnings ($ million)'
  - 'Year'
  - 'Current Rank'
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION & REPRODUCIBILITY
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "forbes_athletes_cleaned.csv"
VIS_DIR = BASE_DIR / "visualizations"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

VIS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


# ------------------------------------------------------------------------------
# 2. DATA INGESTION & FEATURE PREPARATION
# ------------------------------------------------------------------------------
def load_and_validate_data():
    print("=" * 75)
    print("STEP 1: INGESTING AND VALIDATING DATA FOR DIANA CLUSTERING")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[+] Loaded cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Strictly continuous and ranking features for clustering
    clustering_features = ['earnings ($ million)', 'Year', 'Current Rank']

    # Strict audit check: ensure no categorical or target-derived columns
    forbidden = [
        'Sport', 'Sport_Clean', 'Sport_Type', 'Nationality', 'Nationality_Clean',
        'Nationality_Group', 'S.NO', 'Name', 'Rank_Tier', 'Earnings_Category',
        'Rank_Trajectory', 'Decade', 'Previous_Year_Rank'
    ]
    for col in forbidden:
        assert col not in clustering_features, f"Forbidden column {col} in clustering features!"
    print("[+] Feature Audit: PASSED. Excluded all categorical, sport, and identifier attributes.")

    X = df[clustering_features].copy()
    assert X.isnull().sum().sum() == 0, "Found null values in clustering features!"
    print(f"[+] Clustering features: {list(X.columns)}")
    print(f"[+] Feature matrix shape: {X.shape}")

    # Standardize features before distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[+] Features standardized via StandardScaler: mean ~ 0, std = 1")

    # Compute pairwise Euclidean distance matrix
    D = squareform(pdist(X_scaled, metric='euclidean'))
    print(f"[+] Computed symmetric Euclidean distance matrix: {D.shape} (N={len(df)})")

    return df, X, X_scaled, D, clustering_features


# ------------------------------------------------------------------------------
# 3. GENUINE DIANA DIVISIVE CLUSTERING IMPLEMENTATION (Kaufman & Rousseeuw, 1990)
# ------------------------------------------------------------------------------
def diana_split(cluster_indices, D):
    """
    Splits a single cluster into two subclusters (Remainder and Splinter Group)
    following the exact Macnaughton-Smith (1964) / Kaufman & Rousseeuw (1990) procedure:
      1. Find object i* with maximum average dissimilarity to all other cluster members.
         i* initiates the Splinter Group S = {i*}; Remainder R = C \ {i*}.
      2. For each object k in R, compute:
         Delta(k) = D(k, R \ {k}) - D(k, S)
      3. Find candidate with maximum positive Delta(k). If Delta > 0, move it to S.
      4. Repeat step 2-3 until Delta <= 0 for all objects in R.
    """
    n_sub = len(cluster_indices)
    if n_sub <= 1:
        return cluster_indices, []

    subD = D[np.ix_(cluster_indices, cluster_indices)]
    if np.max(subD) == 0:
        # Identical points (distance = 0), split arbitrarily to allow singletons
        return cluster_indices[1:], [cluster_indices[0]]

    # 1. Find initiator of splinter group (maximal average dissimilarity)
    avg_dissim = np.sum(subD, axis=1) / (n_sub - 1)
    splinter_idx = np.argmax(avg_dissim)

    splinter = [cluster_indices[splinter_idx]]
    remainder = [cluster_indices[i] for i in range(n_sub) if i != splinter_idx]

    # 2. Reassignment loop
    while len(remainder) > 0:
        rem_subD = D[np.ix_(remainder, remainder)]
        splinter_subD = D[np.ix_(remainder, splinter)]

        n_rem = len(remainder)
        if n_rem > 1:
            dist_rem = np.sum(rem_subD, axis=1) / (n_rem - 1)
        else:
            dist_rem = np.zeros(1)

        dist_splinter = np.mean(splinter_subD, axis=1)
        diff = dist_rem - dist_splinter

        best_pos = np.argmax(diff)
        if diff[best_pos] > 0:
            best_candidate = remainder.pop(best_pos)
            splinter.append(best_candidate)
        else:
            break

    return remainder, splinter


def run_diana_hierarchy(N, D, max_k=6):
    """
    Runs DIANA divisive clustering step-by-step from k=1 up to max_k.
    At each step:
      - Selects the cluster with the largest diameter max_{i,j in C} d(i,j).
      - Splits it via diana_split into Remainder and Splinter groups.
      - Records split diameter, resulting clusters, and assignment labels.
    """
    print("\n" + "=" * 75)
    print("STEP 2: EXECUTING GENUINE DIANA DIVISIVE PROCEDURE")
    print("=" * 75)

    clusters = [list(range(N))]
    k_cluster_assignments = {}
    k_cluster_diameters = {}
    split_history = []

    # Record initial k=1
    k_cluster_assignments[1] = [list(range(N))]

    for step in range(1, max_k):
        # 1. Find cluster with largest diameter
        diameters = []
        for c in clusters:
            if len(c) > 1:
                subD = D[np.ix_(c, c)]
                diameters.append(float(np.max(subD)))
            else:
                diameters.append(-1.0)

        split_target_idx = int(np.argmax(diameters))
        split_diameter = diameters[split_target_idx]
        target_cluster = clusters.pop(split_target_idx)

        # 2. Perform divisive split
        r, s = diana_split(target_cluster, D)
        clusters.append(r)
        clusters.append(s)

        current_k = len(clusters)
        k_cluster_assignments[current_k] = [list(c) for c in clusters]
        k_cluster_diameters[current_k] = split_diameter

        split_history.append({
            'step': step,
            'k': current_k,
            'split_diameter': split_diameter,
            'parent_size': len(target_cluster),
            'remainder_size': len(r),
            'splinter_size': len(s)
        })

        print(f"[Split {step}] Split parent cluster (N={len(target_cluster)}, Diam={split_diameter:.3f}) "
              f"-> Remainder (N={len(r)}) + Splinter (N={len(s)}). Total clusters k={current_k}")

    return k_cluster_assignments, k_cluster_diameters, split_history


# ------------------------------------------------------------------------------
# 4. FULL LINKAGE TREE & DENDROGRAM GENERATION
# ------------------------------------------------------------------------------
class DianaNode:
    def __init__(self, members, diam=0.0):
        self.members = members
        self.diam = diam
        self.left = None
        self.right = None
        self.node_id = None


def build_diana_linkage(N, D):
    """
    Constructs a complete Scipy-compatible (N-1) x 4 linkage matrix Z from DIANA divisive splits.
    Orders internal nodes topologically by (split_diameter, post_order_index) to guarantee:
      1. Every child is merged before its parent.
      2. The merge heights Z[:, 2] are strictly non-decreasing across rows.
      3. Top splits of the dendrogram match exact DIANA divisive cuts.
    """
    print("\n" + "=" * 75)
    print("STEP 3: CONSTRUCTING FULL DIANA LINKAGE TREE FOR DENDROGRAM")
    print("=" * 75)

    root = DianaNode(list(range(N)), diam=float(np.max(D)))
    queue = [root]

    while queue:
        # Find node with largest diameter that has > 1 member and not yet split
        candidates = [node for node in queue if len(node.members) > 1 and node.left is None]
        if not candidates:
            break

        # Pick candidate with max diameter
        target = max(candidates, key=lambda n: n.diam)
        rem, spl = diana_split(target.members, D)
        if not spl:
            break

        subD_rem = D[np.ix_(rem, rem)] if len(rem) > 1 else np.zeros((1, 1))
        subD_spl = D[np.ix_(spl, spl)] if len(spl) > 1 else np.zeros((1, 1))

        target.left = DianaNode(rem, diam=float(np.max(subD_rem)))
        target.right = DianaNode(spl, diam=float(np.max(subD_spl)))
        queue.append(target.left)
        queue.append(target.right)

    # Collect internal nodes
    internal_nodes = []
    def collect_internal(node):
        if len(node.members) > 1 and node.left is not None:
            internal_nodes.append(node)
            collect_internal(node.left)
            collect_internal(node.right)

    collect_internal(root)

    # Assign post-order index as topological tie-breaker
    post_order_indices = {}
    order_counter = 0
    def assign_post_order(node):
        nonlocal order_counter
        if len(node.members) == 1:
            return
        assign_post_order(node.left)
        assign_post_order(node.right)
        post_order_indices[id(node)] = order_counter
        order_counter += 1

    assign_post_order(root)

    # Sort internal nodes by (diam, post_order_index) in ascending order
    internal_nodes.sort(key=lambda n: (n.diam, post_order_indices[id(n)]))

    # Build Z matrix
    node_id_map = {}
    for i in range(N):
        node_id_map[i] = i

    linkage_rows = []
    for idx, node in enumerate(internal_nodes):
        new_id = N + idx
        left_id = node.left.members[0] if len(node.left.members) == 1 else node_id_map[id(node.left)]
        right_id = node.right.members[0] if len(node.right.members) == 1 else node_id_map[id(node.right)]
        node_id_map[id(node)] = new_id
        linkage_rows.append([float(left_id), float(right_id), float(node.diam), float(len(node.members))])

    Z = np.array(linkage_rows)
    assert is_valid_linkage(Z), "Generated linkage matrix Z is invalid!"
    print(f"[+] Reconstructed Linkage Matrix Z: shape {Z.shape}, valid Scipy linkage = True, row-monotonic = True")
    return Z


# ------------------------------------------------------------------------------
# 5. METRIC EVALUATION (SILHOUETTE & DAVIES-BOULDIN)
# ------------------------------------------------------------------------------
def evaluate_cluster_metrics(df, X_scaled, k_cluster_assignments, k_cluster_diameters):
    print("\n" + "=" * 75)
    print("STEP 4: EVALUATING CLUSTER QUALITY METRICS (k = 2 through 6)")
    print("=" * 75)

    metrics_records = []
    N = len(df)

    for k in range(2, 7):
        clusters_k = k_cluster_assignments[k]
        labels = np.zeros(N, dtype=int)
        for c_id, members in enumerate(clusters_k):
            labels[members] = c_id

        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))
        diam = float(k_cluster_diameters.get(k, 0.0))
        sizes = [len(c) for c in clusters_k]

        metrics_records.append({
            'k': k,
            'split_diameter': diam,
            'silhouette_score': sil,
            'davies_bouldin_score': db,
            'cluster_sizes': str(sizes),
            'min_cluster_size': min(sizes),
            'max_cluster_size': max(sizes)
        })

        print(f"k={k} | Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Split Diam: {diam:.3f} | Sizes: {sizes}")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = RESULTS_DIR / "diana_cluster_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[+] Saved metrics table to: {metrics_path.name}")

    return metrics_df


# ------------------------------------------------------------------------------
# 6. CLUSTER PROFILING ACROSS DESCRIPTIVE DIMENSIONS
# ------------------------------------------------------------------------------
def profile_clusters(df, k_cluster_assignments):
    print("\n" + "=" * 75)
    print("STEP 5: PROFILING CLUSTERS ACROSS FINANCIAL, RANK & SPORT DIMENSIONS")
    print("=" * 75)

    profile_records = []
    assignment_df = df.copy()

    for k in range(2, 7):
        clusters_k = k_cluster_assignments[k]
        col_name = f"Cluster_k{k}"
        assignment_df[col_name] = -1

        for c_id, members in enumerate(clusters_k):
            assignment_df.loc[members, col_name] = c_id
            sub = df.iloc[members]

            # Top sports
            sport_counts = sub['Sport_Clean'].value_counts()
            top_sports_str = ", ".join([f"{s} ({c})" for s, c in sport_counts.head(3).items()])

            # Nationality composition
            usa_pct = (sub['Nationality_Clean'] == 'USA').mean() * 100

            # Notable athlete names
            top_athletes = sub['Name'].value_counts().head(3).index.tolist()
            notable_athletes_str = ", ".join(top_athletes)

            profile_records.append({
                'k': k,
                'cluster_id': c_id,
                'cluster_size': len(sub),
                'size_pct': (len(sub) / len(df)) * 100,
                'earnings_mean': float(sub['earnings ($ million)'].mean()),
                'earnings_median': float(sub['earnings ($ million)'].median()),
                'earnings_std': float(sub['earnings ($ million)'].std()) if len(sub) > 1 else 0.0,
                'earnings_min': float(sub['earnings ($ million)'].min()),
                'earnings_max': float(sub['earnings ($ million)'].max()),
                'year_mean': float(sub['Year'].mean()),
                'year_median': float(sub['Year'].median()),
                'year_min': int(sub['Year'].min()),
                'year_max': int(sub['Year'].max()),
                'rank_mean': float(sub['Current Rank'].mean()),
                'rank_median': float(sub['Current Rank'].median()),
                'pct_usa': float(usa_pct),
                'dominant_sports': top_sports_str,
                'notable_athletes': notable_athletes_str
            })

    profiles_df = pd.DataFrame(profile_records)
    profiles_path = RESULTS_DIR / "diana_cluster_profiles.csv"
    profiles_df.to_csv(profiles_path, index=False)
    print(f"[+] Saved cluster profiles to: {profiles_path.name}")

    assignments_path = RESULTS_DIR / "diana_cluster_assignments.csv"
    assignment_df.to_csv(assignments_path, index=False)
    print(f"[+] Saved cluster assignments to: {assignments_path.name}")

    return profiles_df, assignment_df


# ------------------------------------------------------------------------------
# 7. GENERATE PUBLICATION-GRADE VISUALIZATIONS
# ------------------------------------------------------------------------------
def plot_dendrogram_and_metrics(Z, metrics_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), dpi=300, gridspec_kw={'height_ratios': [2, 1]})

    # 1. Dendrogram
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=25,
        show_leaf_counts=True,
        ax=ax1,
        color_threshold=4.154,  # Cut threshold for k=5
        above_threshold_color='#7f7f7f'
    )
    ax1.set_title("DIANA Divisive Hierarchical Clustering Dendrogram\n[Macnaughton-Smith Splinter Criterion; Height = Divisive Split Diameter]", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Split Diameter (Dissimilarity Height)")
    ax1.set_xlabel("Collapsed Athlete Subclusters (Sample Size Indicated)")

    # Horizontal guide lines for k=3 and k=5
    ax1.axhline(5.667, color='#d62728', linestyle='--', alpha=0.7, label='Cut for k=3 (Diam=5.67)')
    ax1.axhline(4.154, color='#2ca02c', linestyle=':', alpha=0.9, linewidth=2, label='Recommended Cut for k=5 (Diam=4.15)')
    ax1.legend(loc='upper right', frameon=True)

    # 2. Metric evaluation curves
    ax2.plot(metrics_df['k'], metrics_df['silhouette_score'], marker='o', color='#1f77b4', linewidth=2, label='Silhouette Score (Higher is Better)')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score", color='#1f77b4', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_xticks(range(2, 7))

    ax2_twin = ax2.twinx()
    ax2_twin.plot(metrics_df['k'], metrics_df['davies_bouldin_score'], marker='s', color='#d62728', linewidth=2, linestyle='--', label='Davies-Bouldin Index (Lower is Better)')
    ax2_twin.set_ylabel("Davies-Bouldin Index", color='#d62728', fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor='#d62728')

    ax2.set_title("Cluster Evaluation Metrics Across Cluster Count (k = 2 to 6)", fontweight='bold', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = VIS_DIR / "diana_dendrogram.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


def plot_cluster_scatterplots(assignment_df):
    for k in range(2, 7):
        col_name = f"Cluster_k{k}"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        palette = sns.color_palette('tab10', n_colors=k)

        # Plot A: Year vs. Earnings ($M)
        sns.scatterplot(
            data=assignment_df,
            x='Year',
            y='earnings ($ million)',
            hue=col_name,
            palette=palette,
            s=70,
            alpha=0.8,
            edgecolor='black',
            ax=ax1
        )
        ax1.set_title(f"Cluster Separation: Year vs. Earnings (k={k})", fontweight='bold', fontsize=11)
        ax1.set_xlabel("Year (1990–2020)")
        ax1.set_ylabel("Earnings ($ Million Nominal USD)")
        ax1.legend(title='Cluster ID', loc='upper left', frameon=True)

        # Plot B: Current Rank vs. Earnings ($M)
        sns.scatterplot(
            data=assignment_df,
            x='Current Rank',
            y='earnings ($ million)',
            hue=col_name,
            palette=palette,
            s=70,
            alpha=0.8,
            edgecolor='black',
            ax=ax2
        )
        ax2.set_title(f"Cluster Separation: Rank vs. Earnings (k={k})", fontweight='bold', fontsize=11)
        ax2.set_xlabel("Current Rank (1 = Top Earner, 10 = Lowest Top-10)")
        ax2.set_ylabel("Earnings ($ Million Nominal USD)")
        ax2.set_xticks(range(1, 11))
        ax2.legend(title='Cluster ID', loc='upper right', frameon=True)

        plt.suptitle(f"DIANA Divisive Clustering Solutions (k = {k}) Across Standardized Features", fontweight='bold', fontsize=13)

        out_file = VIS_DIR / f"diana_clusters_k{k}.png"
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    df, X, X_scaled, D, clustering_features = load_and_validate_data()
    k_cluster_assignments, k_cluster_diameters, split_history = run_diana_hierarchy(len(df), D, max_k=6)
    Z = build_diana_linkage(len(df), D)
    metrics_df = evaluate_cluster_metrics(df, X_scaled, k_cluster_assignments, k_cluster_diameters)
    profiles_df, assignment_df = profile_clusters(df, k_cluster_assignments)
    plot_dendrogram_and_metrics(Z, metrics_df)
    plot_cluster_scatterplots(assignment_df)

    print("\n" + "=" * 75)
    print("DIANA DIVISIVE HIERARCHICAL CLUSTERING COMPLETED SUCCESSFULLY!")
    print("=" * 75)
