"""
================================================================================
PHASE 3, STEP 5: COFI-BASED CLOSED FREQUENT ITEMSET & ASSOCIATION RULE MINING
================================================================================
Dataset: Forbes Richest Athletes (1990-2020) Cleaned Dataset (N = 301)
Implementation: Genuine COFI (Co-occurrence Frequent Itemset) Tree Mining
Author: Antigravity Agentic Assistant (Viva-Ready Practical Exam Edition)
Date: November 2026

ALGORITHMIC ARCHITECTURE:
1. Two-pass Global Base FP-Tree construction with Header Table and Node-Links.
2. Bottom-up mining using dedicated, localized COFI-Trees for each frequent item.
3. COFI-tree nodes track 'branch_count', 'support', and 'participation' counters.
4. Local trees are mined non-recursively via tree branch traversal, eliminating
   recursive conditional tree construction overhead.
5. Immediate memory reclamation (COFI-trees are discarded immediately after mining).
6. Non-redundant candidate generation by restricting co-occurring items to prefix order.
7. Closed Frequent Itemset (COFI-CLOSED) identification via exact closure checking:
   An itemset X is closed iff no proper superset Y contains the exact same support.
8. Closed Association Rule generation: non-trivial rules derived from closed itemsets.
9. Independent Brute-Force Validation over all 44,551 combinatorial candidate sets.
================================================================================
"""

import sys
import json
import time
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATION & STYLING
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "forbes_athletes_cleaned.csv"
TRANS_PATH = BASE_DIR / "results" / "dic_transactions.csv"
RESULTS_DIR = BASE_DIR / "results"
VIS_DIR = BASE_DIR / "visualizations"
REPORTS_DIR = BASE_DIR / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


# ------------------------------------------------------------------------------
# 2. STEP 1: TRANSACTION LOADING & VERIFICATION
# ------------------------------------------------------------------------------
def load_transactions():
    """
    Loads and verifies the 5-dimensional categorical transactions.
    Reuses approved dic_transactions.csv directly to ensure 100% consistency with DIC.
    Dimensions & Boundaries (derived in Phase 1 preprocessing):
      1. Sport: Canonical sport category (11 distinct sports)
      2. Earnings Tier:
         - Moderate: <= $25.0M (Bottom quartile, Q1 = $24.0M)
         - High: > $25.0M and <= $60.0M (Interquartile range, Median = $39.0M, Q3 = $59.4M)
         - Ultra-High: > $60.0M (Top quartile peak earners)
      3. Rank Tier:
         - Podium: Ranks 1-3
         - Mid-Tier: Ranks 4-7
         - Lower: Ranks 8-10
      4. Decade: 1990s, 2000s, 2010s, 2020s
      5. Nationality Group: USA, International
    """
    print("=" * 75)
    print("STEP 1: TRANSACTION LOADING & VERIFICATION")
    print("=" * 75)

    if TRANS_PATH.exists():
        trans_df = pd.read_csv(TRANS_PATH)
        transactions = [frozenset(t.split("; ")) for t in trans_df['Items']]
        print(f"[+] Loaded {len(transactions)} transactions directly from approved: {TRANS_PATH.name}")
    else:
        df = pd.read_csv(DATA_PATH)
        transactions = []
        records = []
        for idx, row in df.iterrows():
            earn_cat = str(row['Earnings_Category']).split(' ')[0]
            rank_cat = str(row['Rank_Tier']).split(' ')[0]
            sport_cat = str(row['Sport_Clean']).strip()
            decade_cat = str(row['Decade']).strip()
            nation_cat = str(row['Nationality_Group']).strip()

            items = [
                f"Sport={sport_cat}",
                f"Earnings={earn_cat}",
                f"Rank={rank_cat}",
                f"Decade={decade_cat}",
                f"Nation={nation_cat}"
            ]
            transactions.append(frozenset(items))
            records.append({
                'Transaction_ID': idx + 1,
                'Athlete_Name': row['Name'],
                'Year': row['Year'],
                'Items': "; ".join(sorted(items))
            })
        trans_df = pd.DataFrame(records)
        trans_df.to_csv(TRANS_PATH, index=False)
        print(f"[+] Constructed {len(transactions)} transactions from cleaned dataset.")

    unique_items = set()
    for t in transactions:
        assert len(t) == 5, f"Transaction malformed: {t}"
        unique_items.update(t)

    print(f"[+] Transaction Count (N): {len(transactions)}")
    print(f"[+] Distinct Items in Universe: {len(unique_items)}")
    return transactions, unique_items


# ------------------------------------------------------------------------------
# 3. STEP 2: COFI DATA STRUCTURES & MINING ENGINE
# ------------------------------------------------------------------------------
class FPNode:
    """
    Node in the Global Base FP-Tree.
    Tracks item label, frequency count, parent pointer, child map, and header node-link.
    """
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None


class BaseFPTree:
    """
    Global compact frequent pattern tree constructed from transactions.
    Equipped with a Header Table and horizontal node-links for rapid traversal.
    """
    def __init__(self):
        self.root = FPNode(None, 0, None)
        self.header_table = {}  # item -> [support, head_node_link]

    def add_transaction(self, items, count=1):
        curr = self.root
        for item in items:
            if item not in curr.children:
                new_node = FPNode(item, count, curr)
                curr.children[item] = new_node
                # Maintain horizontal linked list from header table
                if item in self.header_table:
                    new_node.node_link = self.header_table[item][1]
                    self.header_table[item][1] = new_node
                else:
                    self.header_table[item] = [0, new_node]
            else:
                curr.children[item].count += count
            curr = curr.children[item]


class COFINode:
    """
    Node in a localized Co-occurrence Frequent Item (COFI) Tree.
    Tracks item label, branch_count (count terminating at this node),
    support (total branch support passing through), participation counter,
    parent reference, and child nodes dictionary.
    """
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.branch_count = count
        self.support = count
        self.participation = 0
        self.parent = parent
        self.children = {}


class COFITree:
    """
    Localized Co-occurrence Frequent Item Tree for a single frequent item.
    Constructed directly from the item's conditional pattern base in the Base FP-Tree.
    Summarizes all co-occurrences of locally frequent items with the target item.
    Mined non-recursively via tree branch traversal, and immediately discarded.
    """
    def __init__(self, target_item):
        self.target_item = target_item
        self.root = COFINode(None, 0, None)
        self.local_header = defaultdict(int)

    def add_branch(self, branch_items, count):
        """Inserts a prefix path into the local COFI-tree."""
        curr = self.root
        for item in branch_items:
            self.local_header[item] += count
            if item not in curr.children:
                new_node = COFINode(item, 0, curr)
                curr.children[item] = new_node
            curr = curr.children[item]
            curr.support += count
        curr.branch_count += count

    def mine_patterns(self, minsup_cnt, item_order):
        """
        Non-recursive pattern mining of the COFI-Tree.
        Traverses tree branches directly to extract all frequent itemsets containing
        target_item combined with locally frequent co-occurring items.
        Updates participation counters on tree nodes.
        """
        frequent_patterns = {}
        locally_frequent = [item for item, cnt in self.local_header.items() if cnt >= minsup_cnt]
        if not locally_frequent:
            return frequent_patterns

        locally_frequent.sort(key=lambda x: item_order[x])

        # Traverse the COFI-tree structure to extract unique branches from tree nodes
        branches = []
        def dfs(node, path_items, path_nodes):
            if node.branch_count > 0:
                branches.append((set(path_items), node.branch_count, list(path_nodes)))
            for child in node.children.values():
                dfs(child, path_items + [child.item], path_nodes + [child])

        dfs(self.root, [], [])

        # Mine combinations of locally frequent items directly on the extracted branches
        for r in range(1, len(locally_frequent) + 1):
            for combo in combinations(locally_frequent, r):
                combo_set = set(combo)
                supp_cnt = 0
                participating_nodes = []
                for b_items, b_cnt, b_nodes in branches:
                    if combo_set.issubset(b_items):
                        supp_cnt += b_cnt
                        participating_nodes.extend(b_nodes)

                if supp_cnt >= minsup_cnt:
                    full_pattern = frozenset(combo_set.union([self.target_item]))
                    frequent_patterns[full_pattern] = supp_cnt
                    # Update participation counters on the tree nodes
                    for node in participating_nodes:
                        if node.item in combo_set:
                            node.participation += 1

        return frequent_patterns


class COFIMiner:
    """
    End-to-end COFI (Co-occurrence Frequent Itemset) Mining Engine.
    Implements:
      1. Base FP-tree construction.
      2. Bottom-up COFI-tree building and mining.
      3. Non-recursive pattern growth with immediate tree deallocation.
      4. Closed Frequent Itemset (COFI-CLOSED) identification.
    """
    def __init__(self, transactions, minsup=0.083):
        self.transactions = transactions
        self.N = len(transactions)
        self.minsup = minsup
        self.minsup_cnt = int(np.ceil(minsup * self.N))
        self.item_counts = defaultdict(int)
        self.frequent_1 = {}
        self.sorted_items = []
        self.item_order = {}
        self.base_tree = None
        self.all_frequent_itemsets = {}
        self.closed_frequent_itemsets = {}
        self.cofi_tree_stats = []

    def build_base_tree(self):
        """Pass 1 & Pass 2: Base FP-Tree Construction."""
        # Pass 1: 1-item frequency counting
        for t in self.transactions:
            for item in t:
                self.item_counts[item] += 1

        self.frequent_1 = {item: cnt for item, cnt in self.item_counts.items() if cnt >= self.minsup_cnt}
        # Order items descending by frequency, then alphabetically for deterministic order
        self.sorted_items = sorted(self.frequent_1.keys(), key=lambda x: (self.frequent_1[x], x), reverse=True)
        self.item_order = {item: i for i, item in enumerate(self.sorted_items)}

        # Pass 2: Base FP-Tree construction
        self.base_tree = BaseFPTree()
        for item in self.sorted_items:
            self.base_tree.header_table[item] = [self.frequent_1[item], None]

        for t in self.transactions:
            filtered = [item for item in t if item in self.frequent_1]
            filtered.sort(key=lambda x: self.item_order[x])
            if filtered:
                self.base_tree.add_transaction(filtered, 1)

        print(f"[+] Base FP-Tree constructed with {len(self.base_tree.header_table)} frequent 1-items.")

    def mine(self):
        """Executes genuine COFI-Tree bottom-up mining."""
        print("\n" + "=" * 75)
        print("STEP 2: COFI-TREE BOTTOM-UP MINING EXECUTION")
        print(f"Minimum Support Threshold: {self.minsup:.4f} ({self.minsup_cnt} transactions)")
        print("=" * 75)

        start_time = time.perf_counter()
        self.build_base_tree()

        # Add 1-itemsets to frequent collection
        for item, cnt in self.frequent_1.items():
            self.all_frequent_itemsets[frozenset([item])] = cnt

        # Bottom-up processing: from least frequent item to most frequent item
        for target_item in reversed(self.sorted_items):
            target_rank = self.item_order[target_item]

            # Collect conditional pattern base from Base FP-Tree
            patterns = []
            node = self.base_tree.header_table[target_item][1]
            while node is not None:
                path = []
                p = node.parent
                while p is not None and p.item is not None:
                    # Enforce strict prefix ordering to prevent duplicate candidate generation
                    if self.item_order[p.item] < target_rank:
                        path.append(p.item)
                    p = p.parent
                if path:
                    path.reverse()
                    patterns.append((path, node.count))
                node = node.node_link

            if not patterns:
                self.cofi_tree_stats.append({
                    'item': target_item,
                    'patterns_in_cpb': 0,
                    'locally_frequent': 0,
                    'mined_patterns': 0
                })
                continue

            # Compute local co-occurrence frequencies
            local_counts = defaultdict(int)
            for path, cnt in patterns:
                for item in path:
                    local_counts[item] += cnt

            # Filter locally frequent items
            locally_frequent = {item for item, cnt in local_counts.items() if cnt >= self.minsup_cnt}
            if not locally_frequent:
                self.cofi_tree_stats.append({
                    'item': target_item,
                    'patterns_in_cpb': len(patterns),
                    'locally_frequent': 0,
                    'mined_patterns': 0
                })
                continue

            # Construct localized COFI-Tree
            cofi_tree = COFITree(target_item)
            for path, cnt in patterns:
                filtered_path = [item for item in path if item in locally_frequent]
                if filtered_path:
                    cofi_tree.add_branch(filtered_path, cnt)

            # Mine patterns directly from the COFI-Tree
            new_patterns = cofi_tree.mine_patterns(self.minsup_cnt, self.item_order)
            self.all_frequent_itemsets.update(new_patterns)

            self.cofi_tree_stats.append({
                'item': target_item,
                'patterns_in_cpb': len(patterns),
                'locally_frequent': len(locally_frequent),
                'mined_patterns': len(new_patterns)
            })

            # Explicit memory release of local COFI-Tree
            del cofi_tree

        elapsed_time = time.perf_counter() - start_time
        print(f"[+] COFI Mining completed in {elapsed_time * 1000:.2f} ms.")
        print(f"[+] Total Frequent Itemsets Discovered: {len(self.all_frequent_itemsets)}")

        # Step 2b: Closed Itemset Extraction (COFI-CLOSED)
        self.extract_closed_itemsets()
        return elapsed_time

    def extract_closed_itemsets(self):
        """
        Extracts Closed Frequent Itemsets from discovered frequent itemsets.
        Definition: An itemset X is closed iff no proper superset Y contains the exact same support:
                    supp(Y) < supp(X) for all Y > X.
        """
        print("\n" + "=" * 75)
        print("STEP 2B: CLOSED FREQUENT ITEMSET (COFI-CLOSED) FILTERING")
        print("=" * 75)

        self.closed_frequent_itemsets = {}
        for itemset, supp in self.all_frequent_itemsets.items():
            is_closed = True
            for other_itemset, other_supp in self.all_frequent_itemsets.items():
                if len(other_itemset) > len(itemset) and itemset.issubset(other_itemset):
                    if other_supp == supp:
                        is_closed = False
                        break
            if is_closed:
                self.closed_frequent_itemsets[itemset] = supp

        non_closed_count = len(self.all_frequent_itemsets) - len(self.closed_frequent_itemsets)
        print(f"[+] Total Frequent Itemsets: {len(self.all_frequent_itemsets)}")
        print(f"[+] Closed Frequent Itemsets: {len(self.closed_frequent_itemsets)}")
        print(f"[+] Non-Closed (Subsumed) Itemsets Pruned: {non_closed_count}")


# ------------------------------------------------------------------------------
# 4. STEP 3: INDEPENDENT BRUTE-FORCE VALIDATION
# ------------------------------------------------------------------------------
def run_independent_validation(transactions, minsup_cnt, cofi_all_frequent, cofi_closed):
    """
    Performs exhaustive combinatorial validation over the Forbes transactions.
    Validates:
      1. Frequent itemset equality (all 87 itemsets).
      2. Support count equality on every itemset.
      3. Closed frequent itemset equality (all 79 closed itemsets).
      4. Strict closedness property verification.
      5. Absence of false positives and false negatives.
    """
    print("\n" + "=" * 75)
    print("STEP 3: INDEPENDENT BRUTE-FORCE VALIDATION SUITE")
    print("=" * 75)

    all_items = set()
    for t in transactions:
        all_items.update(t)

    # Exhaustive combinatorial search up to max transaction size (k=1..5)
    bf_frequent = {}
    for k in range(1, 6):
        for cand in combinations(all_items, k):
            c_set = frozenset(cand)
            cnt = sum(1 for t in transactions if c_set.issubset(t))
            if cnt >= minsup_cnt:
                bf_frequent[c_set] = cnt

    print(f"[+] Brute-Force Frequent Itemsets Count: {len(bf_frequent)}")
    print(f"[+] COFI Discovered Frequent Itemsets Count: {len(cofi_all_frequent)}")

    # 1. Frequent Itemset Assertions
    assert len(bf_frequent) == len(cofi_all_frequent), "Mismatch in total frequent itemsets count!"
    for itemset, cnt in bf_frequent.items():
        assert itemset in cofi_all_frequent, f"Missing frequent itemset in COFI: {itemset}"
        assert cofi_all_frequent[itemset] == cnt, f"Support mismatch for {itemset}: COFI={cofi_all_frequent[itemset]}, BF={cnt}"
    print("    [PASS] 100% Exact match on all frequent itemsets and exact support counts.")

    # 2. Brute-Force Closed Itemset Identification
    bf_closed = {}
    for itemset, supp in bf_frequent.items():
        is_closed = True
        for other, other_supp in bf_frequent.items():
            if len(other) > len(itemset) and itemset.issubset(other):
                if other_supp == supp:
                    is_closed = False
                    break
        if is_closed:
            bf_closed[itemset] = supp

    print(f"[+] Brute-Force Closed Frequent Itemsets Count: {len(bf_closed)}")
    print(f"[+] COFI Closed Frequent Itemsets Count: {len(cofi_closed)}")

    # 3. Closed Itemset Assertions
    assert len(bf_closed) == len(cofi_closed), "Mismatch in closed frequent itemsets count!"
    for itemset, cnt in bf_closed.items():
        assert itemset in cofi_closed, f"Missing closed itemset in COFI: {itemset}"
        assert cofi_closed[itemset] == cnt, f"Support mismatch on closed itemset {itemset}"

    # 4. Closedness Property Direct Verification
    for itemset, supp in cofi_closed.items():
        for other, other_supp in bf_frequent.items():
            if len(other) > len(itemset) and itemset.issubset(other):
                assert other_supp < supp, f"Closedness violation! {itemset} has superset {other} with same support {supp}"
    print("    [PASS] 100% Exact match on all closed frequent itemsets and closedness verification.")
    print("    [PASS] Zero false positives, zero false negatives across complete candidate space.")


# ------------------------------------------------------------------------------
# 5. STEP 4: CLOSED ASSOCIATION RULE GENERATION
# ------------------------------------------------------------------------------
def generate_closed_association_rules(closed_itemsets, all_frequent_itemsets, N, min_confidence=0.50, min_lift=1.10):
    """
    Generates non-trivial, high-confidence, high-lift association rules directly
    from the closed frequent itemsets.
    Thresholds: min_confidence = 0.50, min_lift = 1.10 (aligned with DIC for comparative rigor).
    """
    print("\n" + "=" * 75)
    print("STEP 4: CLOSED ASSOCIATION RULE GENERATION")
    print(f"Parameters: min_confidence = {min_confidence:.2f} | min_lift = {min_lift:.2f}")
    print("=" * 75)

    rules = []
    seen_rules = set()

    for itemset, itemset_cnt in closed_itemsets.items():
        if len(itemset) >= 2:
            itemset_sup = itemset_cnt / N

            # Generate all non-empty proper subsets as antecedent
            for r in range(1, len(itemset)):
                for ant_tuple in combinations(itemset, r):
                    ant = frozenset(ant_tuple)
                    con = itemset - ant

                    # Both antecedent and consequent must have known frequent supports
                    ant_cnt = all_frequent_itemsets.get(ant)
                    con_cnt = all_frequent_itemsets.get(con)

                    if ant_cnt and con_cnt:
                        conf = itemset_cnt / ant_cnt
                        lift = conf / (con_cnt / N)

                        rule_signature = (ant, con)
                        if conf >= min_confidence and lift >= min_lift and rule_signature not in seen_rules:
                            seen_rules.add(rule_signature)
                            rules.append({
                                'antecedent': ", ".join(sorted(list(ant))),
                                'consequent': ", ".join(sorted(list(con))),
                                'support_count': int(itemset_cnt),
                                'support': round(itemset_sup, 4),
                                'confidence': round(conf, 4),
                                'lift': round(lift, 4),
                                'antecedent_len': len(ant),
                                'consequent_len': len(con),
                                'underlying_closed_itemset': ", ".join(sorted(list(itemset)))
                            })

    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(by=['lift', 'confidence', 'support'], ascending=False).reset_index(drop=True)
        rules_path = RESULTS_DIR / "cofi_closed_rules.csv"
        rules_df.to_csv(rules_path, index=False)
        print(f"[+] Discovered {len(rules_df)} closed association rules meeting thresholds.")
        print(f"[+] Saved rules to: {rules_path.name}")
    else:
        print("[-] No rules met thresholds.")

    return rules_df


# ------------------------------------------------------------------------------
# 6. STEP 5: VISUALIZATIONS GENERATION
# ------------------------------------------------------------------------------
def generate_visualizations(all_frequent, closed_frequent, rules_df):
    """
    Generates three publication-grade visualizations:
      1. cofi_closed_itemsets_dist.png: Itemset count by size (Frequent vs Closed) & Support distribution.
      2. cofi_rules_scatter.png: Support vs. Confidence colored/sized by Lift.
      3. cofi_top_rules_bar.png: Top 15 Closed Association Rules ranked by Lift.
    """
    print("\n" + "=" * 75)
    print("STEP 5: GENERATING PUBLICATION VISUALIZATIONS")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Visualization 1: Itemsets by Size & Support Distribution
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Size breakdown comparison
    sizes = [1, 2, 3, 4]
    freq_counts = [sum(1 for s in all_frequent if len(s) == k) for k in sizes]
    closed_counts = [sum(1 for s in closed_frequent if len(s) == k) for k in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, freq_counts, width, label='All Frequent (DIC)', color='#3498db', edgecolor='black', alpha=0.85)
    bars2 = ax1.bar(x + width/2, closed_counts, width, label='Closed Frequent (COFI)', color='#2ecc71', edgecolor='black', alpha=0.85)

    ax1.set_xlabel('Itemset Size (k)', fontweight='bold')
    ax1.set_ylabel('Number of Itemsets', fontweight='bold')
    ax1.set_title('(A) Frequent vs. Closed Itemsets by Cardinality', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in sizes])
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, int(yval), ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, int(yval), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e8449')

    # Panel B: Support Distribution of Closed Itemsets
    closed_supports = [cnt for cnt in closed_frequent.values()]
    sns.histplot(closed_supports, bins=12, kde=True, ax=ax2, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax2.axvline(25, color='#e74c3c', linestyle='--', linewidth=1.8, label='minsup threshold (25)')
    ax2.axvline(np.median(closed_supports), color='#2c3e50', linestyle='-', linewidth=1.8, label=f'Median supp ({np.median(closed_supports):.0f})')
    ax2.set_xlabel('Support Count (out of 301)', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('(B) Support Distribution of Closed Itemsets (N=79)', fontweight='bold', pad=10)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    p1_path = VIS_DIR / "cofi_closed_itemsets_dist.png"
    plt.savefig(p1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved visualization 1: {p1_path.name}")

    # -------------------------------------------------------------------------
    # Visualization 2: Rules Scatter (Support vs Confidence vs Lift)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    scatter = plt.scatter(
        rules_df['support'],
        rules_df['confidence'],
        c=rules_df['lift'],
        s=rules_df['lift'] * 55,
        cmap='viridis',
        edgecolor='black',
        alpha=0.85,
        linewidth=0.8
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Lift', fontweight='bold')

    plt.axhline(0.50, color='#e74c3c', linestyle='--', linewidth=1.2, label='min_conf (0.50)')
    plt.axvline(25/301, color='#e67e22', linestyle='--', linewidth=1.2, label='min_supp (0.083)')

    for idx in range(min(4, len(rules_df))):
        row = rules_df.iloc[idx]
        short_ant = row['antecedent'].replace('Sport=', '').replace('Decade=', '').replace('Nation=', '').replace('Earnings=', '')
        short_con = row['consequent'].replace('Sport=', '').replace('Decade=', '').replace('Nation=', '').replace('Earnings=', '')
        plt.annotate(
            f"{{{short_ant}}} => {{{short_con}}}\n(Lift={row['lift']:.2f})",
            (row['support'], row['confidence']),
            textcoords="offset points",
            xytext=(10, -10 if idx % 2 == 0 else 8),
            ha='left',
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85)
        )

    plt.xlabel('Support', fontweight='bold')
    plt.ylabel('Confidence', fontweight='bold')
    plt.title('COFI Closed Association Rules: Support vs. Confidence vs. Lift\n(56 Non-Redundant Rules; Size & Color Proportional to Lift)', fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    p2_path = VIS_DIR / "cofi_rules_scatter.png"
    plt.savefig(p2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved visualization 2: {p2_path.name}")

    # -------------------------------------------------------------------------
    # Visualization 3: Top 15 Rules Bar Chart
    # -------------------------------------------------------------------------
    top15 = rules_df.head(15).iloc[::-1]
    plt.figure(figsize=(11, 7), dpi=300)
    rule_labels = [f"{{{r['antecedent']}}} => {{{r['consequent']}}}" for _, r in top15.iterrows()]
    bars = plt.barh(rule_labels, top15['lift'], color='#16a085', edgecolor='black', alpha=0.85)

    plt.axvline(1.0, color='#c0392b', linestyle='--', linewidth=1.5, label='Independence Baseline (Lift = 1.0)')
    plt.xlabel('Lift Metric', fontweight='bold')
    plt.title('Top 15 Closed Association Rules Ranked by Lift\n(COFI-Closed Mining on Forbes Athletes Dataset)', fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for bar, (_, r) in zip(bars, top15.iterrows()):
        plt.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height()/2.0,
            f"Lift={r['lift']:.2f} | Conf={r['confidence']*100:.1f}% | Supp={r['support_count']}",
            ha='left', va='center', fontsize=8.5, fontweight='bold', color='#2c3e50'
        )

    plt.xlim(0, max(top15['lift']) + 0.8)
    plt.legend(loc='lower right', frameon=True, facecolor='white')
    plt.tight_layout()
    p3_path = VIS_DIR / "cofi_top_rules_bar.png"
    plt.savefig(p3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved visualization 3: {p3_path.name}")


# ------------------------------------------------------------------------------
# 7. STEP 6: OUTPUT FILE GENERATION & SUMMARY EXPORT
# ------------------------------------------------------------------------------
def export_results(miner, rules_df, elapsed_time):
    """Exports structured results CSV and JSON summary metadata."""
    print("\n" + "=" * 75)
    print("STEP 6: EXPORTING STRUCTURED ARTIFACTS")
    print("=" * 75)

    # 1. Closed Itemsets CSV
    closed_records = []
    for itemset, cnt in miner.closed_frequent_itemsets.items():
        closed_records.append({
            'itemset': "; ".join(sorted(list(itemset))),
            'size': len(itemset),
            'support_count': cnt,
            'support': round(cnt / miner.N, 4),
            'is_closed': True
        })
    closed_df = pd.DataFrame(closed_records)
    closed_df = closed_df.sort_values(by=['size', 'support_count'], ascending=[True, False]).reset_index(drop=True)
    closed_path = RESULTS_DIR / "cofi_closed_itemsets.csv"
    closed_df.to_csv(closed_path, index=False)
    print(f"[+] Saved {len(closed_df)} closed itemsets to: {closed_path.name}")

    # 2. Execution Summary JSON
    summary = {
        'algorithm': 'COFI (Co-occurrence Frequent Itemset) Closed Mining',
        'dataset': 'Forbes Richest Athletes 1990-2020 Cleaned',
        'transaction_count': miner.N,
        'unique_items': len(miner.item_counts),
        'frequent_1_items': len(miner.frequent_1),
        'minsup_threshold': miner.minsup,
        'minsup_count': miner.minsup_cnt,
        'min_confidence': 0.50,
        'min_lift': 1.10,
        'execution_time_ms': round(elapsed_time * 1000, 2),
        'total_frequent_itemsets': len(miner.all_frequent_itemsets),
        'closed_frequent_itemsets': len(miner.closed_frequent_itemsets),
        'non_closed_subsumed_itemsets': len(miner.all_frequent_itemsets) - len(miner.closed_frequent_itemsets),
        'closed_itemsets_by_size': {
            f"size_{k}": sum(1 for s in miner.closed_frequent_itemsets if len(s) == k)
            for k in range(1, 5)
        },
        'total_closed_rules': len(rules_df),
        'dic_total_rules': 61,
        'subsumed_rules_pruned': 61 - len(rules_df),
        'independent_validation': {
            'brute_force_combinations_evaluated': 44551,
            'frequent_itemsets_exact_match': True,
            'closed_itemsets_exact_match': True,
            'support_counts_exact_match': True,
            'closedness_property_verified': True,
            'zero_false_positives': True,
            'zero_false_negatives': True
        }
    }

    summary_path = RESULTS_DIR / "cofi_execution_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"[+] Saved execution metadata to: {summary_path.name}")
    return summary


# ------------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("STARTING COFI CLOSED FREQUENT ITEMSET & ASSOCIATION MINING")
    print("=" * 75)

    transactions, unique_items = load_transactions()
    miner = COFIMiner(transactions, minsup=0.083)
    elapsed_time = miner.mine()

    # Step 3: Run Validation Suite
    run_independent_validation(
        transactions,
        miner.minsup_cnt,
        miner.all_frequent_itemsets,
        miner.closed_frequent_itemsets
    )

    # Step 4: Closed Association Rules
    rules_df = generate_closed_association_rules(
        miner.closed_frequent_itemsets,
        miner.all_frequent_itemsets,
        miner.N,
        min_confidence=0.50,
        min_lift=1.10
    )

    # Step 5: Visualizations
    generate_visualizations(miner.all_frequent_itemsets, miner.closed_frequent_itemsets, rules_df)

    # Step 6: Export Results & Summary
    summary = export_results(miner, rules_df, elapsed_time)

    print("\n" + "=" * 75)
    print("COFI EXPERIMENT SUCCESSFULLY COMPLETED")
    print(f"Total Frequent: {summary['total_frequent_itemsets']} | Closed Frequent: {summary['closed_frequent_itemsets']}")
    print(f"Closed Rules: {summary['total_closed_rules']} | Subsumed Rules Pruned: {summary['subsumed_rules_pruned']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
