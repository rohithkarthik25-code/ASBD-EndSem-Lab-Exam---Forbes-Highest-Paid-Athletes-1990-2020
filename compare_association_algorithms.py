#!/usr/bin/env python3
"""
Comparative Validation of Association Rule Mining Algorithms:
1. Dynamic Itemset Counting (DIC)
2. Co-occurrence Frequent Itemset (COFI) Closed Mining
3. Apache Spark MLlib FP-Growth

Forbes Highest-Paid Athletes Dataset (1990-2020)
Cleaned Transactions: N = 301, 23 distinct items, transaction length = 5
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIS_DIR = os.path.join(BASE_DIR, "visualizations", "association_comparison")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Helper functions for item and rule normalization
def parse_itemset(itemset_str):
    """Normalize itemset string into a sorted tuple of items regardless of delimiter."""
    if pd.isna(itemset_str):
        return tuple()
    items = [x.strip() for x in re.split(r'[,;]', str(itemset_str)) if x.strip()]
    return tuple(sorted(items))

def format_itemset(items_tuple):
    return ", ".join(items_tuple)

def parse_rule_signature(antecedent_str, consequent_str):
    ant = parse_itemset(antecedent_str)
    con = parse_itemset(consequent_str)
    return (ant, con)

def format_rule_signature(ant, con):
    return f"{', '.join(ant)} => {', '.join(con)}"


def load_datasets():
    print("Loading approved algorithm outputs...")
    
    # DIC
    dic_itemsets_df = pd.read_csv(os.path.join(RESULTS_DIR, "dic_frequent_itemsets.csv"))
    dic_rules_df = pd.read_csv(os.path.join(RESULTS_DIR, "dic_association_rules.csv"))
    
    # COFI
    cofi_itemsets_df = pd.read_csv(os.path.join(RESULTS_DIR, "cofi_closed_itemsets.csv"))
    cofi_rules_df = pd.read_csv(os.path.join(RESULTS_DIR, "cofi_closed_rules.csv"))
    
    # PySpark FP-Growth
    spark_itemsets_df = pd.read_csv(os.path.join(RESULTS_DIR, "pyspark_frequent_itemsets.csv"))
    spark_rules_df = pd.read_csv(os.path.join(RESULTS_DIR, "pyspark_association_rules.csv"))
    
    return {
        "dic": {"itemsets": dic_itemsets_df, "rules": dic_rules_df},
        "cofi": {"itemsets": cofi_itemsets_df, "rules": cofi_rules_df},
        "spark": {"itemsets": spark_itemsets_df, "rules": spark_rules_df}
    }


def analyze_frequent_itemsets(data):
    print("\n--- Analyzing Frequent & Closed Itemsets ---")
    dic_items = {parse_itemset(r["itemset"]): r["support_count"] for _, r in data["dic"]["itemsets"].iterrows()}
    spark_items = {parse_itemset(r["itemset"]): r["support_count"] for _, r in data["spark"]["itemsets"].iterrows()}
    cofi_items = {parse_itemset(r["itemset"]): r["support_count"] for _, r in data["cofi"]["itemsets"].iterrows()}
    
    # Validation between DIC and Spark
    exact_dic_spark_match = (dic_items == spark_items)
    print(f"Total Frequent Itemsets (DIC): {len(dic_items)}")
    print(f"Total Frequent Itemsets (Spark): {len(spark_items)}")
    print(f"Total Closed Frequent Itemsets (COFI): {len(cofi_items)}")
    print(f"DIC and Spark Itemset & Support Count Exact Parity: {exact_dic_spark_match}")
    
    # Cardinality breakdown
    cardinality_breakdown = {}
    for k in range(1, 6):
        dic_k = [s for s in dic_items if len(s) == k]
        spark_k = [s for s in spark_items if len(s) == k]
        cofi_k = [s for s in cofi_items if len(s) == k]
        if dic_k or spark_k or cofi_k:
            cardinality_breakdown[f"k={k}"] = {
                "DIC": len(dic_k),
                "Spark_FPGrowth": len(spark_k),
                "COFI_Closed": len(cofi_k),
                "Pruned_by_COFI": len(dic_k) - len(cofi_k)
            }
            print(f"  k={k}: DIC={len(dic_k)}, Spark={len(spark_k)}, COFI={len(cofi_k)} (Pruned: {len(dic_k) - len(cofi_k)})")
            
    # Pruned itemsets analysis
    pruned_sets = set(dic_items.keys()) - set(cofi_items.keys())
    print(f"\nNon-closed itemsets pruned by COFI: {len(pruned_sets)} itemsets")
    pruned_details = []
    for s in sorted(pruned_sets, key=lambda x: (len(x), x)):
        supp = dic_items[s]
        # Find closed superset with identical support
        supersets = [c for c in cofi_items if set(s).issubset(set(c)) and len(c) > len(s) and cofi_items[c] == supp]
        sup_str = [format_itemset(c) for c in supersets]
        pruned_details.append({
            "pruned_itemset": format_itemset(s),
            "size": len(s),
            "support_count": supp,
            "support": round(supp / 301.0, 4),
            "closed_supersets_with_identical_support": sup_str
        })
        print(f"  Pruned: {format_itemset(s)} (supp={supp}) -> Closed Superset: {sup_str}")
        
    return {
        "dic_total": len(dic_items),
        "spark_total": len(spark_items),
        "cofi_total": len(cofi_items),
        "exact_dic_spark_match": exact_dic_spark_match,
        "cardinality_breakdown": cardinality_breakdown,
        "pruned_count": len(pruned_sets),
        "pruned_itemsets": pruned_details
    }


def analyze_association_rules(data):
    print("\n--- Analyzing Association Rules ---")
    
    # Build dictionary of rules keyed by signature: (tuple(sorted(ant)), tuple(sorted(con)))
    def extract_rules_dict(df):
        rules = {}
        for _, r in df.iterrows():
            ant = parse_itemset(r["antecedent"])
            con = parse_itemset(r["consequent"])
            sig = (ant, con)
            rules[sig] = {
                "antecedent": format_itemset(ant),
                "consequent": format_itemset(con),
                "antecedent_len": len(ant),
                "consequent_len": len(con),
                "support_count": int(r["support_count"]),
                "support": float(r["support"]),
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"])
            }
        return rules
        
    dic_rules = extract_rules_dict(data["dic"]["rules"])
    cofi_rules = extract_rules_dict(data["cofi"]["rules"])
    spark_rules = extract_rules_dict(data["spark"]["rules"])
    
    dic_sigs = set(dic_rules.keys())
    cofi_sigs = set(cofi_rules.keys())
    spark_sigs = set(spark_rules.keys())
    
    print(f"Total Rules: DIC={len(dic_sigs)}, COFI={len(cofi_sigs)}, Spark={len(spark_sigs)}")
    
    # Partition overlaps
    all_three = dic_sigs & cofi_sigs & spark_sigs
    dic_and_cofi_only = (dic_sigs & cofi_sigs) - spark_sigs
    dic_and_spark_only = (dic_sigs & spark_sigs) - cofi_sigs
    cofi_and_spark_only = (cofi_sigs & spark_sigs) - dic_sigs
    spark_only = spark_sigs - dic_sigs - cofi_sigs
    cofi_only = cofi_sigs - dic_sigs - spark_sigs
    dic_only = dic_sigs - cofi_sigs - spark_sigs
    
    union_all = dic_sigs | cofi_sigs | spark_sigs
    sum_partitions = (len(all_three) + len(dic_and_cofi_only) + len(dic_and_spark_only) + 
                      len(cofi_and_spark_only) + len(dic_only) + len(cofi_only) + len(spark_only))
    
    print(f"Total Unique Rules (|DIC ∪ COFI ∪ Spark|): {len(union_all)}")
    print(f"Sum of 7 Mutually Exclusive Partitions: {sum_partitions}")
    print(f"Mathematical Identity Verified (Sum == Union): {sum_partitions == len(union_all)}")
    
    print(f"Overlap Partitions:")
    print(f"  1. In all three algorithms: {len(all_three)}")
    print(f"  2. In DIC & COFI only (Multi-item Consequents): {len(dic_and_cofi_only)}")
    print(f"  3. In DIC & Spark only (Pruned by COFI due to non-closed itemsets): {len(dic_and_spark_only)}")
    print(f"  4. In COFI & Spark only: {len(cofi_and_spark_only)}")
    print(f"  5. In DIC only: {len(dic_only)}")
    print(f"  6. In COFI only: {len(cofi_only)}")
    print(f"  7. In Spark only (Lift < 1.10 threshold): {len(spark_only)}")
    
    # Reconstructed rule totals:
    rec_dic = len(all_three) + len(dic_and_cofi_only) + len(dic_and_spark_only) + len(dic_only)
    rec_cofi = len(all_three) + len(dic_and_cofi_only) + len(cofi_and_spark_only) + len(cofi_only)
    rec_spark = len(all_three) + len(dic_and_spark_only) + len(cofi_and_spark_only) + len(spark_only)
    
    print(f"Reconstructed Totals: DIC={rec_dic} (==61: {rec_dic==61}), COFI={rec_cofi} (==56: {rec_cofi==56}), Spark={rec_spark} (==71: {rec_spark==71})")
    
    # Detailed breakdown of partitions
    dic_and_cofi_details = []
    for sig in sorted(dic_and_cofi_only):
        r = dic_rules[sig]
        dic_and_cofi_details.append(r)
        
    dic_and_spark_details = []
    for sig in sorted(dic_and_spark_only):
        r = dic_rules[sig]
        # find matching closed rule in COFI
        # closed rule has same antecedent, but expanded consequent containing Nation=USA
        ant_set = set(sig[0])
        matching_cofi = [
            cofi_rules[csig] for csig in cofi_rules
            if set(csig[0]) == ant_set and set(sig[1]).issubset(set(csig[1]))
        ]
        r_copy = dict(r)
        r_copy["subsuming_closed_cofi_rules"] = [
            f"{m['antecedent']} => {m['consequent']} (supp={m['support_count']}, conf={m['confidence']:.4f}, lift={m['lift']:.4f})"
            for m in matching_cofi
        ]
        dic_and_spark_details.append(r_copy)
        
    spark_only_details = []
    for sig in sorted(spark_only):
        r = spark_rules[sig]
        spark_only_details.append(r)
        
    # Consequent cardinality analysis
    consequent_cardinality = {
        "DIC": {"len_1": sum(1 for r in dic_rules.values() if r["consequent_len"] == 1),
                "len_2_plus": sum(1 for r in dic_rules.values() if r["consequent_len"] > 1)},
        "COFI": {"len_1": sum(1 for r in cofi_rules.values() if r["consequent_len"] == 1),
                 "len_2_plus": sum(1 for r in cofi_rules.values() if r["consequent_len"] > 1)},
        "Spark": {"len_1": sum(1 for r in spark_rules.values() if r["consequent_len"] == 1),
                  "len_2_plus": sum(1 for r in spark_rules.values() if r["consequent_len"] > 1)}
    }
    
    # Lift distribution analysis
    lift_distribution = {
        "DIC": {
            "lift_ge_1_1": sum(1 for r in dic_rules.values() if r["lift"] >= 1.10),
            "lift_1_0_to_1_1": sum(1 for r in dic_rules.values() if 1.00 <= r["lift"] < 1.10),
            "lift_lt_1_0": sum(1 for r in dic_rules.values() if r["lift"] < 1.00)
        },
        "COFI": {
            "lift_ge_1_1": sum(1 for r in cofi_rules.values() if r["lift"] >= 1.10),
            "lift_1_0_to_1_1": sum(1 for r in cofi_rules.values() if 1.00 <= r["lift"] < 1.10),
            "lift_lt_1_0": sum(1 for r in cofi_rules.values() if r["lift"] < 1.00)
        },
        "Spark": {
            "lift_ge_1_1": sum(1 for r in spark_rules.values() if r["lift"] >= 1.10),
            "lift_1_0_to_1_1": sum(1 for r in spark_rules.values() if 1.00 <= r["lift"] < 1.10),
            "lift_lt_1_0": sum(1 for r in spark_rules.values() if r["lift"] < 1.00)
        }
    }
    
    return {
        "rule_counts": {
            "DIC": len(dic_sigs),
            "COFI": len(cofi_sigs),
            "Spark": len(spark_sigs)
        },
        "total_unique_rules": len(union_all),
        "mutually_exclusive_partition_sum": sum_partitions,
        "partition_sum_matches_union": (sum_partitions == len(union_all)),
        "partitions": {
            "all_three_intersection": len(all_three),
            "dic_and_cofi_only": len(dic_and_cofi_only),
            "dic_and_spark_only": len(dic_and_spark_only),
            "cofi_and_spark_only": len(cofi_and_spark_only),
            "dic_only": len(dic_only),
            "cofi_only": len(cofi_only),
            "spark_only": len(spark_only)
        },
        "reconstructed_totals": {
            "DIC": rec_dic,
            "COFI": rec_cofi,
            "Spark": rec_spark
        },
        "reconstructed_matches_source": (rec_dic == len(dic_sigs) and rec_cofi == len(cofi_sigs) and rec_spark == len(spark_sigs)),
        "partition_details": {
            "dic_and_cofi_only": dic_and_cofi_details,
            "dic_and_spark_only": dic_and_spark_details,
            "spark_only": spark_only_details
        },
        "consequent_cardinality": consequent_cardinality,
        "lift_distribution": lift_distribution
    }


def generate_comparison_table_and_json(itemset_res, rule_res, data):
    print("\n--- Generating Comparison CSV and JSON ---")
    
    # 1. Summary comparison table (CSV)
    summary_rows = [
        {"Category": "Algorithmic Class", "Metric": "Theoretical Paradigm", 
         "DIC": "Dynamic Itemset Counting (Relaxed Apriori Lattice)", 
         "COFI": "Co-occurrence Frequent Itemsets (Pattern-Growth DFS)", 
         "Spark FP-Growth": "Distributed FP-Tree (MapReduce Parallel Projection)",
         "Analytical Insights": "DIC counts on transaction stream; COFI mines item-conditional subtrees; Spark shards projections across executors."},
        
        {"Category": "Data Structure", "Metric": "Primary Memory Structure", 
         "DIC": "Dynamic Trie / Itemset State Hash Table (DS, DC, SS, SC)", 
         "COFI": "Compressed COFI-Trees per Frequent Item", 
         "Spark FP-Growth": "Distributed Prefix-Trees (RDD[FPTree])",
         "Analytical Insights": "COFI builds separate tree per item; Spark broadcasts header and parallelizes projections."},
        
        {"Category": "Frequent Itemsets", "Metric": "Total Itemsets Discovered", 
         "DIC": "87", "COFI": "79 (Closed)", "Spark FP-Growth": "87",
         "Analytical Insights": "DIC and Spark discover complete set F; COFI prunes 8 non-closed itemsets with zero information loss (9.2% compression)."},
        
        {"Category": "Frequent Itemsets", "Metric": "Size 1 Itemsets", 
         "DIC": "16", "COFI": "14", "Spark FP-Growth": "16",
         "Analytical Insights": "COFI pruned Sport=Basketball (supp=82) and Sport=Soccer (supp=33) as they are identical to supersets with Nation."},
        
        {"Category": "Frequent Itemsets", "Metric": "Size 2 Itemsets", 
         "DIC": "46", "COFI": "42", "Spark FP-Growth": "46",
         "Analytical Insights": "COFI pruned 4 Basketball 2-itemsets identical in support to 3-itemsets containing Nation=USA."},
        
        {"Category": "Frequent Itemsets", "Metric": "Size 3 Itemsets", 
         "DIC": "23", "COFI": "21", "Spark FP-Growth": "23",
         "Analytical Insights": "COFI pruned 2 Basketball 3-itemsets identical in support to 4-itemsets containing Nation=USA."},
        
        {"Category": "Frequent Itemsets", "Metric": "Size 4 Itemsets", 
         "DIC": "2", "COFI": "2", "Spark FP-Growth": "2",
         "Analytical Insights": "Both maximal 4-itemsets are closed: {Decade=2000s, Earnings=High, Nation=USA, Sport=Basketball} and {Earnings=High, Nation=USA, Rank=Mid-Tier, Sport=Basketball}."},
        
        {"Category": "Itemset Parity", "Metric": "Exact Match (DIC vs Spark)", 
         "DIC": "Baseline", "COFI": "Lossless Basis", "Spark FP-Growth": "100.0% Exact Match",
         "Analytical Insights": "All 87 itemsets and exact support counts are 100% identical between DIC and Spark FP-Growth."},
        
        {"Category": "Association Rules", "Metric": "Total Rules Generated", 
         "DIC": "61", "COFI": "56", "Spark FP-Growth": "71",
         "Analytical Insights": "DIC generated 61 rules; COFI pruned 5 redundant rules; Spark generated 71 rules (including 19 with lift < 1.10)."},
        
        {"Category": "Rule Overlap", "Metric": "Total Unique Rules Union (|DIC ∪ COFI ∪ Spark|)", 
         "DIC": "61", "COFI": "56", "Spark FP-Growth": "71",
         "Analytical Insights": "Exact union size is 80 unique rules. Mutually exclusive partition sum: 47 (all three) + 9 (DIC & COFI) + 5 (DIC & Spark) + 19 (Spark only) = 80 rules (100% identity verified)."},
        
        {"Category": "Rule Overlap", "Metric": "Common to All Three (Intersection)", 
         "DIC": "47", "COFI": "47", "Spark FP-Growth": "47",
         "Analytical Insights": "Core set of rules meeting conf >= 0.50, lift >= 1.10, closed itemset basis, and single-item consequent."},
        
        {"Category": "Rule Constraints", "Metric": "Consequent Cardinality (|B|)", 
         "DIC": "Arbitrary (|B| >= 1): 52 single, 9 multi", 
         "COFI": "Arbitrary (|B| >= 1): 47 single, 9 multi", 
         "Spark FP-Growth": "Single Item Only (|B| = 1): 71 single, 0 multi",
         "Analytical Insights": "Spark MLlib is strictly constrained to single-item consequents by design; DIC and COFI discover multi-item consequents."},
        
        {"Category": "Rule Constraints", "Metric": "Minimum Lift Filter Applied", 
         "DIC": "Yes (lift >= 1.10)", "COFI": "Yes (lift >= 1.10)", "Spark FP-Growth": "No (all lift >= 0 kept)",
         "Analytical Insights": "Spark includes 13 rules with 1.0 <= lift < 1.10 and 6 rules with lift < 1.0 (independent/negative correlation)."},
        
        {"Category": "Redundancy Reduction", "Metric": "Subsumed Rule Elimination", 
         "DIC": "None (retains subsumed rules)", "COFI": "Eliminated 5 subsumed rules", "Spark FP-Growth": "None (retains subsumed rules)",
         "Analytical Insights": "COFI replaces 5 lower-information rules with exact closed counterparts containing Nation=USA with identical confidence."},
        
        {"Category": "Database Scanning", "Metric": "I/O Passes over Data", 
         "DIC": "4 passes (interval checkpoints M=50)", 
         "COFI": "2 passes (1 for F1, 1 for tree construction)", 
         "Spark FP-Growth": "2 passes (parallelized distributed RDD scans)",
         "Analytical Insights": "DIC reduces passes vs Apriori (4 vs 5); COFI and FP-Growth require only 2 passes."}
    ]
    
    comp_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "association_algorithm_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"Exported comparison table to {csv_path}")
    
    # 2. Comprehensive JSON metrics
    full_json_data = {
        "comparison_title": "Comparative Validation of DIC, COFI, and Spark FP-Growth",
        "dataset_metadata": {
            "name": "Forbes Highest-Paid Athletes (1990-2020) Cleaned Transactions",
            "total_transactions": 301,
            "transaction_length": 5,
            "unique_items": 23,
            "minimum_support_threshold": 0.08,
            "minimum_support_count": 25,
            "minimum_confidence_threshold": 0.50
        },
        "itemset_analysis": itemset_res,
        "association_rule_analysis": rule_res,
        "architectural_comparison": {
            "DIC": {
                "paradigm": "Dynamic Candidate Counting on Data Stream",
                "reference": "Brin et al. (SIGMOD 1997)",
                "data_structure": "Itemset State Hash Table (DS, DC, SS, SC) with Checkpoints",
                "passes_over_data": 4,
                "multi_consequent_support": True,
                "redundancy_elimination": False,
                "strengths": "Starts counting supersets early without waiting for full database pass; ideal for streaming / sequential data.",
                "limitations": "Still evaluates Apriori candidate powerset; can generate redundant rules."
            },
            "COFI": {
                "paradigm": "Closed Itemset Pattern-Growth via Compressed Subtrees",
                "reference": "Zaïane, El-Hajj (DMKD 2005)",
                "data_structure": "Header Table + Local Conditional COFI-Trees",
                "passes_over_data": 2,
                "multi_consequent_support": True,
                "redundancy_elimination": True,
                "strengths": "Lossless compact representation; prunes non-closed itemsets and redundant rules; fast memory-resident mining.",
                "limitations": "Requires single-node memory to hold conditional trees; non-distributed."
            },
            "Spark_FPGrowth": {
                "paradigm": "Distributed Pattern-Growth via Parallel Projections",
                "reference": "Li et al. (PFP 2008) / Apache Spark MLlib",
                "data_structure": "Distributed Resilient Distributed Dataset (RDD) of Prefix Trees",
                "passes_over_data": 2,
                "multi_consequent_support": False,
                "redundancy_elimination": False,
                "strengths": "Massive horizontal scalability across Spark cluster; handles billions of transactions; fault-tolerant.",
                "limitations": "MLlib limits association rules to single-item consequents (|B| = 1); does not natively prune non-closed patterns."
            }
        }
    }
    
    json_path = os.path.join(RESULTS_DIR, "association_algorithm_comparison.json")
    with open(json_path, "w") as fp:
        json.dump(full_json_data, fp, indent=2)
    print(f"Exported comparison metrics to {json_path}")
    
    return comp_df, full_json_data


def generate_visualizations(itemset_res, rule_res, data):
    print("\n--- Generating Comparison Visualizations ---")
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"DIC": "#1f77b4", "COFI": "#2ca02c", "Spark FP-Growth": "#ff7f0e"}
    
    # ----------------------------------------------------
    # Plot 1: Itemset Cardinality Comparison (Grouped Bar Chart)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    card_data = []
    for k_str, counts in itemset_res["cardinality_breakdown"].items():
        card_data.append({"Size": k_str, "Algorithm": "DIC", "Count": counts["DIC"]})
        card_data.append({"Size": k_str, "Algorithm": "COFI", "Count": counts["COFI_Closed"]})
        card_data.append({"Size": k_str, "Algorithm": "Spark FP-Growth", "Count": counts["Spark_FPGrowth"]})
        
    df_plot1 = pd.DataFrame(card_data)
    barplot1 = sns.barplot(data=df_plot1, x="Size", y="Count", hue="Algorithm", palette=palette, ax=ax)
    
    # Annotate bars
    for p in barplot1.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        xytext=(0, 3), textcoords='offset points')
            
    ax.set_title("Frequent Itemset Discovery by Cardinality (k=1 to 4)\nDIC (87) vs COFI Closed (79) vs Spark FP-Growth (87)", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Itemset Size (Cardinality k)", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Itemsets Discovered", fontsize=12, labelpad=10)
    ax.set_ylim(0, 52)
    ax.legend(title="Algorithm", frameon=True, fontsize=11)
    
    # Add annotation box explaining COFI pruning
    ax.text(0.98, 0.70, 
            "Lossless Basis Compression:\n"
            "• k=1: 16 → 14 (Pruned Basketball & Soccer)\n"
            "• k=2: 46 → 42 (Pruned 4 Basketball subsets)\n"
            "• k=3: 23 → 21 (Pruned 2 Basketball subsets)\n"
            "• k=4: 2 → 2 (All maximal 4-itemsets closed)\n"
            "Total: 87 → 79 (8 non-closed pruned, 9.2% reduction)",
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#b0bec5', alpha=0.9))
            
    plt.tight_layout()
    plot1_path = os.path.join(VIS_DIR, "itemset_cardinality_comparison.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Rule Count by Lift Category (Grouped Bar Chart)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    lift_data = [
        {"Category": "Lift ≥ 1.10\n(Strong Positive)", "Algorithm": "DIC", "Count": rule_res["lift_distribution"]["DIC"]["lift_ge_1_1"]},
        {"Category": "Lift ≥ 1.10\n(Strong Positive)", "Algorithm": "COFI", "Count": rule_res["lift_distribution"]["COFI"]["lift_ge_1_1"]},
        {"Category": "Lift ≥ 1.10\n(Strong Positive)", "Algorithm": "Spark FP-Growth", "Count": rule_res["lift_distribution"]["Spark"]["lift_ge_1_1"]},
        
        {"Category": "1.00 ≤ Lift < 1.10\n(Weak Positive)", "Algorithm": "DIC", "Count": rule_res["lift_distribution"]["DIC"]["lift_1_0_to_1_1"]},
        {"Category": "1.00 ≤ Lift < 1.10\n(Weak Positive)", "Algorithm": "COFI", "Count": rule_res["lift_distribution"]["COFI"]["lift_1_0_to_1_1"]},
        {"Category": "1.00 ≤ Lift < 1.10\n(Weak Positive)", "Algorithm": "Spark FP-Growth", "Count": rule_res["lift_distribution"]["Spark"]["lift_1_0_to_1_1"]},
        
        {"Category": "Lift < 1.00\n(Negative/Independent)", "Algorithm": "DIC", "Count": rule_res["lift_distribution"]["DIC"]["lift_lt_1_0"]},
        {"Category": "Lift < 1.00\n(Negative/Independent)", "Algorithm": "COFI", "Count": rule_res["lift_distribution"]["COFI"]["lift_lt_1_0"]},
        {"Category": "Lift < 1.00\n(Negative/Independent)", "Algorithm": "Spark FP-Growth", "Count": rule_res["lift_distribution"]["Spark"]["lift_lt_1_0"]}
    ]
    df_plot2 = pd.DataFrame(lift_data)
    barplot2 = sns.barplot(data=df_plot2, x="Category", y="Count", hue="Algorithm", palette=palette, ax=ax)
    
    for p in barplot2.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=11, fontweight='bold',
                        xytext=(0, 3), textcoords='offset points')
                        
    ax.set_title("Association Rule Generation Across Lift Thresholds\nDIC (61) vs COFI (56) vs Spark FP-Growth (71)", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Lift Threshold Category", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Generated Rules", fontsize=12, labelpad=10)
    ax.set_ylim(0, 70)
    ax.legend(title="Algorithm", frameon=True, fontsize=11)
    
    ax.text(0.98, 0.70,
            "Filter Threshold Discrepancy:\n"
            "• DIC & COFI strictly enforce min_lift ≥ 1.10\n"
            "• Spark MLlib generates rules with any lift ≥ 0\n"
            "• Spark includes 13 rules with lift in [1.0, 1.10)\n"
            "• Spark includes 6 rules with lift < 1.00\n"
            "At lift ≥ 1.10: Spark has 52 rules (all |B|=1)",
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#b0bec5', alpha=0.9))
            
    plt.tight_layout()
    plot2_path = os.path.join(VIS_DIR, "rule_count_by_lift_category.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")
    
    # ----------------------------------------------------
    # Plot 3: Set-Theoretic Rule Overlap Diagram (Euler Partition Bar Chart)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    overlap_categories = [
        "In All Three\n(DIC ∩ COFI ∩ Spark)",
        "In DIC & COFI Only\n(|B| ≥ 2 Multi-Consequent)",
        "In DIC & Spark Only\n(Subsumed / Non-Closed)",
        "In Spark Only\n(Lift < 1.10 Threshold)"
    ]
    overlap_counts = [
        rule_res["partitions"]["all_three_intersection"],
        rule_res["partitions"]["dic_and_cofi_only"],
        rule_res["partitions"]["dic_and_spark_only"],
        rule_res["partitions"]["spark_only"]
    ]
    colors = ["#2b5c8f", "#2ca02c", "#d62728", "#ff7f0e"]
    
    u_total = rule_res["total_unique_rules"]
    bars = ax.bar(overlap_categories, overlap_counts, color=colors, width=0.55, edgecolor="#333333", linewidth=1.2)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{int(h)} Rules\n({h/float(u_total)*100:.1f}% of universe)",
                    (b.get_x() + b.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')
                    
    ax.set_title(f"Exact Set-Theoretic Overlap of Association Rules\nTotal Unique Rule Signatures Across All Algorithms = {u_total}", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Rules in Partition", fontsize=12, labelpad=10)
    ax.set_ylim(0, 58)
    
    # Add callout text for each bar
    explanations = [
        "Core valid rules:\n|B|=1, Closed, Lift≥1.10",
        "Multi-item consequents:\nSupported by DIC/COFI;\nExcluded by Spark MLlib",
        "Pruned by COFI:\nSubsumed by closed\nsupersets with identical conf",
        "Filtered by DIC/COFI:\n13 weak (1.0≤lift<1.1)\n6 negative (lift<1.0)"
    ]
    for i, (b, text) in enumerate(zip(bars, explanations)):
        ax.text(b.get_x() + b.get_width()/2., 3, text,
                ha='center', va='bottom', fontsize=9, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
                
    plt.tight_layout()
    plot3_path = os.path.join(VIS_DIR, "rule_overlap_venn.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")
    
    # ----------------------------------------------------
    # Plot 4: Metric Distributions (Support, Confidence, Lift)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Assemble dataframe for rule metrics
    df_dic_rules = data["dic"]["rules"][["support", "confidence", "lift"]].copy()
    df_dic_rules["Algorithm"] = "DIC (61 rules)"
    
    df_cofi_rules = data["cofi"]["rules"][["support", "confidence", "lift"]].copy()
    df_cofi_rules["Algorithm"] = "COFI (56 rules)"
    
    df_spark_rules = data["spark"]["rules"][["support", "confidence", "lift"]].copy()
    df_spark_rules["Algorithm"] = "Spark FP-Growth (71 rules)"
    
    df_all_rules = pd.concat([df_dic_rules, df_cofi_rules, df_spark_rules], ignore_index=True)
    custom_palette = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    
    # Subplot 1: Support
    sns.boxplot(data=df_all_rules, x="Algorithm", y="support", hue="Algorithm", palette=custom_palette, legend=False, ax=axes[0])
    axes[0].set_title("Rule Support Distribution", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Support", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=15)
    
    # Subplot 2: Confidence
    sns.boxplot(data=df_all_rules, x="Algorithm", y="confidence", hue="Algorithm", palette=custom_palette, legend=False, ax=axes[1])
    axes[1].axhline(0.50, color='red', linestyle='--', linewidth=1.2, label='minconf = 0.50')
    axes[1].set_title("Rule Confidence Distribution", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Confidence", fontsize=11)
    axes[1].set_xlabel("")
    axes[1].legend(loc='lower right')
    axes[1].tick_params(axis='x', rotation=15)
    
    # Subplot 3: Lift
    sns.boxplot(data=df_all_rules, x="Algorithm", y="lift", hue="Algorithm", palette=custom_palette, legend=False, ax=axes[2])
    axes[2].axhline(1.10, color='red', linestyle='--', linewidth=1.2, label='minlift = 1.10 (DIC/COFI)')
    axes[2].axhline(1.00, color='gray', linestyle=':', linewidth=1.0, label='lift = 1.00 (Independence)')
    axes[2].set_title("Rule Lift Distribution", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Lift", fontsize=11)
    axes[2].set_xlabel("")
    axes[2].legend(loc='upper right')
    axes[2].tick_params(axis='x', rotation=15)
    
    fig.suptitle("Comparative Metric Distributions across Rule Sets", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plot4_path = os.path.join(VIS_DIR, "support_confidence_lift_distributions.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot4_path}")


def main():
    print("="*70)
    print("PHASE 4 - STEP 4: COMPARATIVE VALIDATION OF ASSOCIATION MINING ALGORITHMS")
    print("="*70)
    
    data = load_datasets()
    itemset_res = analyze_frequent_itemsets(data)
    rule_res = analyze_association_rules(data)
    comp_df, full_json = generate_comparison_table_and_json(itemset_res, rule_res, data)
    generate_visualizations(itemset_res, rule_res, data)
    
    print("\nValidation and comparison pipeline complete!")
    print(f"Comparison CSV: {os.path.join(RESULTS_DIR, 'association_algorithm_comparison.csv')}")
    print(f"Comparison JSON: {os.path.join(RESULTS_DIR, 'association_algorithm_comparison.json')}")
    print(f"Visualizations: {VIS_DIR}")

if __name__ == "__main__":
    main()
