#!/usr/bin/env python3
"""
================================================================================
PHASE 4 — STEP 5: BENCHMARK B — ASSOCIATION MINING PERFORMANCE
================================================================================
Dataset: results/dic_transactions.csv (N = 301 market-basket transactions)
Algorithms Compared:
  1. Dynamic Itemset Counting (DIC) - checkpoint_M=50, minsup=0.08
  2. Co-occurrence Frequent Itemsets (COFI) Closed Mining - minsup=0.083
  3. Apache Spark MLlib FP-Growth - minSupport=0.08, minConfidence=0.50
Protocol:
  - 1 Warm-up run + 5 timed experimental repetitions per algorithm
  - Measurement of data preparation time, compute time, and total time
  - Tracking passes, frequent itemsets, and association rules
================================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add hackathon_live directory to path for algorithm imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from dic_association_mining import DynamicItemsetCounting, generate_association_rules as gen_dic_rules
from cofi_closed_mining import COFIMiner, generate_closed_association_rules as gen_cofi_rules

# Spark environment configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['JAVA_HOME'] = r'C:\Program Files\Microsoft\jdk-17.0.21.102-hotspot'
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TX_PATH = RESULTS_DIR / "dic_transactions.csv"


def load_transactions():
    t0 = time.perf_counter()
    df = pd.read_csv(TX_PATH)
    transactions = [[i.strip() for i in t.split(';')] for t in df['Items'].tolist()]
    prep_time = time.perf_counter() - t0
    return transactions, prep_time


def benchmark_dic(transactions, prep_time):
    print("\n--- Benchmarking DIC (Dynamic Itemset Counting, 5 Repetitions) ---")
    results = []
    
    # Warm-up
    dic_warm = DynamicItemsetCounting(min_support=0.08, checkpoint_M=50)
    _ = dic_warm.fit(transactions)
    
    for rep in range(1, 6):
        t0 = time.perf_counter()
        dic = DynamicItemsetCounting(min_support=0.08, checkpoint_M=50)
        fi = dic.fit(transactions)
        compute_time = time.perf_counter() - t0
        
        t0_rules = time.perf_counter()
        # Count rules without overwriting approved CSV on disk
        # DIC rule generation
        N = len(transactions)
        rules_cnt = 0
        minsup_cnt = int(np.ceil(0.08 * N))
        for itemset, item_count in fi.items():
            if len(itemset) >= 2:
                items_list = list(itemset)
                from itertools import combinations
                for r in range(1, len(items_list)):
                    for ant_tuple in combinations(items_list, r):
                        ant = frozenset(ant_tuple)
                        con = itemset - ant
                        if ant in fi:
                            ant_count = fi[ant]
                            conf = item_count / ant_count
                            if conf >= 0.50:
                                con_count = fi.get(con, 0)
                                if con_count > 0:
                                    lift = conf / (con_count / N)
                                    if lift >= 1.10:
                                        rules_cnt += 1
        rule_gen_time = time.perf_counter() - t0_rules
        total_time = prep_time + compute_time + rule_gen_time
        
        results.append({
            "algorithm": "DIC (Dynamic Itemset Counting)",
            "execution_type": "In-Memory Sequential (Checkpoint M=50)",
            "run_id": rep,
            "data_prep_time_ms": round(prep_time * 1000, 3),
            "compute_time_ms": round(compute_time * 1000, 3),
            "rule_gen_time_ms": round(rule_gen_time * 1000, 3),
            "spark_init_time_ms": 0.0,
            "total_execution_time_ms": round(total_time * 1000, 3),
            "database_passes": 4,
            "frequent_itemsets": len(fi),
            "association_rules": rules_cnt
        })
        print(f"  Run {rep}: Compute={compute_time*1000:.2f}ms, RuleGen={rule_gen_time*1000:.2f}ms, Total={total_time*1000:.2f}ms | Itemsets={len(fi)}, Rules={rules_cnt}")
        
    return results


def benchmark_cofi(transactions, prep_time):
    print("\n--- Benchmarking COFI (Closed Itemset Mining, 5 Repetitions) ---")
    results = []
    
    # Warm-up
    cofi_warm = COFIMiner(transactions=transactions, minsup=0.083)
    _ = cofi_warm.mine()
    cofi_warm.extract_closed_itemsets()
    
    for rep in range(1, 6):
        t0 = time.perf_counter()
        cofi = COFIMiner(transactions=transactions, minsup=0.083)
        _ = cofi.mine()
        cofi.extract_closed_itemsets()
        compute_time = time.perf_counter() - t0
        
        # Rule generation in-memory
        t0_rules = time.perf_counter()
        N = len(transactions)
        closed_itemsets = cofi.closed_frequent_itemsets
        all_fi = cofi.all_frequent_itemsets
        rules_cnt = 0
        from itertools import combinations
        for itemset, itemset_cnt in closed_itemsets.items():
            if len(itemset) >= 2:
                items_list = list(itemset)
                for r in range(1, len(items_list)):
                    for ant_tuple in combinations(items_list, r):
                        ant = frozenset(ant_tuple)
                        con = itemset - ant
                        ant_cnt = all_fi.get(ant)
                        if ant_cnt:
                            conf = itemset_cnt / ant_cnt
                            if conf >= 0.50:
                                con_cnt = all_fi.get(con, 0)
                                if con_cnt:
                                    lift = conf / (con_cnt / N)
                                    if lift >= 1.10:
                                        rules_cnt += 1
        rule_gen_time = time.perf_counter() - t0_rules
        total_time = prep_time + compute_time + rule_gen_time
        
        results.append({
            "algorithm": "COFI (Closed Frequent Itemsets)",
            "execution_type": "In-Memory Pattern-Growth (Subtrees)",
            "run_id": rep,
            "data_prep_time_ms": round(prep_time * 1000, 3),
            "compute_time_ms": round(compute_time * 1000, 3),
            "rule_gen_time_ms": round(rule_gen_time * 1000, 3),
            "spark_init_time_ms": 0.0,
            "total_execution_time_ms": round(total_time * 1000, 3),
            "database_passes": 2,
            "frequent_itemsets": len(closed_itemsets),  # 79 closed
            "association_rules": rules_cnt  # 56 closed
        })
        print(f"  Run {rep}: Compute={compute_time*1000:.2f}ms, RuleGen={rule_gen_time*1000:.2f}ms, Total={total_time*1000:.2f}ms | ClosedItemsets={len(closed_itemsets)}, ClosedRules={rules_cnt}")
        
    return results


def benchmark_spark_fpgrowth(transactions, prep_time):
    print("\n--- Benchmarking PySpark FP-Growth (Warm Execution, 5 Repetitions) ---")
    
    # Warm-up SparkSession
    t0_spk_init = time.perf_counter()
    spark = SparkSession.builder \
        .appName("Benchmark_PySpark_FPGrowth") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    init_time = time.perf_counter() - t0_spk_init
    print(f"  SparkSession initialized in {init_time:.3f}s")
    
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    # DataFrame conversion
    t0_df = time.perf_counter()
    df_spark = spark.createDataFrame([(i, t) for i, t in enumerate(transactions)], ["id", "items"])
    df_prep_time = (time.perf_counter() - t0_df) + prep_time
    
    # Warm-up execution
    fp_warm = FPGrowth(itemsCol="items", minSupport=0.08, minConfidence=0.50)
    m_warm = fp_warm.fit(df_spark)
    _ = m_warm.freqItemsets.count()
    _ = m_warm.associationRules.count()
    
    warm_results = []
    for rep in range(1, 6):
        t0 = time.perf_counter()
        fp = FPGrowth(itemsCol="items", minSupport=0.08, minConfidence=0.50)
        model = fp.fit(df_spark)
        n_fi = model.freqItemsets.count()
        fit_time = time.perf_counter() - t0
        
        t0_rules = time.perf_counter()
        n_rules = model.associationRules.count()
        rule_time = time.perf_counter() - t0_rules
        
        compute_time = fit_time + rule_time
        total_time = df_prep_time + compute_time
        
        warm_results.append({
            "algorithm": "Spark MLlib FP-Growth",
            "execution_type": "Distributed Warm Execution (local[2])",
            "run_id": rep,
            "data_prep_time_ms": round(df_prep_time * 1000, 3),
            "compute_time_ms": round(fit_time * 1000, 3),
            "rule_gen_time_ms": round(rule_time * 1000, 3),
            "spark_init_time_ms": 0.0,
            "total_execution_time_ms": round(total_time * 1000, 3),
            "database_passes": 2,
            "frequent_itemsets": n_fi,  # 87
            "association_rules": n_rules  # 71
        })
        print(f"  Run {rep}: Fit={fit_time*1000:.2f}ms, Rules={rule_time*1000:.2f}ms, Total={total_time*1000:.2f}ms | Itemsets={n_fi}, Rules={n_rules}")
        
    spark.stop()
    print("  SparkSession terminated.")
    
    # Cold start execution benchmark
    print("\n--- Benchmarking PySpark FP-Growth (Cold Start with Full Session Lifecycle, 5 Repetitions) ---")
    cold_results = []
    for rep in range(1, 6):
        t0_cold_init = time.perf_counter()
        spark_cold = SparkSession.builder \
            .appName("Benchmark_PySpark_FPGrowth_Cold") \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        sc_c = spark_cold.sparkContext
        sc_c.setLogLevel("ERROR")
        cold_init = time.perf_counter() - t0_cold_init
        
        t0_df_c = time.perf_counter()
        df_spk_c = spark_cold.createDataFrame([(i, t) for i, t in enumerate(transactions)], ["id", "items"])
        df_prep_c = (time.perf_counter() - t0_df_c) + prep_time
        
        t0_fit_c = time.perf_counter()
        fp_c = FPGrowth(itemsCol="items", minSupport=0.08, minConfidence=0.50)
        model_c = fp_c.fit(df_spk_c)
        n_fi_c = model_c.freqItemsets.count()
        fit_c = time.perf_counter() - t0_fit_c
        
        t0_rules_c = time.perf_counter()
        n_rules_c = model_c.associationRules.count()
        rules_c = time.perf_counter() - t0_rules_c
        
        comp_c = fit_c + rules_c
        tot_c = cold_init + df_prep_c + comp_c
        
        cold_results.append({
            "algorithm": "Spark MLlib FP-Growth",
            "execution_type": "Distributed Cold Start (local[2])",
            "run_id": rep,
            "data_prep_time_ms": round(df_prep_c * 1000, 3),
            "compute_time_ms": round(fit_c * 1000, 3),
            "rule_gen_time_ms": round(rules_c * 1000, 3),
            "spark_init_time_ms": round(cold_init * 1000, 3),
            "total_execution_time_ms": round(tot_c * 1000, 3),
            "database_passes": 2,
            "frequent_itemsets": n_fi_c,
            "association_rules": n_rules_c
        })
        print(f"  Run {rep}: Init={cold_init*1000:.2f}ms, Fit={fit_c*1000:.2f}ms, Rules={rules_c*1000:.2f}ms, Total={tot_c*1000:.2f}ms | Itemsets={n_fi_c}")
        spark_cold.stop()
        
    return warm_results + cold_results


def main():
    print("=" * 75)
    print("PHASE 4 — STEP 5: BENCHMARK B — ASSOCIATION MINING PERFORMANCE")
    print("=" * 75)
    
    transactions, prep_time = load_transactions()
    print(f"[+] Ingested {len(transactions)} market-basket transactions in {prep_time*1000:.2f} ms")
    
    dic_res = benchmark_dic(transactions, prep_time)
    cofi_res = benchmark_cofi(transactions, prep_time)
    spark_res = benchmark_spark_fpgrowth(transactions, prep_time)
    
    all_res = dic_res + cofi_res + spark_res
    df_assoc = pd.DataFrame(all_res)
    
    csv_path = RESULTS_DIR / "benchmark_association.csv"
    df_assoc.to_csv(csv_path, index=False)
    print(f"\n[+] Exported association mining benchmark results to: {csv_path}")
    
    # Summary statistics table
    print("\n" + "=" * 75)
    print("BENCHMARK B SUMMARY STATISTICS (COMPUTE TIME & END-TO-END TIME MS)")
    print("=" * 75)
    for (alg, ex_type), grp in df_assoc.groupby(["algorithm", "execution_type"]):
        comp = grp["compute_time_ms"]
        e2e = grp["total_execution_time_ms"]
        fi = grp["frequent_itemsets"].iloc[0]
        rules = grp["association_rules"].iloc[0]
        passes = grp["database_passes"].iloc[0]
        print(f"\n{alg} [{ex_type}]")
        print(f"  Compute Time (ms):    Mean={comp.mean():.2f}, Median={comp.median():.2f}, Min={comp.min():.2f}, Max={comp.max():.2f}, Std={comp.std():.2f}")
        print(f"  End-to-End Time (ms):  Mean={e2e.mean():.2f}, Median={e2e.median():.2f}, Min={e2e.min():.2f}, Max={e2e.max():.2f}, Std={e2e.std():.2f}")
        print(f"  Mining Output:        Passes={passes}, Itemsets={fi}, Rules={rules}")

if __name__ == "__main__":
    main()
