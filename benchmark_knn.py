#!/usr/bin/env python3
"""
================================================================================
PHASE 4 — STEP 5: BENCHMARK A — CONVENTIONAL KNN VS PYSPARK KNN
================================================================================
Dataset: Forbes Richest Athletes (1990-2020) Cleaned (N = 301)
Task: Binary Classification of Sport_Type ('Team Sport' vs 'Individual Sport')
Protocol:
  - Stratified 80/20 split (Train: 240, Test: 61, random_state=42)
  - Features: earnings ($ million), Year, Current Rank, Nationality_Group
  - Preprocessing: StandardScaler + OneHotEncoder (fit on train only)
  - Model: K=9, Manhattan distance (p=1), Uniform weights
  - 1 Warm-up run + 5 timed experimental repetitions
================================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Spark environment configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['JAVA_HOME'] = r'C:\Program Files\Microsoft\jdk-17.0.21.102-hotspot'
from pyspark.sql import SparkSession

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "forbes_athletes_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_split_data():
    df = pd.read_csv(DATA_PATH)
    team_sports = ['Basketball', 'Soccer', 'American Football', 'Baseball', 'Hockey']
    df['Sport_Type'] = df['Sport_Clean'].apply(
        lambda x: 'Team Sport' if x in team_sports else 'Individual Sport'
    )
    feature_cols = ['earnings ($ million)', 'Year', 'Current Rank', 'Nationality_Group']
    target_col = 'Sport_Type'
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def preprocess(X_train, X_test):
    numeric_features = ['earnings ($ million)', 'Year', 'Current Rank']
    categorical_features = ['Nationality_Group']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])
    
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    return X_train_trans, X_test_trans

def run_sklearn_knn_run(X_train_trans, y_train, X_test_trans, y_test):
    # Model preparation (fit)
    t0_fit = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=9, p=1, metric='manhattan', weights='uniform')
    knn.fit(X_train_trans, y_train)
    fit_time = time.perf_counter() - t0_fit
    
    # Prediction
    t0_pred = time.perf_counter()
    y_pred = knn.predict(X_test_trans)
    pred_time = time.perf_counter() - t0_pred
    
    acc = accuracy_score(y_test, y_pred)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    return {
        "fit_time": fit_time,
        "pred_time": pred_time,
        "total_compute_time": fit_time + pred_time,
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": y_pred
    }

def run_pyspark_knn_run(X_train_trans, y_train, X_test_trans, y_test, spark=None, measure_init=False):
    init_time = 0.0
    created_spark = False
    
    if spark is None:
        t0_init = time.perf_counter()
        spark = SparkSession.builder \
            .appName("Benchmark_PySpark_KNN") \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        init_time = time.perf_counter() - t0_init
        created_spark = True
        
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    
    # Compute phase
    t0_compute = time.perf_counter()
    
    # Broadcast train data
    b_train_X = sc.broadcast(X_train_trans)
    b_train_y = sc.broadcast(y_train.values)
    
    # Distribute test points
    test_records = [(i, X_test_trans[i].tolist()) for i in range(len(X_test_trans))]
    test_rdd = sc.parallelize(test_records, numSlices=2)
    
    def compute_partition_knn(iterator):
        train_X = b_train_X.value
        train_y = b_train_y.value
        res = []
        for idx, vec in iterator:
            q = np.array(vec)
            dist = np.sum(np.abs(train_X - q), axis=1)
            nn_idx = np.argsort(dist, kind='stable')[:9]
            nn_labels = train_y[nn_idx]
            team_votes = int(np.sum(nn_labels == 'Team Sport'))
            indiv_votes = int(np.sum(nn_labels == 'Individual Sport'))
            pred_class = 'Team Sport' if team_votes > indiv_votes else 'Individual Sport'
            res.append((idx, pred_class))
        return iter(res)
    
    raw_preds = test_rdd.mapPartitions(compute_partition_knn).collect()
    compute_time = time.perf_counter() - t0_compute
    
    # Clean up broadcasts to avoid memory accumulation
    b_train_X.unpersist()
    b_train_y.unpersist()
    
    raw_preds.sort(key=lambda x: x[0])
    y_pred = np.array([p[1] for p in raw_preds])
    
    acc = accuracy_score(y_test, y_pred)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    if created_spark and not measure_init:
        spark.stop()
        
    return {
        "init_time": init_time,
        "compute_time": compute_time,
        "total_time": init_time + compute_time,
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": y_pred,
        "spark_session": spark if created_spark else None
    }

def main():
    print("=" * 75)
    print("PHASE 4 — STEP 5: BENCHMARK A — CONVENTIONAL KNN VS PYSPARK KNN")
    print("=" * 75)
    
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Preprocessing benchmark
    prep_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        X_tr, X_te = preprocess(X_train, X_test)
        prep_times.append(time.perf_counter() - t0)
    median_prep_time = np.median(prep_times)
    print(f"[+] Feature Preprocessing Median Time: {median_prep_time*1000:.3f} ms")
    
    X_train_trans, X_test_trans = preprocess(X_train, X_test)
    
    # -------------------------------------------------------------------------
    # 1. Warm-Up Runs
    # -------------------------------------------------------------------------
    print("\n--- Executing Warm-Up Runs ---")
    _ = run_sklearn_knn_run(X_train_trans, y_train, X_test_trans, y_test)
    
    # Cold Spark warm-up to initialize JVM
    print("  Initializing warm-up SparkSession...")
    t0_spk_init = time.perf_counter()
    spark_session = SparkSession.builder \
        .appName("Benchmark_PySpark_KNN_Warmup") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    cold_init_time = time.perf_counter() - t0_spk_init
    print(f"  Warm-up SparkSession created in {cold_init_time:.3f}s")
    
    _ = run_pyspark_knn_run(X_train_trans, y_train, X_test_trans, y_test, spark=spark_session)
    print("  Warm-up complete.")
    
    # -------------------------------------------------------------------------
    # 2. Benchmarking Scikit-Learn KNN (5 Repetitions)
    # -------------------------------------------------------------------------
    print("\n--- Benchmarking Scikit-Learn KNN (5 Repetitions) ---")
    sklearn_results = []
    for rep in range(1, 6):
        res = run_sklearn_knn_run(X_train_trans, y_train, X_test_trans, y_test)
        sklearn_results.append({
            "framework": "Scikit-Learn (Conventional)",
            "run_id": rep,
            "preprocessing_time_ms": round(median_prep_time * 1000, 3),
            "model_prep_time_ms": round(res["fit_time"] * 1000, 3),
            "prediction_time_ms": round(res["pred_time"] * 1000, 3),
            "compute_time_ms": round(res["total_compute_time"] * 1000, 3),
            "spark_init_time_ms": 0.0,
            "total_execution_time_ms": round((median_prep_time + res["total_compute_time"]) * 1000, 3),
            "accuracy": round(res["accuracy"], 4),
            "macro_f1": round(res["macro_f1"], 4),
            "weighted_f1": round(res["weighted_f1"], 4)
        })
        print(f"  Run {rep}: Fit={res['fit_time']*1000:.2f}ms, Pred={res['pred_time']*1000:.2f}ms, Total={res['total_compute_time']*1000:.2f}ms | Acc={res['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # 3. Benchmarking PySpark KNN (Warm SparkSession, 5 Repetitions)
    # -------------------------------------------------------------------------
    print("\n--- Benchmarking PySpark KNN (Warm SparkSession, 5 Repetitions) ---")
    spark_warm_results = []
    for rep in range(1, 6):
        res = run_pyspark_knn_run(X_train_trans, y_train, X_test_trans, y_test, spark=spark_session)
        spark_warm_results.append({
            "framework": "PySpark KNN (Warm Execution)",
            "run_id": rep,
            "preprocessing_time_ms": round(median_prep_time * 1000, 3),
            "model_prep_time_ms": 0.0,  # Spark broadcasts as part of compute
            "prediction_time_ms": round(res["compute_time"] * 1000, 3),
            "compute_time_ms": round(res["compute_time"] * 1000, 3),
            "spark_init_time_ms": 0.0,
            "total_execution_time_ms": round((median_prep_time + res["compute_time"]) * 1000, 3),
            "accuracy": round(res["accuracy"], 4),
            "macro_f1": round(res["macro_f1"], 4),
            "weighted_f1": round(res["weighted_f1"], 4)
        })
        print(f"  Run {rep}: Compute={res['compute_time']*1000:.2f}ms | Acc={res['accuracy']:.4f}")
        
    spark_session.stop()
    print("  Warm SparkSession terminated.")

    # -------------------------------------------------------------------------
    # 4. Benchmarking PySpark KNN (Cold Start, 5 Repetitions)
    # -------------------------------------------------------------------------
    print("\n--- Benchmarking PySpark KNN (Cold Start with Full Session Lifecycle, 5 Repetitions) ---")
    spark_cold_results = []
    for rep in range(1, 6):
        res = run_pyspark_knn_run(X_train_trans, y_train, X_test_trans, y_test, spark=None, measure_init=True)
        spark_cold_results.append({
            "framework": "PySpark KNN (Cold Start)",
            "run_id": rep,
            "preprocessing_time_ms": round(median_prep_time * 1000, 3),
            "model_prep_time_ms": 0.0,
            "prediction_time_ms": round(res["compute_time"] * 1000, 3),
            "compute_time_ms": round(res["compute_time"] * 1000, 3),
            "spark_init_time_ms": round(res["init_time"] * 1000, 3),
            "total_execution_time_ms": round((res["init_time"] + res["compute_time"] + median_prep_time) * 1000, 3),
            "accuracy": round(res["accuracy"], 4),
            "macro_f1": round(res["macro_f1"], 4),
            "weighted_f1": round(res["weighted_f1"], 4)
        })
        print(f"  Run {rep}: Init={res['init_time']*1000:.2f}ms, Compute={res['compute_time']*1000:.2f}ms, Total={res['total_time']*1000:.2f}ms | Acc={res['accuracy']:.4f}")
        if res["spark_session"]:
            res["spark_session"].stop()

    # Consolidate results into DataFrame
    all_runs = sklearn_results + spark_warm_results + spark_cold_results
    df_runs = pd.DataFrame(all_runs)
    
    csv_path = RESULTS_DIR / "benchmark_knn.csv"
    df_runs.to_csv(csv_path, index=False)
    print(f"\n[+] Exported KNN benchmark results to: {csv_path}")
    
    # Compute summary statistics
    print("\n" + "=" * 75)
    print("BENCHMARK A SUMMARY STATISTICS (METRIC: TOTAL COMPUTE TIME MS)")
    print("=" * 75)
    for fw in df_runs["framework"].unique():
        sub = df_runs[df_runs["framework"] == fw]
        times = sub["compute_time_ms"]
        e2e = sub["total_execution_time_ms"]
        print(f"\nFramework: {fw}")
        print(f"  Compute Time (ms):  Mean={times.mean():.2f}, Median={times.median():.2f}, Min={times.min():.2f}, Max={times.max():.2f}, Std={times.std():.2f}")
        print(f"  End-to-End Time (ms): Mean={e2e.mean():.2f}, Median={e2e.median():.2f}, Min={e2e.min():.2f}, Max={e2e.max():.2f}, Std={e2e.std():.2f}")
        print(f"  Classification Acc: {sub['accuracy'].iloc[0]:.4f}, Macro F1: {sub['macro_f1'].iloc[0]:.4f}")

if __name__ == "__main__":
    main()
