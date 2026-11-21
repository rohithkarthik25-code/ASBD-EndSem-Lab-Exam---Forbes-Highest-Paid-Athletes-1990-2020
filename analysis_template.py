"""
================================================================================
                    LIVE HACKATHON / EXAM STARTER TEMPLATE
================================================================================
Instructions:
1. Place your live dataset CSV in 'data/' or keep it in the project root.
2. Update the configuration placeholders below (DATASET_PATH, TARGET_COL, TASK_TYPE).
3. Run this script to perform non-destructive profiling and get immediate EDA insights.
4. Uncomment the corresponding task section (Classification, Regression, Clustering,
   or PySpark) to execute baseline modeling and algorithm benchmarks using exam_toolkit.
================================================================================
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# ENSURE TOOLKIT & DEPENDENCIES ARE IMPORTABLE
# ------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

try:
    import exam_toolkit as et
    TOOLKIT_AVAILABLE = True
except ImportError:
    TOOLKIT_AVAILABLE = False
    print("[WARNING] exam_toolkit.py could not be imported from parent directory.")

# ------------------------------------------------------------------------------
# PLACEHOLDERS: CONFIGURE WHEN LIVE PROBLEM ARRIVES
# ------------------------------------------------------------------------------
# Set path to the live dataset (e.g., 'data/train.csv' or an existing path)
DATASET_PATH = None  # e.g., CURRENT_DIR / "data" / "dataset.csv"

# Set target column name (None for unsupervised clustering / association mining)
TARGET_COL = None    # e.g., "Churn" or "SalePrice"

# Task type: "classification" | "regression" | "clustering" | "rules" | "spark"
TASK_TYPE = None

# Evaluation metric goal (e.g., "accuracy", "f1_macro", "roc_auc", "rmse", "r2")
EVAL_METRIC = None

# Output directories
RESULTS_DIR = CURRENT_DIR / "results"
REPORTS_DIR = CURRENT_DIR / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# STEP 1: NON-DESTRUCTIVE DATASET PROFILING
# ------------------------------------------------------------------------------
def profile_dataset(file_path):
    """
    Safely loads and inspects a CSV dataset without modifying anything.
    """
    if not file_path or not Path(file_path).exists():
        print(f"[!] Please specify a valid DATASET_PATH. Current: {file_path}")
        return None

    print("\n" + "=" * 70)
    print(f"PROFILING DATASET: {Path(file_path).name}")
    print("=" * 70)

    df = pd.read_csv(file_path)
    print(f"\n[+] Dimensions: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    print("\n--- FIRST 5 ROWS ---")
    print(df.head(5))

    print("\n--- COLUMN OVERVIEW & DATA TYPES ---")
    dtypes_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Values': df.nunique()
    })
    print(dtypes_df)

    dupes = df.duplicated().sum()
    print(f"\n[+] Duplicate Rows: {dupes:,} ({(dupes / len(df) * 100):.2f}%)")

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe().T.round(2))

    # Target inspection if defined
    if TARGET_COL and TARGET_COL in df.columns:
        print("\n" + "-" * 50)
        print(f"TARGET COLUMN ANALYSIS: '{TARGET_COL}'")
        print("-" * 50)
        if df[TARGET_COL].dtype in ['object', 'category', 'bool'] or df[TARGET_COL].nunique() <= 10:
            print("\nTarget Class Distribution:")
            vc = df[TARGET_COL].value_counts(dropna=False)
            vc_pct = df[TARGET_COL].value_counts(normalize=True, dropna=False) * 100
            target_dist = pd.DataFrame({'Count': vc, 'Percentage (%)': vc_pct.round(2)})
            print(target_dist)
            # Imbalance warning
            min_class_ratio = vc.min() / vc.max()
            if min_class_ratio < 0.2:
                print(f"[!] Warning: Class imbalance detected (min/max ratio: {min_class_ratio:.2f}). Consider stratify=y and balanced metrics.")
        else:
            print(f"\nTarget Summary (Continuous Numeric):")
            print(df[TARGET_COL].describe().round(2))
            print(f"Skewness: {df[TARGET_COL].skew():.2f}")
    elif TARGET_COL:
        print(f"\n[!] Specified TARGET_COL '{TARGET_COL}' not found in columns: {list(df.columns)}")

    return df


# ------------------------------------------------------------------------------
# STEP 2: STARTER WORKFLOWS (UNCOMMENT APPLICABLE SECTION AFTER CONFIGURING)
# ------------------------------------------------------------------------------

def run_classification_workflow(df, target_col):
    """
    Template for Supervised Classification using exam_toolkit.
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 70)
    print("RUNNING CLASSIFICATION WORKFLOW")
    print("=" * 70)

    # 1. Feature / Target separation
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Strict train/test split BEFORE any transformation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 10 else None
    )
    print(f"Train split: {X_train.shape}, Test split: {X_test.shape}")

    # 3. Multi-Algorithm Benchmark using exam_toolkit
    if TOOLKIT_AVAILABLE:
        leaderboard, fitted_models, preprocessor = et.run_all_classifiers(
            X_train, y_train, X_test, y_test
        )
        print("\n--- CLASSIFIER LEADERBOARD ---")
        print(leaderboard)
        leaderboard.to_csv(RESULTS_DIR / "classifier_leaderboard.csv", index=False)
        print(f"[+] Leaderboard saved to {RESULTS_DIR / 'classifier_leaderboard.csv'}")
        return leaderboard, fitted_models
    else:
        print("[!] exam_toolkit not loaded. Please ensure exam_toolkit.py is in the root directory.")
        return None, None


def run_regression_workflow(df, target_col):
    """
    Template for Supervised Regression using exam_toolkit.
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 70)
    print("RUNNING REGRESSION WORKFLOW")
    print("=" * 70)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train split: {X_train.shape}, Test split: {X_test.shape}")

    if TOOLKIT_AVAILABLE:
        leaderboard, fitted_models, preprocessor = et.run_all_regressors(
            X_train, y_train, X_test, y_test
        )
        print("\n--- REGRESSOR LEADERBOARD ---")
        print(leaderboard)
        leaderboard.to_csv(RESULTS_DIR / "regressor_leaderboard.csv", index=False)
        print(f"[+] Leaderboard saved to {RESULTS_DIR / 'regressor_leaderboard.csv'}")
        return leaderboard, fitted_models
    else:
        print("[!] exam_toolkit not loaded.")
        return None, None


def run_clustering_workflow(df):
    """
    Template for Unsupervised Clustering using exam_toolkit.
    """
    print("\n" + "=" * 70)
    print("RUNNING CLUSTERING WORKFLOW")
    print("=" * 70)

    if TOOLKIT_AVAILABLE:
        # cluster_benchmark automatically evaluates k=2..6
        benchmark_df = et.cluster_benchmark(df, k_range=range(2, 7))
        print("\n--- CLUSTER BENCHMARK ---")
        print(benchmark_df)
        benchmark_df.to_csv(RESULTS_DIR / "cluster_benchmark.csv", index=False)
        return benchmark_df
    else:
        print("[!] exam_toolkit not loaded.")
        return None


def run_pyspark_workflow(csv_path, target_col, is_classification=True):
    """
    Template for Distributed PySpark MLlib Workflow using exam_toolkit.
    """
    print("\n" + "=" * 70)
    print("RUNNING PYSPARK MLLIB WORKFLOW")
    print("=" * 70)

    if not TOOLKIT_AVAILABLE:
        print("[!] exam_toolkit not loaded.")
        return None

    # 1. Initialize SparkSession
    spark = et.spark_init(app_name="LiveHackathonSpark")

    try:
        # 2. Ingest CSV
        spark_df = et.spark_load_csv(spark, str(csv_path))
        et.spark_inspect(spark_df)

        # 3. Clean missing values
        clean_df = et.spark_handle_missing(spark_df, strategy='drop')

        # 4. Strict train/test split BEFORE fitting
        train_df, test_df = clean_df.randomSplit([0.8, 0.2], seed=42)
        print(f"Spark Train rows: {train_df.count():,}, Test rows: {test_df.count():,}")

        # 5. Run MLlib models
        if is_classification:
            leaderboard, models = et.spark_run_classifiers(train_df, test_df, target_col=target_col)
        else:
            leaderboard, models = et.spark_run_regressors(train_df, test_df, target_col=target_col)

        print("\n--- SPARK LEADERBOARD ---")
        print(leaderboard)
        leaderboard.to_csv(RESULTS_DIR / "spark_leaderboard.csv", index=False)
        return leaderboard
    finally:
        spark.stop()
        print("[+] SparkSession cleanly stopped.")


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=================================================================")
    print("         EXAM WORKSPACE ACTIVE: analysis_template.py             ")
    print("=================================================================")

    if DATASET_PATH is None:
        print("\n[READY] Environment and toolkit are verified.")
        print("[ACTION] When your live dataset arrives:")
        print("         1. Copy it to 'hackathon_live/data/<your_file>.csv'")
        print("         2. Set DATASET_PATH in this file")
        print("         3. Set TARGET_COL and TASK_TYPE")
        print("         4. Run: python hackathon_live/analysis_template.py")
    else:
        df = profile_dataset(DATASET_PATH)

        if df is not None:
            if TASK_TYPE == "classification" and TARGET_COL:
                run_classification_workflow(df, TARGET_COL)
            elif TASK_TYPE == "regression" and TARGET_COL:
                run_regression_workflow(df, TARGET_COL)
            elif TASK_TYPE == "clustering":
                run_clustering_workflow(df)
            elif TASK_TYPE == "spark" and TARGET_COL:
                run_pyspark_workflow(DATASET_PATH, TARGET_COL, is_classification=True)
            else:
                print(f"\n[INFO] Dataset profiled. To run models, set TASK_TYPE ('classification', 'regression', 'clustering', 'spark').")
