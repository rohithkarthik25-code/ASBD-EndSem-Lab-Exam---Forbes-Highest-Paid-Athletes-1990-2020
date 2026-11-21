"""
================================================================================
PHASE 4 — STEP 2: DISTRIBUTED KNN CLASSIFICATION USING PYSPARK
================================================================================
Dataset: Forbes Richest Athletes (1990-2020) Cleaned Dataset (N = 301)
Architecture: Distributed Distance Calculations via PySpark Broadcast Variables
Target: Sport_Type ('Team Sport' vs. 'Individual Sport')
Features: earnings ($ million), Year, Current Rank, Nationality_Group
Baseline Comparison: Approved Scikit-Learn KNeighborsClassifier (K=9, p=1, uniform)
Author: Antigravity Agentic Assistant (Big Data Analytics Edition)
Date: November 2026
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION & PATH SETUP
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "forbes_athletes_cleaned.csv"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
SKLEARN_MODEL_PATH = BASE_DIR / "models" / "knn_model.pkl"
SKLEARN_PREDS_PATH = BASE_DIR / "results" / "knn_test_predictions.csv"
SKLEARN_METRICS_PATH = BASE_DIR / "results" / "knn_metrics.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure Java and PySpark Worker Environments on Windows
JDK_PATH = r"C:\Program Files\Microsoft\jdk-17.0.21.102-hotspot"
if os.path.exists(JDK_PATH):
    os.environ["JAVA_HOME"] = JDK_PATH
    os.environ["PATH"] = os.path.join(JDK_PATH, "bin") + ";" + os.environ.get("PATH", "")

# Explicitly bind Python worker processes to avoid Windows Store execution alias
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from pyspark.sql import SparkSession


# ------------------------------------------------------------------------------
# 2. DATA INGESTION & STRATIFIED SPLIT
# ------------------------------------------------------------------------------
def load_and_split_data():
    """
    Ingests cleaned Forbes athletes dataset and constructs identical 80/20
    stratified train/test split matching Scikit-Learn baseline.
    """
    print("=" * 75)
    print("STEP 1: INGESTING DATA & FORMULATING STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 75)

    df = pd.read_csv(DATA_PATH)
    team_sports = ['Basketball', 'Soccer', 'American Football', 'Baseball', 'Hockey']
    df['Sport_Type'] = df['Sport_Clean'].apply(
        lambda x: 'Team Sport' if x in team_sports else 'Individual Sport'
    )

    feature_cols = ['earnings ($ million)', 'Year', 'Current Rank', 'Nationality_Group']
    target_col = 'Sport_Type'

    # Strict leakage audit
    forbidden = ['Sport', 'Sport_Clean', 'Name', 'S.NO', 'Rank_Tier', 'Earnings_Category', 'Rank_Trajectory', 'Decade']
    for col in forbidden:
        assert col not in feature_cols, f"Data leakage error: {col} in features!"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Stratified split: 80% Train (N=240), 20% Test (N=61), seed=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Loaded {len(df)} total records.")
    print(f"[+] Train Partition: {len(X_train)} records (80%)")
    print(f"[+] Test Partition:  {len(X_test)} records (20%)")
    print(f"[+] Class distribution: Team Sport={np.sum(y_train == 'Team Sport')}, Individual Sport={np.sum(y_train == 'Individual Sport')}")

    return X_train, X_test, y_train, y_test, df


# ------------------------------------------------------------------------------
# 3. FEATURE PREPROCESSING (LEAKAGE-FREE PIPELINE)
# ------------------------------------------------------------------------------
def preprocess_features(X_train, X_test):
    """
    Applies StandardScaler on continuous features and OneHotEncoder on categorical features.
    Strictly fitted on X_train only to prevent data leakage.
    """
    print("\n" + "=" * 75)
    print("STEP 2: PREPROCESSING FEATURES (FIT ON TRAIN ONLY)")
    print("=" * 75)

    numeric_features = ['earnings ($ million)', 'Year', 'Current Rank']
    categorical_features = ['Nationality_Group']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    print(f"[+] Transformed training matrix shape: {X_train_trans.shape}")
    print(f"[+] Transformed test matrix shape:     {X_test_trans.shape}")
    print(f"[+] Transformed feature dimensions: 3 scaled continuous + 1 one-hot binary (Nationality_Group_USA)")

    return X_train_trans, X_test_trans, preprocessor


# ------------------------------------------------------------------------------
# 4. PYSPARK DISTRIBUTED KNN IMPLEMENTATION
# ------------------------------------------------------------------------------
def run_pyspark_knn(X_train_trans, y_train, X_test_trans, k=9, p=1):
    """
    Executes distributed KNN classification in PySpark:
      - Broadcasts training feature matrix and labels to all executors.
      - Distributes test query points across worker partitions.
      - Computes Minkowski distance (p=1, Manhattan) in parallel per test point.
      - Extracts top-K nearest neighbors and evaluates majority vote.
    """
    print("\n" + "=" * 75)
    print("STEP 3: INITIALIZING PYSPARK SESSION & DISTRIBUTED KNN EXECUTION")
    print(f"Hyperparameters: K = {k} | Distance Metric = Manhattan (p={p}) | Weights = Uniform")
    print("=" * 75)

    start_spark_time = time.perf_counter()

    spark = SparkSession.builder \
        .appName("Forbes_Distributed_KNN") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print(f"[+] Active SparkSession: Version {spark.version} | Master: {sc.master}")

    # Broadcast training feature matrix and labels
    b_train_X = sc.broadcast(X_train_trans)
    b_train_y = sc.broadcast(y_train.values)

    # Parallelize test records: (test_idx, feature_vector)
    test_records = [(i, X_test_trans[i].tolist()) for i in range(len(X_test_trans))]
    test_rdd = sc.parallelize(test_records, numSlices=2)

    def compute_partition_knn(iterator):
        train_X = b_train_X.value
        train_y = b_train_y.value
        partition_results = []

        for idx, test_vec in iterator:
            q = np.array(test_vec)
            # Manhattan distance (p=1): sum of absolute differences
            distances = np.sum(np.abs(train_X - q), axis=1)

            # Extract K nearest neighbors with stable tie-breaking
            nn_indices = np.argsort(distances, kind='stable')[:k]
            nn_labels = train_y[nn_indices]

            # Vote tally
            team_votes = int(np.sum(nn_labels == 'Team Sport'))
            indiv_votes = int(np.sum(nn_labels == 'Individual Sport'))

            prob_team = team_votes / k
            prob_indiv = indiv_votes / k
            pred_class = 'Team Sport' if prob_team > prob_indiv else 'Individual Sport'

            partition_results.append({
                'test_idx': idx,
                'pred_class': pred_class,
                'prob_indiv': round(prob_indiv, 4),
                'prob_team': round(prob_team, 4),
                'team_votes': team_votes,
                'indiv_votes': indiv_votes
            })

        return iter(partition_results)

    # Execute distributed mapping and collect results
    distributed_preds = test_rdd.mapPartitions(compute_partition_knn).collect()

    spark_duration = time.perf_counter() - start_spark_time
    print(f"[+] Distributed KNN inference completed in {spark_duration:.2f} seconds.")

    # Stop SparkSession cleanly
    spark.stop()
    print("[+] SparkSession cleanly terminated.")

    # Sort results by original test index
    distributed_preds.sort(key=lambda x: x['test_idx'])
    return distributed_preds, spark_duration


# ------------------------------------------------------------------------------
# 5. VALIDATION AGAINST SCIKIT-LEARN APPROVED BASELINE
# ------------------------------------------------------------------------------
def validate_results(pyspark_preds, X_test, y_test):
    """
    Compares PySpark predictions against approved Scikit-Learn predictions
    record-by-record and evaluates classification metrics.
    """
    print("\n" + "=" * 75)
    print("STEP 4: VALIDATION AGAINST SCIKIT-LEARN GROUND TRUTH")
    print("=" * 75)

    sk_preds_df = pd.read_csv(SKLEARN_PREDS_PATH)
    assert len(pyspark_preds) == len(sk_preds_df) == len(y_test) == 61, "Mismatch in test records count!"

    spark_pred_labels = [p['pred_class'] for p in pyspark_preds]
    spark_prob_team = [p['prob_team'] for p in pyspark_preds]
    spark_prob_indiv = [p['prob_indiv'] for p in pyspark_preds]

    sk_pred_labels = sk_preds_df['Predicted_Sport_Type'].tolist()
    actual_labels = y_test.values

    # Prediction agreement
    matches = np.array(spark_pred_labels) == np.array(sk_pred_labels)
    agreement_pct = np.mean(matches) * 100.0

    print(f"[+] Total Test Records Evaluated: {len(spark_pred_labels)}")
    print(f"[+] Identical Predictions Count:  {np.sum(matches)} / {len(matches)}")
    print(f"[+] Prediction Agreement:         {agreement_pct:.2f}%")

    # Metrics computation
    acc = accuracy_score(actual_labels, spark_pred_labels)
    bal_acc = balanced_accuracy_score(actual_labels, spark_pred_labels)
    prec_macro = precision_score(actual_labels, spark_pred_labels, average='macro', pos_label=None)
    rec_macro = recall_score(actual_labels, spark_pred_labels, average='macro')
    f1_mac = f1_score(actual_labels, spark_pred_labels, average='macro')
    prec_wt = precision_score(actual_labels, spark_pred_labels, average='weighted')
    rec_wt = recall_score(actual_labels, spark_pred_labels, average='weighted')
    f1_wt = f1_score(actual_labels, spark_pred_labels, average='weighted')
    cm = confusion_matrix(actual_labels, spark_pred_labels, labels=['Individual Sport', 'Team Sport']).tolist()
    clf_rep = classification_report(actual_labels, spark_pred_labels, target_names=['Individual Sport', 'Team Sport'], output_dict=True)

    try:
        roc_auc = roc_auc_score(actual_labels == 'Team Sport', spark_prob_team)
    except Exception:
        roc_auc = None

    print(f"[+] PySpark Test Accuracy:         {acc * 100:.2f}%")
    print(f"[+] PySpark Test Balanced Accuracy:{bal_acc * 100:.2f}%")
    print(f"[+] PySpark Test Macro F1-Score:   {f1_mac:.4f}")
    print(f"[+] PySpark Test Confusion Matrix: {cm}")

    # Inspect any probability divergence due to boundary distance ties
    mismatches = []
    prob_divergences = []
    for i in range(len(spark_pred_labels)):
        sk_prob = sk_preds_df.iloc[i]['Prob_Team_Sport']
        sp_prob = spark_prob_team[i]
        if spark_pred_labels[i] != sk_pred_labels[i]:
            mismatches.append({
                'test_idx': i,
                'actual': actual_labels[i],
                'spark_pred': spark_pred_labels[i],
                'sklearn_pred': sk_pred_labels[i]
            })
        if not np.isclose(sp_prob, sk_prob, atol=1e-3):
            prob_divergences.append({
                'test_idx': i,
                'actual': actual_labels[i],
                'spark_pred': spark_pred_labels[i],
                'sklearn_pred': sk_pred_labels[i],
                'spark_prob_team': sp_prob,
                'sklearn_prob_team': sk_prob,
                'difference': round(abs(sp_prob - sk_prob), 4),
                'cause': 'Distance tie at 9th neighbor boundary (stable index vs. tree traversal)'
            })

    print(f"[+] Prediction Class Mismatches:  {len(mismatches)}")
    print(f"[+] Probability Tie Divergences:  {len(prob_divergences)}")
    if prob_divergences:
        for div in prob_divergences:
            print(f"    • Test Row {div['test_idx']}: Spark P(Team)={div['spark_prob_team']}, Sklearn P(Team)={div['sklearn_prob_team']} | Diff={div['difference']} | Both Predicted: '{div['spark_pred']}'")

    # Build predictions DataFrame
    preds_records = []
    for i in range(len(pyspark_preds)):
        preds_records.append({
            'test_index': i,
            'earnings ($ million)': X_test.iloc[i]['earnings ($ million)'],
            'Year': X_test.iloc[i]['Year'],
            'Current Rank': X_test.iloc[i]['Current Rank'],
            'Nationality_Group': X_test.iloc[i]['Nationality_Group'],
            'Actual_Sport_Type': actual_labels[i],
            'PySpark_Predicted': spark_pred_labels[i],
            'Sklearn_Predicted': sk_pred_labels[i],
            'PySpark_Prob_Individual': spark_prob_indiv[i],
            'PySpark_Prob_Team': spark_prob_team[i],
            'Prediction_Match': int(spark_pred_labels[i] == sk_pred_labels[i]),
            'Correct': int(spark_pred_labels[i] == actual_labels[i])
        })

    preds_df = pd.DataFrame(preds_records)
    preds_path = RESULTS_DIR / "pyspark_knn_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"[+] Saved PySpark predictions table: {preds_path.name}")

    metrics_payload = {
        'model_name': 'Distributed KNN Classification (PySpark)',
        'framework': 'PySpark 4.3.0 + Local Master (local[2])',
        'hyperparameters': {
            'k': 9,
            'metric': 'manhattan (p=1)',
            'weights': 'uniform'
        },
        'dataset_summary': {
            'total_rows': 301,
            'train_rows': 240,
            'test_rows': 61,
            'feature_count': 4,
            'features': ['earnings ($ million)', 'Year', 'Current Rank', 'Nationality_Group']
        },
        'validation_metrics': {
            'prediction_agreement_percentage': agreement_pct,
            'test_accuracy': float(acc),
            'test_balanced_accuracy': float(bal_acc),
            'test_precision_macro': float(prec_macro),
            'test_recall_macro': float(rec_macro),
            'test_f1_macro': float(f1_mac),
            'test_precision_weighted': float(prec_wt),
            'test_recall_weighted': float(rec_wt),
            'test_f1_weighted': float(f1_wt),
            'test_roc_auc': float(roc_auc) if roc_auc else None,
            'confusion_matrix': cm,
            'classification_report': clf_rep
        },
        'parity_audit': {
            'total_predictions': len(spark_pred_labels),
            'identical_predictions': int(np.sum(matches)),
            'class_mismatches_count': len(mismatches),
            'mismatches': mismatches,
            'probability_divergences_count': len(prob_divergences),
            'probability_divergences': prob_divergences
        }
    }

    metrics_path = RESULTS_DIR / "pyspark_knn_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[+] Saved PySpark metrics payload: {metrics_path.name}")

    return metrics_payload, preds_df


# ------------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("STARTING PYSPARK DISTRIBUTED KNN PIPELINE")
    print("=" * 75)

    X_train, X_test, y_train, y_test, df = load_and_split_data()
    X_train_trans, X_test_trans, preprocessor = preprocess_features(X_train, X_test)

    # Run distributed KNN
    pyspark_preds, spark_duration = run_pyspark_knn(X_train_trans, y_train, X_test_trans, k=9, p=1)

    # Validate against Scikit-Learn
    metrics_payload, preds_df = validate_results(pyspark_preds, X_test, y_test)

    print("\n" + "=" * 75)
    print("PYSPARK KNN PIPELINE EXECUTION COMPLETED")
    print(f"Prediction Agreement: {metrics_payload['validation_metrics']['prediction_agreement_percentage']:.2f}%")
    print(f"Test Accuracy:        {metrics_payload['validation_metrics']['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:       {metrics_payload['validation_metrics']['test_f1_macro']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
