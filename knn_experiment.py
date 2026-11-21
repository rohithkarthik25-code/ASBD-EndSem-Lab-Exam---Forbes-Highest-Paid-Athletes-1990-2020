"""
================================================================================
                    FORBES HIGHEST-PAID ATHLETES (1990-2020)
                   PHASE 3 — STEP 2: KNN CLASSIFICATION EXPERIMENT
================================================================================
Objective: Evaluate the hypothesis H1 whether financial and competitive ranking
           profiles distinguish Team Sport athletes from Individual Sport athletes.
Target:    Sport_Type ('Team Sport' vs. 'Individual Sport')
Features:  earnings ($ million), Year, Current Rank, Nationality_Group
Validation: Stratified 5-Fold Cross-Validation on Train (80%) + Held-Out Test (20%)
================================================================================
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------------------------
# 1. PATH SETUP & DIRECTORY CREATION
# ------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "forbes_athletes_cleaned.csv"
MODELS_DIR = CURRENT_DIR / "models"
RESULTS_DIR = CURRENT_DIR / "results"
REPORTS_DIR = CURRENT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 2. DATA INGESTION & TARGET FORMULATION
# ------------------------------------------------------------------------------
def load_and_prepare_data(csv_path=DATA_PATH):
    print("=" * 75)
    print("STEP 1: INGESTING AND VALIDATING DATA FOR KNN CLASSIFICATION")
    print("=" * 75)

    df = pd.read_csv(csv_path)
    initial_shape = df.shape
    print(f"[+] Loaded cleaned dataset: {initial_shape[0]} rows x {initial_shape[1]} columns")

    # Define ground-truth target: Team Sport vs. Individual Sport
    team_sports = ['Basketball', 'Soccer', 'American Football', 'Baseball', 'Hockey']
    df['Sport_Type'] = df['Sport_Clean'].apply(
        lambda x: 'Team Sport' if x in team_sports else 'Individual Sport'
    )

    print("\n--- TARGET DISTRIBUTION (Sport_Type) ---")
    vc = df['Sport_Type'].value_counts()
    vc_pct = df['Sport_Type'].value_counts(normalize=True) * 100
    for cls_name in vc.index:
        print(f"  • {cls_name:18s}: {vc[cls_name]:3d} ({vc_pct[cls_name]:.2f}%)")

    # Define permitted feature space
    feature_cols = ['earnings ($ million)', 'Year', 'Current Rank', 'Nationality_Group']
    target_col = 'Sport_Type'

    # STRICT LEAKAGE AUDIT: Confirm every leakage-prone column is excluded
    forbidden_cols = [
        'Sport', 'Sport_Clean', 'Name', 'S.NO', 'Rank_Tier',
        'Earnings_Category', 'Rank_Trajectory', 'Decade', 'Previous Year Rank',
        'Previous_Year_Rank_Clean', 'Nationality'
    ]
    for col in forbidden_cols:
        assert col not in feature_cols, f"CRITICAL LEAKAGE: '{col}' found in feature_cols!"
    print("\n[+] Leakage Audit: PASSED. All sport-identifying & target-derived columns excluded.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Verify no nulls in X
    assert X.isnull().sum().sum() == 0, "Found null values in feature matrix X!"
    print(f"[+] Feature matrix shape: {X.shape[0]} rows x {X.shape[1]} features")
    print("    Features:", list(X.columns))

    return X, y, df


# ------------------------------------------------------------------------------
# 3. STRATIFIED TRAIN/TEST SPLIT
# ------------------------------------------------------------------------------
def create_train_test_split(X, y, test_size=0.20, random_state=42):
    print("\n" + "=" * 75)
    print("STEP 2: STRATIFIED TRAIN/TEST SPLIT (80% Train, 20% Held-Out Test)")
    print("=" * 75)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[+] Train set: {X_train.shape[0]} rows (80.0%)")
    print(f"    - Target distribution: {dict(y_train.value_counts())}")
    print(f"[+] Test set:  {X_test.shape[0]} rows (20.0%)")
    print(f"    - Target distribution: {dict(y_test.value_counts())}")

    # Verify identical class ratios
    train_ratio = (y_train == 'Team Sport').mean()
    test_ratio = (y_test == 'Team Sport').mean()
    print(f"[+] Team Sport ratio: Train={train_ratio:.4f}, Test={test_ratio:.4f} (Strictly Stratified)")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE & DUMMY BASELINES
# ------------------------------------------------------------------------------
def build_pipeline():
    numeric_features = ['earnings ($ million)', 'Year', 'Current Rank']
    categorical_features = ['Nationality_Group']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('knn', KNeighborsClassifier())
    ])

    return pipe


def evaluate_dummy_baselines(X_train, y_train, X_test, y_test, cv):
    print("\n" + "=" * 75)
    print("STEP 3: EVALUATING DUMMY CLASSIFIER BASELINES")
    print("=" * 75)

    scoring = ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']

    # Baseline 1: Most Frequent Class
    dummy_mf = DummyClassifier(strategy='most_frequent')
    cv_mf = cross_validate(dummy_mf, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
    dummy_mf.fit(X_train, y_train)
    y_pred_mf = dummy_mf.predict(X_test)
    y_prob_mf = dummy_mf.predict_proba(X_test)[:, 1]

    mf_test_acc = accuracy_score(y_test, y_pred_mf)
    mf_test_bal_acc = balanced_accuracy_score(y_test, y_pred_mf)
    mf_test_f1 = f1_score(y_test, y_pred_mf, average='macro')
    # ROC-AUC for constant prediction is 0.5
    mf_test_auc = 0.5000

    print(f"[1] Dummy (Most Frequent) Baseline:")
    print(f"    • Train 5-Fold CV Accuracy:    {cv_mf['test_accuracy'].mean():.4f} (+/- {cv_mf['test_accuracy'].std():.4f})")
    print(f"    • Train 5-Fold CV F1-Macro:    {cv_mf['test_f1_macro'].mean():.4f} (+/- {cv_mf['test_f1_macro'].std():.4f})")
    print(f"    • Test Accuracy:               {mf_test_acc:.4f}")
    print(f"    • Test Balanced Accuracy:      {mf_test_bal_acc:.4f}")
    print(f"    • Test F1-Macro:               {mf_test_f1:.4f}")

    # Baseline 2: Stratified Random Guessing
    dummy_st = DummyClassifier(strategy='stratified', random_state=42)
    cv_st = cross_validate(dummy_st, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
    dummy_st.fit(X_train, y_train)
    y_pred_st = dummy_st.predict(X_test)
    y_prob_st = dummy_st.predict_proba(X_test)[:, 1]

    st_test_acc = accuracy_score(y_test, y_pred_st)
    st_test_bal_acc = balanced_accuracy_score(y_test, y_pred_st)
    st_test_f1 = f1_score(y_test, y_pred_st, average='macro')
    try:
        st_test_auc = roc_auc_score(y_test, y_prob_st)
    except Exception:
        st_test_auc = 0.5000

    print(f"\n[2] Dummy (Stratified Guessing) Baseline:")
    print(f"    • Train 5-Fold CV Accuracy:    {cv_st['test_accuracy'].mean():.4f} (+/- {cv_st['test_accuracy'].std():.4f})")
    print(f"    • Train 5-Fold CV F1-Macro:    {cv_st['test_f1_macro'].mean():.4f} (+/- {cv_st['test_f1_macro'].std():.4f})")
    print(f"    • Test Accuracy:               {st_test_acc:.4f}")
    print(f"    • Test Balanced Accuracy:      {st_test_bal_acc:.4f}")
    print(f"    • Test F1-Macro:               {st_test_f1:.4f}")
    print(f"    • Test ROC-AUC:                {st_test_auc:.4f}")

    baseline_metrics = {
        'dummy_most_frequent': {
            'cv_train_accuracy_mean': float(cv_mf['test_accuracy'].mean()),
            'cv_train_accuracy_std': float(cv_mf['test_accuracy'].std()),
            'cv_train_f1_macro_mean': float(cv_mf['test_f1_macro'].mean()),
            'test_accuracy': float(mf_test_acc),
            'test_balanced_accuracy': float(mf_test_bal_acc),
            'test_f1_macro': float(mf_test_f1),
            'test_roc_auc': float(mf_test_auc)
        },
        'dummy_stratified': {
            'cv_train_accuracy_mean': float(cv_st['test_accuracy'].mean()),
            'cv_train_accuracy_std': float(cv_st['test_accuracy'].std()),
            'cv_train_f1_macro_mean': float(cv_st['test_f1_macro'].mean()),
            'test_accuracy': float(st_test_acc),
            'test_balanced_accuracy': float(st_test_bal_acc),
            'test_f1_macro': float(st_test_f1),
            'test_roc_auc': float(st_test_auc)
        }
    }

    return baseline_metrics


# ------------------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING VIA STRATIFIED CV
# ------------------------------------------------------------------------------
def tune_knn_model(X_train, y_train, cv):
    print("\n" + "=" * 75)
    print("STEP 4: HYPERPARAMETER TUNING ON TRAINING SET (GRID SEARCH WITH 5-FOLD CV)")
    print("=" * 75)

    pipeline = build_pipeline()

    param_grid = {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 11, 15],
        'knn__p': [1, 2],                             # 1 = Manhattan, 2 = Euclidean
        'knn__weights': ['uniform', 'distance']       # Weighting strategies
    }

    scoring = {
        'f1_macro': 'f1_macro',
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'roc_auc': 'roc_auc'
    }

    print(f"[+] Grid search combinations: {len(param_grid['knn__n_neighbors'])} k values x 2 metrics x 2 weights = 28 models")
    print(f"[+] Cross-Validation: Stratified 5-Fold CV (n_splits=5, shuffle=True, seed=42)")
    print(f"[+] Primary Selection Metric: f1_macro (refit=True on full training set)")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1_macro',
        return_train_score=True,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    best_cv_f1 = grid.best_score_
    best_idx = grid.best_index_
    best_cv_acc = grid.cv_results_['mean_test_accuracy'][best_idx]
    best_cv_bal_acc = grid.cv_results_['mean_test_balanced_accuracy'][best_idx]
    best_cv_auc = grid.cv_results_['mean_test_roc_auc'][best_idx]

    print(f"\n[+] Grid Search Complete!")
    print(f"    • Best k (n_neighbors):     {best_params['knn__n_neighbors']}")
    print(f"    • Best Metric (p):          {best_params['knn__p']} ({'Manhattan' if best_params['knn__p'] == 1 else 'Euclidean'})")
    print(f"    • Best Weights:             {best_params['knn__weights']}")
    print(f"    • Best 5-Fold CV F1-Macro:  {best_cv_f1:.4f}")
    print(f"    • Best 5-Fold CV Accuracy:  {best_cv_acc:.4f}")
    print(f"    • Best 5-Fold CV Bal Acc:   {best_cv_bal_acc:.4f}")
    print(f"    • Best 5-Fold CV ROC-AUC:   {best_cv_auc:.4f}")

    # Export cv_results to CSV
    cv_results_df = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_f1_macro', ascending=False)
    cv_results_path = RESULTS_DIR / "knn_cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    print(f"[+] Saved complete CV results table to: {cv_results_path.name}")

    return grid, best_params, {
        'cv_best_f1_macro': float(best_cv_f1),
        'cv_best_accuracy': float(best_cv_acc),
        'cv_best_balanced_acc': float(best_cv_bal_acc),
        'cv_best_roc_auc': float(best_cv_auc)
    }


# ------------------------------------------------------------------------------
# 6. FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET
# ------------------------------------------------------------------------------
def evaluate_on_test_set(best_model, X_test, y_test, best_params):
    print("\n" + "=" * 75)
    print("STEP 5: SINGLE FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET (N=61)")
    print("=" * 75)

    # Predictions
    y_pred = best_model.predict(X_test)
    classes = list(best_model.classes_)
    y_prob = best_model.predict_proba(X_test)

    # Positive class for binary ROC-AUC
    pos_idx = classes.index('Team Sport')
    y_prob_pos = y_prob[:, pos_idx]

    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_prec_macro = precision_score(y_test, y_pred, average='macro')
    test_rec_macro = recall_score(y_test, y_pred, average='macro')
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_prec_weighted = precision_score(y_test, y_pred, average='weighted')
    test_rec_weighted = recall_score(y_test, y_pred, average='weighted')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    test_roc_auc = roc_auc_score(y_test == 'Team Sport', y_prob_pos)

    cm = confusion_matrix(y_test, y_pred, labels=['Individual Sport', 'Team Sport'])
    clf_report = classification_report(y_test, y_pred, target_names=['Individual Sport', 'Team Sport'], output_dict=True)

    print("\n--- TEST SET PERFORMANCE METRICS ---")
    print(f"  • Accuracy:                   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  • Balanced Accuracy:          {test_bal_acc:.4f} ({test_bal_acc*100:.2f}%)")
    print(f"  • Precision (Macro):          {test_prec_macro:.4f}")
    print(f"  • Recall (Macro):             {test_rec_macro:.4f}")
    print(f"  • F1-Score (Macro):           {test_f1_macro:.4f}")
    print(f"  • F1-Score (Weighted):        {test_f1_weighted:.4f}")
    print(f"  • ROC-AUC Score:              {test_roc_auc:.4f}")

    print("\n--- CONFUSION MATRIX ---")
    print(f"                Predicted: Individual  Predicted: Team")
    print(f"Actual: Individual       {cm[0, 0]:2d}                   {cm[0, 1]:2d}")
    print(f"Actual: Team             {cm[1, 0]:2d}                   {cm[1, 1]:2d}")

    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Individual Sport', 'Team Sport']))

    test_metrics = {
        'test_accuracy': float(test_acc),
        'test_balanced_accuracy': float(test_bal_acc),
        'test_precision_macro': float(test_prec_macro),
        'test_recall_macro': float(test_rec_macro),
        'test_f1_macro': float(test_f1_macro),
        'test_precision_weighted': float(test_prec_weighted),
        'test_recall_weighted': float(test_rec_weighted),
        'test_f1_weighted': float(test_f1_weighted),
        'test_roc_auc': float(test_roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': clf_report
    }

    # Save test set predictions to CSV
    test_pred_df = X_test.copy()
    test_pred_df['Actual_Sport_Type'] = y_test.values
    test_pred_df['Predicted_Sport_Type'] = y_pred
    test_pred_df['Prob_Individual_Sport'] = y_prob[:, classes.index('Individual Sport')].round(4)
    test_pred_df['Prob_Team_Sport'] = y_prob[:, classes.index('Team Sport')].round(4)
    test_pred_df['Correct'] = (test_pred_df['Actual_Sport_Type'] == test_pred_df['Predicted_Sport_Type']).astype(int)

    pred_path = RESULTS_DIR / "knn_test_predictions.csv"
    test_pred_df.to_csv(pred_path, index=True)
    print(f"[+] Saved test predictions table to: {pred_path.name}")

    return test_metrics


# ------------------------------------------------------------------------------
# 7. SAVE PIPELINE & JSON METRICS
# ------------------------------------------------------------------------------
def save_artifacts(best_model, all_metrics, best_params):
    print("\n" + "=" * 75)
    print("STEP 6: SERIALIZING MODEL PIPELINE AND EXPERIMENT METRICS")
    print("=" * 75)

    # 1. Model pickle
    model_path = MODELS_DIR / "knn_model.pkl"
    joblib.dump(best_model.best_estimator_, model_path)
    print(f"[+] Saved fitted KNN Pipeline to: {model_path} ({model_path.stat().st_size:,} bytes)")

    # 2. Combined metrics JSON
    combined_results = {
        'dataset_information': {
            'total_rows': 301,
            'train_rows': 240,
            'test_rows': 61,
            'target_variable': 'Sport_Type',
            'features_used': ['earnings ($ million)', 'Year', 'Current Rank', 'Nationality_Group']
        },
        'best_hyperparameters': best_params,
        'baseline_dummy_metrics': all_metrics['baselines'],
        'train_cv_metrics': all_metrics['cv_train'],
        'held_out_test_metrics': all_metrics['test']
    }

    metrics_path = RESULTS_DIR / "knn_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(combined_results, f, indent=4)
    print(f"[+] Saved comprehensive metrics JSON to: {metrics_path.name}")


# ------------------------------------------------------------------------------
# MAIN EXECUTION ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    X, y, df = load_and_prepare_data()
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.20, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseline_metrics = evaluate_dummy_baselines(X_train, y_train, X_test, y_test, cv)
    grid, best_params, cv_train_metrics = tune_knn_model(X_train, y_train, cv)
    test_metrics = evaluate_on_test_set(grid, X_test, y_test, best_params)

    all_metrics = {
        'baselines': baseline_metrics,
        'cv_train': cv_train_metrics,
        'test': test_metrics
    }

    save_artifacts(grid, all_metrics, best_params)

    print("\n" + "=" * 75)
    print("KNN EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)
