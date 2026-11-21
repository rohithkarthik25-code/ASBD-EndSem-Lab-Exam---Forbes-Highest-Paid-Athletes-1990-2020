================================================================================
                    BIG DATA & ML PRACTICAL EXAM / HACKATHON
                             RAPID RESPONSE GUIDE
================================================================================

CRITICAL DIRECTIVE:
DO NOT blindly apply classification algorithms when the task is Regression,
Clustering, Dimensionality Reduction, or Association Rule Mining!
First determine the Task Type and the Target Column.

--------------------------------------------------------------------------------
THE 13-STEP EXAM WORKFLOW
--------------------------------------------------------------------------------

STEP 1: READ THE PROBLEM STATEMENT (5-10 min)
  - What is the business/practical objective?
  - Identify problem category:
      * Supervised Classification (categorical target)
      * Supervised Regression (continuous numeric target)
      * Unsupervised Clustering (no target: group similar records)
      * Association Rule Mining (transactional / co-occurrence)
      * Big Data / Distributed Processing (explicit PySpark / RDD requirement)

STEP 2: IDENTIFY TARGET & METRIC (2 min)
  - Exact target column name (case-sensitive).
  - Exact evaluation metric requested:
      * Classification: Accuracy, F1-macro, F1-weighted, ROC-AUC, Recall/Precision.
      * Regression: RMSE, MAE, R2 score.
      * Clustering: Silhouette Score, Davies-Bouldin, Inertia / Elbow.
      * Association: Support, Confidence, Lift.

STEP 3: INGEST DATASET (2-5 min)
  - Place CSV inside hackathon_live/data/
  - Or use kagglehub / pd.read_csv('hackathon_live/data/<file>.csv')
  - For PySpark: spark_load_csv(spark, 'hackathon_live/data/<file>.csv')

STEP 4: PROFILE DATASET (NON-DESTRUCTIVE) (5 min)
  - Shape: df.shape (rows, columns)
  - Schema: df.dtypes, df.info()
  - Missing Values: df.isnull().sum() (flag any > 50% missing)
  - Duplicates: df.duplicated().sum()
  - Target Distribution:
      * Classification: df[target].value_counts(normalize=True) -> Check for Class Imbalance!
      * Regression: df[target].describe(), skewness, outliers.

STEP 5: EXPLORATORY DATA ANALYSIS (EDA) (5-10 min)
  - Identify ID columns, timestamps, leaks, or text descriptions.
  - Correlation with target: df.corr(numeric_only=True)[target].sort_values()
  - Check feature distributions & high-cardinality categoricals.

STEP 6: TRAIN/TEST SPLIT BEFORE ANY FIT/TRANSFORMATION (CRITICAL)
  - ALWAYS split train and test FIRST:
      X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.2, random_state=42, stratify=y if cls else None)
  - NEVER fit Imputer, Scaler, or Encoder on the entire dataset.
  - Fit on train, transform on test. (exam_toolkit handles this inside pipelines).

STEP 7: BUILD A WORKING BASELINE EARLY (10 min)
  - Start with a simple, robust baseline (e.g., LogisticRegression or Ridge / Dummy).
  - Record the baseline validation score. This is your safety net.

STEP 8: MULTI-ALGORITHM COMPARISON (15 min)
  - Use exam_toolkit to run all relevant models in parallel:
      * Scikit-Learn Classification: et.run_all_classifiers(X_train, y_train, X_test, y_test)
      * Scikit-Learn Regression: et.run_all_regressors(X_train, y_train, X_test, y_test)
      * PySpark Classification: et.spark_run_classifiers(train_df, test_df, target_col='...')
      * PySpark Regression: et.spark_run_regressors(train_df, test_df, target_col='...')
      * Clustering: et.cluster_benchmark(X, k_range=range(2, 7))
  - Review the leaderboard DataFrame.

STEP 9: TUNE THE WINNER (10-15 min)
  - Pick top 1-2 performing architectures (e.g., XGBoost, Random Forest).
  - Run GridSearchCV / spark_tune_model on the training set only (Cross-Validation).
  - Tune max_depth, n_estimators, learning_rate, reg_param.

STEP 10: RIGOROUS TEST SET EVALUATION (5 min)
  - Evaluate best tuned model on the untouched test set.
  - Print Confusion Matrix, Classification Report, or Residual Plot.
  - Confirm metrics are realistic and finite (NOT 1.0000 unless trivial toy data).

STEP 11: TARGET LEAKAGE & INTEGRITY CHECK (5 min)
  - Verify target was not included in feature matrix.
  - Check correlation of top features with target (anything > 0.98 is suspicious).
  - Confirm train and test sets have no overlap.

STEP 12: SAVE OUTPUTS & DELIVERABLES (5 min)
  - Predictions CSV -> hackathon_live/results/predictions.csv
  - Leaderboard / Metrics -> hackathon_live/results/metrics_leaderboard.csv
  - Figures -> hackathon_live/results/*.png
  - Final Report / Summary notes -> hackathon_live/reports/SUMMARY.md

STEP 13: RESTART KERNEL & RUN ALL (5-10 min)
  - Restart the notebook / Python interpreter.
  - Run All Cells from top to bottom.
  - Ensure zero crashes, zero hardcoded paths, and clean reproducibility.

--------------------------------------------------------------------------------
LAB EXPERIMENT CHEAT SHEET
--------------------------------------------------------------------------------
Exp Week  | Topic / Methodology                | Primary Algorithms / Tools
----------+------------------------------------+---------------------------------
Week 1    | Data Exploration & Preprocessing   | Pandas, Seaborn, Null handling
Week 2    | Association Rule Mining            | Apriori, FP-Growth (mlxtend)
Week 3    | Decision Trees & Naive Bayes       | DecisionTreeClassifier, GaussianNB
Week 4    | Linear & Logistic Regression       | LinearRegression, LogisticRegression
Week 5    | Clustering (K-Means & DBSCAN)      | KMeans, DBSCAN, Silhouette Score
Week 6    | Hierarchical Clustering            | AgglomerativeClustering, Dendrogram
Week 7    | Ensembles (Random Forest, XGBoost) | RandomForest, GradientBoosting, XGB
Week 8    | Dimensionality Reduction           | PCA, t-SNE
Week 9    | PySpark DataFrame & Spark SQL      | PySpark DataFrame, SQL queries
Week 10   | PySpark MLlib Classification       | Spark Pipeline, Logistic, RF, GBT
Week 11   | PySpark MLlib Regression & Cluster | Spark LinearReg, Spark KMeans
Week 12   | Distributed Architecture & MapRed  | MapReduce simulation / RDD ops
================================================================================
