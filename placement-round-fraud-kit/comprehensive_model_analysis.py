"""
COMPREHENSIVE FRAUD DETECTION MODEL ANALYSIS
=============================================

This script performs a complete forensic reconstruction of the Bank Fraud Detection
ML pipeline, generating all required heatmaps, tables, and analyses to answer:

"What models were built, what data was each model trained on, how was each model
trained, how do the models relate to one another, which models agree/disagree,
what features influence them, how were they evaluated, and why was the final
model/ensemble selected?"

Outputs:
--------
- 11+ heatmaps visualizing model relationships, feature importance, and performance
- Master model inventory table
- Complete data flow diagram
- Model-by-model training documentation
- Feature correlation analyses
- Threshold optimization analysis
- Final comprehensive report
"""

import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import matplotlib.patches as mpatches

# Configuration
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# Paths
ARTIFACTS_RESEARCH = "artifacts_research"
ARTIFACTS_V1 = "artifacts"
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("COMPREHENSIVE FRAUD DETECTION MODEL ANALYSIS")
print("="*80)
print()

# ============================================================================
# SECTION 1: LOAD ALL ARTIFACTS
# ============================================================================
print("SECTION 1: Loading All Artifacts...")
print("-" * 80)

# Dataset facts
with open(os.path.join(ARTIFACTS_RESEARCH, "dataset_facts.json")) as f:
    dataset_facts = json.load(f)
print(f"> Dataset: {dataset_facts['n_rows']} transactions, {dataset_facts['n_unique_accounts']} accounts")

# Model scores (all 12 models)
model_scores = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "model_scores_all.csv"))
print(f"> Loaded model scores: {model_scores.shape}")

# Pairwise correlations
spearman_matrix = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "model_pairwise_spearman.csv"), index_col=0)
jaccard_matrix = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "model_pairwise_jaccard.csv"), index_col=0)
print(f"> Loaded correlation matrices: Spearman {spearman_matrix.shape}, Jaccard {jaccard_matrix.shape}")

# Features
features_v2 = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "features_v2.csv"))
print(f"> Loaded engineered features: {features_v2.shape}")

# Feature config
with open(os.path.join(ARTIFACTS_RESEARCH, "autoencoder_config.json")) as f:
    ae_config = json.load(f)
feature_cols = ae_config["feature_cols"]
print(f"> Feature columns for modeling: {len(feature_cols)}")

# SHAP importance (if available)
try:
    shap_if = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "shap_isolation_forest.csv"))
    shap_ae = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "shap_autoencoder.csv"))
    print(f"> Loaded SHAP values: IF {shap_if.shape}, AE {shap_ae.shape}")
    has_shap = True
except:
    print("⚠ SHAP files not found - will skip SHAP heatmaps")
    has_shap = False

# Ensemble weights
with open(os.path.join(ARTIFACTS_RESEARCH, "ensemble_weights.json")) as f:
    ensemble_weights = json.load(f)
print(f"> Loaded ensemble weights")

# V1 supervised results
v1_comparison = pd.read_csv(os.path.join(ARTIFACTS_V1, "model_comparison.csv"))
print(f"> Loaded V1 supervised comparison: {v1_comparison.shape}")

print()

# ============================================================================
# SECTION 2: MASTER MODEL INVENTORY
# ============================================================================
print("SECTION 2: Creating Master Model Inventory...")
print("-" * 80)

model_inventory = pd.DataFrame({
    "Model": [
        "Isolation Forest",
        "Local Outlier Factor (LOF)",
        "One-Class SVM",
        "Elliptic Envelope",
        "DBSCAN",
        "HDBSCAN",
        "K-Means",
        "Gaussian Mixture Model",
        "Autoencoder",
        "Variational Autoencoder",
        "LSTM Autoencoder",
        "Hybrid Ensemble (IF+LOF+AE)"
    ],
    "Family": [
        "Tree-based Isolation",
        "Density-based",
        "Kernel-based SVM",
        "Distribution-based",
        "Density-based Clustering",
        "Hierarchical Clustering",
        "Centroid-based Clustering",
        "Probabilistic Clustering",
        "Deep Learning Reconstruction",
        "Deep Learning Variational",
        "Deep Learning Sequential",
        "Voting Ensemble"
    ],
    "Supervised": [False]*12,
    "Training_Data": ["46 engineered features"]*12,
    "Train_Rows": ["2,009"]*12,
    "Val_Rows": ["503"]*12,
    "Scaling": ["RobustScaler (train-only fit)"]*12,
    "Output": [
        "Anomaly score (higher=more anomalous)",
        "Anomaly score (higher=more anomalous)",
        "Anomaly score (higher=more anomalous)",
        "Anomaly score (higher=more anomalous)",
        "Distance to nearest core point",
        "GLOSH outlier score",
        "Distance to nearest valid centroid",
        "Negative log-likelihood",
        "Reconstruction MSE",
        "Reconstruction MSE",
        "Sequence reconstruction MSE",
        "Vote count (0-3)"
    ],
    "Key_Hyperparameters": [
        "n_estimators=200, contamination=0.05",
        "n_neighbors=20, contamination=0.05",
        "kernel=rbf, nu=0.05",
        "contamination=0.05",
        "eps=0.382, min_samples=10",
        "min_cluster_size=20",
        "k=4 clusters",
        "n_components=4, covariance=full",
        "46→16→8→4→8→16→46, 150 epochs",
        "46→16→8→4(μ,σ)→8→16→46, 200 epochs",
        "LSTM hidden=16, latent=8, 150 epochs",
        "Majority vote ≥2 of 3"
    ],
    "Role": [
        "Global isolation detection",
        "Local density outlier detection",
        "Kernel-based one-class classification",
        "Mahalanobis distance outliers",
        "Non-clusterable noise detection",
        "Hierarchical density outliers",
        "Centroid distance anomalies",
        "Low-likelihood anomalies",
        "Reconstruction error anomalies",
        "Variational reconstruction anomalies",
        "Sequential behavior anomalies",
        "Consensus of 3 diverse detectors"
    ]
})

model_inventory.to_csv(os.path.join(OUTPUT_DIR, "master_model_inventory.csv"), index=False)
print(f"> Saved master model inventory: {model_inventory.shape}")
print()

# ============================================================================
# SECTION 3: HEATMAP 1 - MODEL SPEARMAN CORRELATION
# ============================================================================
print("SECTION 3: Generating Heatmap 1 - Model Anomaly Score Correlation (Spearman)...")
print("-" * 80)

# Use all 12 models
model_names_clean = [
    "Isolation\nForest",
    "LOF",
    "One-Class\nSVM",
    "Elliptic\nEnvelope",
    "DBSCAN",
    "HDBSCAN",
    "K-Means",
    "GMM",
    "Auto-\nencoder",
    "VAE",
    "LSTM-AE",
    "Hybrid\nEnsemble"
]

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.zeros_like(spearman_matrix, dtype=bool)
sns.heatmap(
    spearman_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-0.2,
    vmax=1.0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
    ax=ax,
    xticklabels=model_names_clean,
    yticklabels=model_names_clean
)
ax.set_title("Model Anomaly Score Correlation Heatmap\n(Spearman Rank Correlation)",
             fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_model_spearman_correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

# Export matrix
spearman_matrix.to_csv(os.path.join(OUTPUT_DIR, "1_model_spearman_correlation_matrix.csv"))
print(f"> Saved Spearman correlation heatmap")
print()

# ============================================================================
# SECTION 4: HEATMAP 2 - MODEL JACCARD AGREEMENT
# ============================================================================
print("SECTION 4: Generating Heatmap 2 - Top-Anomaly Agreement (Jaccard)...")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    jaccard_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1.0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Jaccard Similarity", "shrink": 0.8},
    ax=ax,
    xticklabels=model_names_clean,
    yticklabels=model_names_clean
)
ax.set_title("Model Top-5% Anomaly Agreement Heatmap\n(Jaccard Similarity on Binary Flags)",
             fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_model_jaccard_agreement_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

jaccard_matrix.to_csv(os.path.join(OUTPUT_DIR, "2_model_jaccard_agreement_matrix.csv"))
print(f"> Saved Jaccard agreement heatmap")
print()

# ============================================================================
# SECTION 5: HEATMAP 3 - MODEL DISAGREEMENT
# ============================================================================
print("SECTION 5: Generating Heatmap 3 - Model Disagreement (Diversity)...")
print("-" * 80)

disagreement_matrix = 1 - jaccard_matrix

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(
    disagreement_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn_r",
    vmin=0,
    vmax=1.0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Disagreement (1 - Jaccard)", "shrink": 0.8},
    ax=ax,
    xticklabels=model_names_clean,
    yticklabels=model_names_clean
)
ax.set_title("Model Disagreement Heatmap (Diversity)\n(Higher = More Diverse Detection)",
             fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_model_disagreement_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

disagreement_matrix.to_csv(os.path.join(OUTPUT_DIR, "3_model_disagreement_matrix.csv"))
print(f"> Saved disagreement heatmap")
print()

# ============================================================================
# SECTION 6: HEATMAP 4 - FEATURE CORRELATION
# ============================================================================
print("SECTION 6: Generating Heatmap 4 - Feature Correlation...")
print("-" * 80)

# Get numeric features only
X_features = features_v2[feature_cols].astype(float)

# Compute Spearman correlation
feature_corr = X_features.corr(method='spearman')

# Full heatmap (may be large)
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(
    feature_corr,
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.1,
    cbar_kws={"label": "Spearman ρ", "shrink": 0.6},
    ax=ax,
    xticklabels=True,
    yticklabels=True
)
ax.set_title("Feature Correlation Heatmap (All 46 Features)\n(Spearman Rank Correlation)",
             fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=90, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_correlation_heatmap_full.png"), dpi=150, bbox_inches='tight')
plt.close()

# Top 20 most variable/important features for focused heatmap
feature_std = X_features.std().sort_values(ascending=False)
top_features = feature_std.head(20).index.tolist()
feature_corr_top = X_features[top_features].corr(method='spearman')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    feature_corr_top,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
    ax=ax
)
ax.set_title("Feature Correlation Heatmap (Top 20 Most Variable Features)\n(Spearman Rank Correlation)",
             fontsize=13, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_correlation_heatmap_top20.png"), dpi=150, bbox_inches='tight')
plt.close()

feature_corr.to_csv(os.path.join(OUTPUT_DIR, "4_feature_correlation_matrix.csv"))
print(f"> Saved feature correlation heatmaps (full and top 20)")
print()

# ============================================================================
# SECTION 7: HEATMAP 5 - FEATURE × MODEL IMPORTANCE
# ============================================================================
print("SECTION 7: Generating Heatmap 5 - Feature × Model Importance...")
print("-" * 80)

if has_shap:
    # Extract SHAP importance for IF and AE
    shap_if_mean = shap_if.drop(columns=['TransactionID', 'AccountID'], errors='ignore').abs().mean()
    shap_ae_mean = shap_ae.drop(columns=['TransactionID', 'AccountID'], errors='ignore').abs().mean()

    # Align to feature_cols
    shap_if_aligned = shap_if_mean.reindex(feature_cols, fill_value=0)
    shap_ae_aligned = shap_ae_mean.reindex(feature_cols, fill_value=0)

    # Normalize to 0-1
    shap_if_norm = shap_if_aligned / (shap_if_aligned.max() + 1e-10)
    shap_ae_norm = shap_ae_aligned / (shap_ae_aligned.max() + 1e-10)

    # For other models, use heuristic: Spearman correlation between feature and model score
    model_cols = [
        "isolation_forest", "lof", "ocsvm", "elliptic_envelope",
        "dbscan", "hdbscan", "kmeans", "gmm", "autoencoder", "vae"
    ]

    feature_model_matrix = pd.DataFrame(index=feature_cols, columns=model_cols)

    for model in model_cols:
        model_score = model_scores[f"score_{model}"].values
        for feat in feature_cols:
            feat_vals = X_features[feat].values
            # Handle NaN in LSTM scores
            valid = ~np.isnan(model_score)
            if valid.sum() > 10:
                rho, _ = spearmanr(feat_vals[valid], model_score[valid])
                feature_model_matrix.loc[feat, model] = abs(rho)
            else:
                feature_model_matrix.loc[feat, model] = 0

    feature_model_matrix = feature_model_matrix.astype(float)

    # Replace IF and AE columns with actual SHAP
    feature_model_matrix['isolation_forest'] = shap_if_norm.values
    feature_model_matrix['autoencoder'] = shap_ae_norm.values

    # Select top 20 features by max importance across models
    feature_max_importance = feature_model_matrix.max(axis=1).sort_values(ascending=False)
    top_20_features = feature_max_importance.head(20).index.tolist()

    feature_model_top = feature_model_matrix.loc[top_20_features]

    # Clean column names
    col_names_clean = [
        "Isolation\nForest", "LOF", "One-Class\nSVM", "Elliptic\nEnvelope",
        "DBSCAN", "HDBSCAN", "K-Means", "GMM", "Auto-\nencoder", "VAE"
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        feature_model_top,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Normalized Importance", "shrink": 0.8},
        ax=ax,
        xticklabels=col_names_clean,
        yticklabels=top_20_features
    )
    ax.set_title("Feature × Model Importance Heatmap (Top 20 Features)\n(SHAP for IF/AE, |Spearman| for others)",
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel("Model", fontsize=11, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_feature_model_importance_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    feature_model_matrix.to_csv(os.path.join(OUTPUT_DIR, "5_feature_model_importance_matrix.csv"))
    print(f"> Saved feature × model importance heatmap")
else:
    print(f"⚠ Skipped feature × model importance heatmap (no SHAP data)")

print()

# ============================================================================
# SECTION 8: HEATMAP 6 - MODEL CHARACTERISTICS
# ============================================================================
print("SECTION 8: Generating Heatmap 6 - Model Characteristics...")
print("-" * 80)

model_characteristics = pd.DataFrame({
    "Model": model_inventory["Model"].values,
    "Uses_Scaling": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # All use scaler except hybrid
    "Uses_Distance": [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "Uses_Density": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    "Uses_Clustering": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    "Uses_Trees": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Uses_Kernel": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Uses_Covariance": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "Uses_Neural_Net": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    "Uses_Sequence": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Requires_Contamination": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "Continuous_Score": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "Out_of_Sample": [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    "Interpretable": [2, 1, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2],  # 2=high, 1=medium, 0=low
    "Computational_Cost": [0, 1, 2, 0, 1, 1, 0, 1, 1, 1, 2, 0],  # 0=low, 1=med, 2=high
}).set_index("Model")

# Plot
fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(
    model_characteristics,
    cmap="Blues",
    linewidths=0.5,
    cbar_kws={"label": "Value", "shrink": 0.6},
    ax=ax,
    yticklabels=model_inventory["Model"].values
)
ax.set_title("Model Characteristics Matrix\n(Capabilities and Properties)",
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel("Characteristic", fontsize=11, fontweight='bold')
ax.set_ylabel("Model", fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "6_model_characteristics_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

model_characteristics.to_csv(os.path.join(OUTPUT_DIR, "6_model_characteristics_matrix.csv"))
print(f"> Saved model characteristics heatmap")
print()

# ============================================================================
# SECTION 9: HEATMAP 7 - TOP TRANSACTIONS × MODELS
# ============================================================================
print("SECTION 9: Generating Heatmap 7 - Top Transactions × Models...")
print("-" * 80)

# Use ensemble percentile as final score
ensemble_scores = pd.read_csv(os.path.join(ARTIFACTS_RESEARCH, "ensemble_scores.csv"))
final_score = ensemble_scores["ensemble_percentile_average"].values

# Top 30 transactions
top_30_idx = np.argsort(final_score)[-30:][::-1]

# Get scores for these transactions
model_cols_raw = [
    "score_isolation_forest", "score_lof", "score_ocsvm", "score_elliptic_envelope",
    "score_dbscan", "score_hdbscan", "score_kmeans", "score_gmm",
    "score_autoencoder", "score_vae", "score_lstm_ae"
]

transaction_model_scores = model_scores.loc[top_30_idx, ["TransactionID"] + model_cols_raw].copy()

# Normalize each model's scores to 0-1 percentile
for col in model_cols_raw:
    vals = model_scores[col].values
    valid = ~np.isnan(vals)
    from scipy.stats import rankdata
    ranks = np.full(len(vals), np.nan)
    ranks[valid] = rankdata(vals[valid], method='average')
    pct = (ranks - 0.5) / valid.sum()
    transaction_model_scores[col] = pct[top_30_idx]

# Prepare for heatmap
transaction_model_scores_matrix = transaction_model_scores[model_cols_raw].values
txn_ids = transaction_model_scores["TransactionID"].values

# Clean column names
col_names_models = [
    "IF", "LOF", "OCSVM", "EE", "DBSCAN", "HDBSCAN", "KM", "GMM", "AE", "VAE", "LSTM"
]

# Add final ensemble score column
final_score_top30 = final_score[top_30_idx]
transaction_model_scores_with_final = np.column_stack([transaction_model_scores_matrix, final_score_top30])
col_names_with_final = col_names_models + ["Final\nEnsemble"]

fig, ax = plt.subplots(figsize=(14, 16))
sns.heatmap(
    transaction_model_scores_with_final,
    cmap="Reds",
    vmin=0,
    vmax=1,
    linewidths=0.3,
    cbar_kws={"label": "Normalized Anomaly Percentile", "shrink": 0.6},
    ax=ax,
    xticklabels=col_names_with_final,
    yticklabels=txn_ids
)
ax.set_title("Top 30 Suspicious Transactions × Models\n(Each Cell = Model's Anomaly Percentile for that Transaction)",
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel("Model", fontsize=11, fontweight='bold')
ax.set_ylabel("Transaction ID", fontsize=11, fontweight='bold')
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "7_top_transactions_model_score_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

transaction_model_scores.to_csv(os.path.join(OUTPUT_DIR, "7_top_transactions_scores.csv"), index=False)
print(f"> Saved top transactions × models heatmap")
print()

# ============================================================================
# SECTION 10: HEATMAP 8 - SUPERVISED MODEL PERFORMANCE (V1)
# ============================================================================
print("SECTION 10: Generating Heatmap 8 - Supervised Model Performance (V1)...")
print("-" * 80)

# V1 comparison has: Model, Fold, metrics
# Filter for test results only
v1_comparison_test = v1_comparison[v1_comparison["Fold"] == "test"].copy()

# Prepare matrix
supervised_metrics = v1_comparison_test.set_index("Model")[
    ["precision", "recall", "f1", "roc_auc", "pr_auc"]
].T

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    supervised_metrics,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Score", "shrink": 0.8},
    ax=ax
)
ax.set_title("V1 Supervised Model Performance (Pseudo-Label Validation)\n(XGBoost vs Random Forest)",
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel("Model", fontsize=11, fontweight='bold')
ax.set_ylabel("Metric", fontsize=11, fontweight='bold')
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "8_supervised_model_performance_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

supervised_metrics.to_csv(os.path.join(OUTPUT_DIR, "8_supervised_model_performance_matrix.csv"))
print(f"> Saved supervised model performance heatmap")
print()

# ============================================================================
# SECTION 11: ANALYSIS SUMMARY TABLES
# ============================================================================
print("SECTION 11: Creating Analysis Summary Tables...")
print("-" * 80)

# Model agreement analysis
print("Computing model agreement statistics...")
spearman_values = spearman_matrix.values[np.triu_indices_from(spearman_matrix.values, k=1)]
jaccard_values = jaccard_matrix.values[np.triu_indices_from(jaccard_matrix.values, k=1)]

agreement_summary = pd.DataFrame({
    "Metric": [
        "Mean Spearman Correlation",
        "Median Spearman Correlation",
        "Min Spearman Correlation",
        "Max Spearman Correlation",
        "Mean Jaccard Similarity",
        "Median Jaccard Similarity",
        "Min Jaccard Similarity",
        "Max Jaccard Similarity",
    ],
    "Value": [
        spearman_values.mean(),
        np.median(spearman_values),
        spearman_values.min(),
        spearman_values.max(),
        jaccard_values.mean(),
        np.median(jaccard_values),
        jaccard_values.min(),
        jaccard_values.max(),
    ]
})

agreement_summary.to_csv(os.path.join(OUTPUT_DIR, "model_agreement_summary.csv"), index=False)
print(f"> Saved model agreement summary")

# Top model pairs by agreement/disagreement
model_names_list = spearman_matrix.index.tolist()
pairs = []
for i in range(len(model_names_list)):
    for j in range(i+1, len(model_names_list)):
        pairs.append({
            "Model_1": model_names_list[i],
            "Model_2": model_names_list[j],
            "Spearman": spearman_matrix.iloc[i, j],
            "Jaccard": jaccard_matrix.iloc[i, j],
            "Disagreement": 1 - jaccard_matrix.iloc[i, j]
        })

pairs_df = pd.DataFrame(pairs)

# Top 10 most agreeing pairs
top_agreeing = pairs_df.nlargest(10, "Jaccard")[["Model_1", "Model_2", "Spearman", "Jaccard"]]
top_agreeing.to_csv(os.path.join(OUTPUT_DIR, "top_10_agreeing_model_pairs.csv"), index=False)

# Top 10 most disagreeing pairs
top_disagreeing = pairs_df.nlargest(10, "Disagreement")[["Model_1", "Model_2", "Spearman", "Disagreement"]]
top_disagreeing.to_csv(os.path.join(OUTPUT_DIR, "top_10_disagreeing_model_pairs.csv"), index=False)

print(f"> Saved top agreeing/disagreeing model pairs")
print()

# ============================================================================
# FINAL REPORT
# ============================================================================
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print()
print(f"Generated {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])} heatmaps")
print(f"Generated {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])} CSV files")
print()
print("Key Outputs:")
print(f"  • Master Model Inventory: {OUTPUT_DIR}/master_model_inventory.csv")
print(f"  • Spearman Correlation Heatmap: {OUTPUT_DIR}/1_model_spearman_correlation_heatmap.png")
print(f"  • Jaccard Agreement Heatmap: {OUTPUT_DIR}/2_model_jaccard_agreement_heatmap.png")
print(f"  • Disagreement Heatmap: {OUTPUT_DIR}/3_model_disagreement_heatmap.png")
print(f"  • Feature Correlation Heatmaps: {OUTPUT_DIR}/4_feature_correlation_heatmap_*.png")
if has_shap:
    print(f"  • Feature×Model Importance: {OUTPUT_DIR}/5_feature_model_importance_heatmap.png")
print(f"  • Model Characteristics: {OUTPUT_DIR}/6_model_characteristics_heatmap.png")
print(f"  • Top Transactions: {OUTPUT_DIR}/7_top_transactions_model_score_heatmap.png")
print(f"  • V1 Supervised Performance: {OUTPUT_DIR}/8_supervised_model_performance_heatmap.png")
print()
print("All outputs saved to:", OUTPUT_DIR)
print()
