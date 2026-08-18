"""
Phase 8 (v2), Part 1 -- Models 1-8 (classical unsupervised anomaly detectors)
on the teammate's 18-feature matrix (artifacts_research/features_teammate_merged.csv).

Mirrors src_research/07_models_classical.py's methodology 1:1:
  - RobustScaler, fit on the train split only, applied to the full dataset
    (all 18 columns; even though the teammate's columns are already
    StandardScaler-scaled, RobustScaler is applied on top for consistency
    with the in-house Phase 6 recommendation and so distance-based models
    are not overly sensitive to the handful of extreme rows RobustScaler is
    specifically chosen to be robust to).
  - Same 80/20 train/val split (random_state=42), reproduced from the exact
    same train_test_split(np.arange(len(df)), ...) call used to train the
    Phase 7 autoencoder -- cross-checked below, not assumed.
  - Same anomaly-score sign convention: every score_<model> column is
    oriented so HIGHER = MORE ANOMALOUS (sklearn decision_function outputs
    are negated for IF/LOF/OCSVM/EE).
  - Two anomaly-rate conventions used side by side: each model's native
    rate, and a standardized top-5%-by-score flag for cross-model Jaccard
    comparability (Section 3 in the writeup).
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import (
    ARTIFACTS_V2_DIR, FEATURE_COLS_V2, MODELS_V2_DIR, PLOTS_V2_DIR, RANDOM_STATE, load_features_v2,
)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

TOP_PCT = 0.05


def savefig(fig, name):
    path = os.path.join(PLOTS_V2_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def load_and_split():
    df = load_features_v2()
    leakage_cols = [c for c in ("vote_count", "risk_tier", "is_fraud") if c in df.columns]
    assert not leakage_cols, f"Leakage-risk columns found: {leakage_cols}"

    feature_cols = FEATURE_COLS_V2
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)

    scaler = RobustScaler().fit(X[idx_train])
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    X_all = scaler.transform(X)

    joblib.dump(scaler, os.path.join(MODELS_V2_DIR, "shared_robust_scaler.pkl"))

    ae_errs = pd.read_csv(os.path.join(ARTIFACTS_V2_DIR, "autoencoder_reconstruction_errors.csv"))
    ae_train_mask = (ae_errs["split"] == "train").values
    my_train_mask = np.isin(np.arange(len(df)), idx_train)
    match = (ae_train_mask == my_train_mask).all()
    print(f"Split reproduction check vs autoencoder_reconstruction_errors.csv: {'MATCH' if match else 'MISMATCH'}")

    return df, feature_cols, X, X_train, X_val, X_all, idx_train, idx_val, scaler


def top_pct_flag(score, pct=TOP_PCT):
    thresh = np.percentile(score, 100 * (1 - pct))
    return (score >= thresh).astype(int)


# --------------------------------------------------------------- Model 1: IF
def model_isolation_forest(X_train, X_all):
    configs = [
        dict(n_estimators=100, max_samples="auto", contamination=0.05, max_features=1.0),
        dict(n_estimators=200, max_samples=0.8, contamination=0.05, max_features=1.0),
        dict(n_estimators=300, max_samples=0.5, contamination=0.05, max_features=0.7),
        dict(n_estimators=200, max_samples="auto", contamination=0.10, max_features=1.0),
        dict(n_estimators=200, max_samples="auto", contamination=0.03, max_features=1.0),
    ]
    rows, fitted = [], []
    t0 = time.time()
    for cfg in configs:
        clf = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
        clf.fit(X_train)
        score = -clf.decision_function(X_all)
        native_flag = (clf.predict(X_all) == -1).astype(int)
        rows.append({**cfg, "native_anomaly_rate": native_flag.mean(), "top5pct_score_std": score.std()})
        fitted.append((clf, score, native_flag))
    fit_time = time.time() - t0

    cand_idx = [i for i, c in enumerate(configs) if c["contamination"] == 0.05]
    best_i = max(cand_idx, key=lambda i: rows[i]["top5pct_score_std"])
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_V2_DIR, "isolation_forest.pkl"))
    summary = {
        "configs_tried": rows, "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": ("O(n log n) per tree; fit+score of 5 configs x 2,512 rows x 18 features took "
                      f"{fit_time:.2f}s total on CPU."),
    }
    print("\n=== Model 1: Isolation Forest ===")
    print(json.dumps(summary, indent=2, default=float))
    return best_score, best_native_flag, summary


# -------------------------------------------------------------- Model 2: LOF
def model_lof(X_train, X_all):
    configs = [
        dict(n_neighbors=10, contamination=0.05),
        dict(n_neighbors=20, contamination=0.05),
        dict(n_neighbors=35, contamination=0.05),
        dict(n_neighbors=20, contamination=0.10),
        dict(n_neighbors=20, contamination=0.03),
    ]
    rows, fitted = [], []
    t0 = time.time()
    for cfg in configs:
        clf = LocalOutlierFactor(novelty=True, **cfg)
        clf.fit(X_train)
        score = -clf.decision_function(X_all)
        native_flag = (clf.predict(X_all) == -1).astype(int)
        rows.append({**cfg, "native_anomaly_rate": native_flag.mean()})
        fitted.append((clf, score, native_flag))
    fit_time = time.time() - t0

    cand_idx = [i for i, c in enumerate(configs) if c["contamination"] == 0.05]
    best_i = cand_idx[1]
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_V2_DIR, "lof.pkl"))
    summary = {
        "configs_tried": rows, "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": "O(n^2) neighbor search by default; novelty=True for out-of-sample scoring.",
    }
    print("\n=== Model 2: Local Outlier Factor ===")
    print(json.dumps(summary, indent=2, default=float))
    return best_score, best_native_flag, summary


# ------------------------------------------------------------ Model 3: OCSVM
def model_ocsvm(X_train, X_all):
    configs = [
        dict(kernel="rbf", nu=0.05, gamma="scale"),
        dict(kernel="rbf", nu=0.05, gamma="auto"),
        dict(kernel="rbf", nu=0.10, gamma="scale"),
        dict(kernel="linear", nu=0.05, gamma="scale"),
        dict(kernel="poly", nu=0.05, gamma="scale"),
    ]
    rows, fitted = [], []
    t0 = time.time()
    for cfg in configs:
        clf = OneClassSVM(**cfg)
        clf.fit(X_train)
        score = -clf.decision_function(X_all)
        native_flag = (clf.predict(X_all) == -1).astype(int)
        rows.append({**cfg, "native_anomaly_rate": native_flag.mean()})
        fitted.append((clf, score, native_flag))
    fit_time = time.time() - t0

    best_i = 0
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_V2_DIR, "ocsvm.pkl"))
    summary = {
        "configs_tried": rows, "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": f"QP solve, roughly O(n^2)-O(n^3); {fit_time:.2f}s for 5 configs at n=2,512.",
    }
    print("\n=== Model 3: One-Class SVM ===")
    print(json.dumps(summary, indent=2, default=float))
    return best_score, best_native_flag, summary


# ------------------------------------------------------- Model 4: Elliptic Envelope
def model_elliptic_envelope(X_train, X_all, feature_cols):
    shapiro_p = []
    for j, col in enumerate(feature_cols):
        col_vals = X_train[:, j]
        if np.std(col_vals) < 1e-10:
            shapiro_p.append(np.nan)
            continue
        _, p = stats.shapiro(col_vals[:min(len(col_vals), 5000)])
        shapiro_p.append(p)
    shapiro_p = np.array(shapiro_p)
    frac_non_normal = float(np.nanmean(shapiro_p < 0.05))

    configs = [
        dict(contamination=0.05, support_fraction=None),
        dict(contamination=0.05, support_fraction=0.8),
        dict(contamination=0.10, support_fraction=None),
    ]
    rows, fitted = [], []
    t0 = time.time()
    for cfg in configs:
        clf = EllipticEnvelope(random_state=RANDOM_STATE, **cfg)
        clf.fit(X_train)
        score = -clf.decision_function(X_all)
        native_flag = (clf.predict(X_all) == -1).astype(int)
        rows.append({**cfg, "native_anomaly_rate": native_flag.mean()})
        fitted.append((clf, score, native_flag))
    fit_time = time.time() - t0

    best_i = 0
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_V2_DIR, "elliptic_envelope.pkl"))
    summary = {
        "configs_tried": rows, "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "shapiro_wilk_frac_features_non_normal_p_lt_0.05": round(frac_non_normal, 4),
        "per_feature_shapiro_p": {feature_cols[j]: (None if np.isnan(shapiro_p[j]) else round(float(shapiro_p[j]), 6))
                                    for j in range(len(feature_cols))},
        "fit_time_sec_all_configs": round(fit_time, 3),
        "gaussian_assumption_note": (
            f"{frac_non_normal*100:.1f}% of the 18 scaled features reject the Shapiro-Wilk normality "
            "null at p<0.05. EllipticEnvelope assumes the joint feature distribution is multivariate "
            "Gaussian; this level of univariate non-normality means its MCD-based Mahalanobis distance "
            "should be read as a rough baseline, not a well-calibrated score."
        ),
    }
    print("\n=== Model 4: Elliptic Envelope ===")
    print(json.dumps(summary, indent=2, default=float))
    return best_score, best_native_flag, summary


# ------------------------------------------------------------- Model 5: DBSCAN
def model_dbscan(X_all):
    min_samples_candidates = [5, 10, 15]
    k = min_samples_candidates[1] - 1
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    dist, _ = nn.kneighbors(X_all)
    kdist = np.sort(dist[:, -1])

    x_norm = np.linspace(0, 1, len(kdist))
    y_norm = (kdist - kdist.min()) / (kdist.max() - kdist.min() + 1e-12)
    p1, p2 = np.array([x_norm[0], y_norm[0]]), np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    pts = np.stack([x_norm, y_norm], axis=1)
    vec_from_p1 = pts - p1
    proj_len = vec_from_p1 @ line_vec_norm
    proj_pt = np.outer(proj_len, line_vec_norm) + p1
    perp_dist = np.linalg.norm(pts - proj_pt, axis=1)
    elbow_idx = int(np.argmax(perp_dist))
    eps_elbow = float(kdist[elbow_idx])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kdist, color="#2F6690", lw=1.5)
    ax.axvline(elbow_idx, color="#D1495B", ls="--", lw=1.2, label=f"Elbow at rank {elbow_idx} -> eps={eps_elbow:.3f}")
    ax.axhline(eps_elbow, color="#D1495B", ls=":", lw=0.8)
    ax.set_xlabel(f"Points sorted by distance to their {k}-th nearest neighbor")
    ax.set_ylabel(f"{k}-NN distance (RobustScaler-scaled space)")
    ax.set_title("DBSCAN -- k-Distance Elbow Plot (v2, 18 features, min_samples=10)")
    ax.legend()
    savefig(fig, "dbscan_kdistance_elbow_v2.png")

    eps_candidates = sorted(set([round(eps_elbow, 3), round(eps_elbow * 0.8, 3), round(eps_elbow * 1.2, 3)]))
    rows, fitted = [], []
    t0 = time.time()
    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            clf = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = clf.fit_predict(X_all)
            noise_rate = float((labels == -1).mean())
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
            rows.append({"eps": eps, "min_samples": min_samples, "noise_rate": noise_rate, "n_clusters": n_clusters})
            fitted.append((labels, noise_rate, n_clusters))
    fit_time = time.time() - t0

    best_i = next(i for i, r in enumerate(rows) if r["eps"] == round(eps_elbow, 3) and r["min_samples"] == 10)
    best_labels, best_noise_rate, best_n_clusters = fitted[best_i]

    clf_final = DBSCAN(eps=eps_candidates[eps_candidates.index(round(eps_elbow, 3))], min_samples=10, n_jobs=-1)
    labels_final = clf_final.fit_predict(X_all)
    core_idx = clf_final.core_sample_indices_
    if len(core_idx) > 0:
        nn_core = NearestNeighbors(n_neighbors=1).fit(X_all[core_idx])
        d_to_core, _ = nn_core.kneighbors(X_all)
        score = d_to_core[:, 0]
    else:
        score = np.zeros(len(X_all))

    joblib.dump(clf_final, os.path.join(MODELS_V2_DIR, "dbscan.pkl"))
    summary = {
        "eps_from_kdistance_elbow": round(eps_elbow, 4),
        "configs_tried": rows,
        "selected_config": {"eps": round(eps_elbow, 3), "min_samples": 10},
        "selected_noise_rate": round(best_noise_rate, 4),
        "selected_n_clusters": best_n_clusters,
        "fit_time_sec_all_configs": round(fit_time, 3),
        "note": (f"Noise rate ranges {min(r['noise_rate'] for r in rows)*100:.1f}%-"
                 f"{max(r['noise_rate'] for r in rows)*100:.1f}% across the 3x3 eps/min_samples grid; "
                 f"{max(r['n_clusters'] for r in rows)} clusters max across the grid."),
    }
    print("\n=== Model 5: DBSCAN ===")
    print(json.dumps(summary, indent=2, default=float))
    return score, (labels_final == -1).astype(int), summary


# ------------------------------------------------------------ Model 6: HDBSCAN
def model_hdbscan(X_all):
    import hdbscan as hdbscan_lib

    configs = [
        dict(min_cluster_size=10, min_samples=5),
        dict(min_cluster_size=20, min_samples=10),
        dict(min_cluster_size=30, min_samples=15),
        dict(min_cluster_size=15, min_samples=None),
    ]
    rows, fitted = [], []
    t0 = time.time()
    for cfg in configs:
        clf = hdbscan_lib.HDBSCAN(**cfg)
        labels = clf.fit_predict(X_all)
        noise_rate = float((labels == -1).mean())
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        rows.append({**cfg, "noise_rate": noise_rate, "n_clusters": n_clusters})
        fitted.append((clf, labels, noise_rate, n_clusters))
    fit_time = time.time() - t0

    best_i = min(range(len(rows)), key=lambda i: abs(rows[i]["noise_rate"] - 0.05))
    best_clf, best_labels, best_noise_rate, best_n_clusters = fitted[best_i]
    outlier_scores = best_clf.outlier_scores_

    joblib.dump(best_clf, os.path.join(MODELS_V2_DIR, "hdbscan.pkl"))
    summary = {
        "configs_tried": rows, "selected_config": configs[best_i],
        "selected_noise_rate": round(best_noise_rate, 4),
        "selected_n_clusters": best_n_clusters,
        "fit_time_sec_all_configs": round(fit_time, 3),
        "note": (f"Noise rate ranges {min(r['noise_rate'] for r in rows)*100:.1f}%-"
                 f"{max(r['noise_rate'] for r in rows)*100:.1f}% across the 4 configs tried."),
    }
    print("\n=== Model 6: HDBSCAN ===")
    print(json.dumps(summary, indent=2, default=float))
    return outlier_scores, (best_labels == -1).astype(int), summary


# ------------------------------------------------------------- Model 7: KMeans
def model_kmeans(X_train, X_all):
    k_range = list(range(2, 11))
    inertias, silhouettes = [], []
    from sklearn.metrics import silhouette_score
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_train)
        inertias.append(km.inertia_)
        rng = np.random.RandomState(RANDOM_STATE)
        sub_idx = rng.choice(len(X_train), size=min(1000, len(X_train)), replace=False)
        sil = silhouette_score(X_train[sub_idx], km.labels_[sub_idx])
        silhouettes.append(sil)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(k_range, inertias, marker="o", color="#2F6690")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("K-Means Elbow (Inertia), v2")
    axes[1].plot(k_range, silhouettes, marker="o", color="#D1495B")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette (1,000-row subsample)")
    axes[1].set_title("K-Means Silhouette by k, v2")
    savefig(fig, "kmeans_elbow_silhouette_v2.png")

    degenerate_k = {}
    for i, k in enumerate(k_range):
        km_probe = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_train)
        sizes = np.bincount(km_probe.labels_)
        if sizes.min() < 0.01 * len(X_train):
            degenerate_k[k] = int(sizes.min())

    best_k_silhouette = k_range[int(np.argmax(silhouettes))]

    # inertia-elbow pick: first k where the marginal inertia drop falls below
    # 15% of the k=2->k=3 drop (a simple, reproducible elbow rule)
    diffs = -np.diff(inertias)
    threshold = 0.15 * diffs[0]
    elbow_pos = next((i for i, d in enumerate(diffs) if d < threshold), len(diffs) - 1)
    best_k = k_range[elbow_pos + 1]

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_train)
    sizes_final = np.bincount(km_final.labels_)
    valid_clusters = np.where(sizes_final >= 0.01 * len(X_train))[0]
    micro_clusters = np.where(sizes_final < 0.01 * len(X_train))[0]
    valid_centroids = km_final.cluster_centers_[valid_clusters] if len(valid_clusters) > 0 else km_final.cluster_centers_

    dists_to_valid = np.linalg.norm(X_all[:, None, :] - valid_centroids[None, :, :], axis=2)
    score_all = dists_to_valid.min(axis=1)
    flag = top_pct_flag(score_all)

    joblib.dump(km_final, os.path.join(MODELS_V2_DIR, "kmeans.pkl"))
    summary = {
        "k_range_tried": k_range,
        "inertia_by_k": [round(v, 2) for v in inertias],
        "silhouette_by_k": [round(v, 4) for v in silhouettes],
        "silhouette_argmax_k": int(best_k_silhouette),
        "degenerate_k_micro_cluster_sizes": degenerate_k,
        "selected_k": int(best_k),
        "selected_k_rationale": "Chosen from the inertia elbow (first k whose marginal inertia drop falls below 15% of the k=2->3 drop); micro-clusters (<1% of train rows) excluded from centroid distance targets.",
        "n_micro_clusters_at_selected_k": int(len(micro_clusters)),
        "top5pct_flagged_rate": round(float(flag.mean()), 4),
    }
    print("\n=== Model 7: K-Means (distance-based) ===")
    print(json.dumps(summary, indent=2, default=float))
    return score_all, flag, summary


# ---------------------------------------------------------------- Model 8: GMM
def model_gmm(X_train, X_all):
    n_range = list(range(1, 11))
    cov_types = ["full", "diag", "tied", "spherical"]
    bic_by_cov = {c: [] for c in cov_types}
    aic_by_cov = {c: [] for c in cov_types}
    for cov in cov_types:
        for n in n_range:
            gm = GaussianMixture(n_components=n, covariance_type=cov, random_state=RANDOM_STATE,
                                  reg_covar=1e-5, max_iter=200).fit(X_train)
            bic_by_cov[cov].append(gm.bic(X_train))
            aic_by_cov[cov].append(gm.aic(X_train))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"full": "#2F6690", "diag": "#D1495B", "tied": "#4C956C", "spherical": "#EDAE49"}
    for cov in cov_types:
        axes[0].plot(n_range, bic_by_cov[cov], marker="o", label=cov, color=colors[cov])
        axes[1].plot(n_range, aic_by_cov[cov], marker="o", label=cov, color=colors[cov])
    axes[0].set_xlabel("n_components"); axes[0].set_ylabel("BIC"); axes[0].set_title("GMM -- BIC by n_components (v2)")
    axes[1].set_xlabel("n_components"); axes[1].set_ylabel("AIC"); axes[1].set_title("GMM -- AIC by n_components (v2)")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    savefig(fig, "gmm_bic_aic_v2.png")

    best_bic, best_cov, best_n = np.inf, None, None
    for cov in cov_types:
        i = int(np.argmin(bic_by_cov[cov]))
        if bic_by_cov[cov][i] < best_bic:
            best_bic, best_cov, best_n = bic_by_cov[cov][i], cov, n_range[i]

    gm_final = GaussianMixture(n_components=best_n, covariance_type=best_cov,
                                random_state=RANDOM_STATE, reg_covar=1e-5, max_iter=200).fit(X_train)
    log_lik = gm_final.score_samples(X_all)
    score = -log_lik
    flag = top_pct_flag(score)

    joblib.dump(gm_final, os.path.join(MODELS_V2_DIR, "gmm.pkl"))
    summary = {
        "n_range_tried": n_range, "cov_types_tried": cov_types,
        "bic_by_cov": {c: [round(v, 1) for v in bic_by_cov[c]] for c in cov_types},
        "selected_n_components": int(best_n), "selected_covariance_type": best_cov,
        "selected_bic": round(float(best_bic), 1),
        "top5pct_flagged_rate": round(float(flag.mean()), 4),
    }
    print("\n=== Model 8: Gaussian Mixture Model ===")
    print(json.dumps(summary, indent=2, default=float))
    return score, flag, summary


def main():
    df, feature_cols, X, X_train, X_val, X_all, idx_train, idx_val, scaler = load_and_split()

    score_if, flag_if, sum_if = model_isolation_forest(X_train, X_all)
    score_lof, flag_lof, sum_lof = model_lof(X_train, X_all)
    score_svm, flag_svm, sum_svm = model_ocsvm(X_train, X_all)
    score_ee, flag_ee, sum_ee = model_elliptic_envelope(X_train, X_all, feature_cols)
    score_db, flag_db, sum_db = model_dbscan(X_all)
    score_hdb, flag_hdb, sum_hdb = model_hdbscan(X_all)
    score_km, flag_km, sum_km = model_kmeans(X_train, X_all)
    score_gmm, flag_gmm, sum_gmm = model_gmm(X_train, X_all)

    out = pd.DataFrame({
        "TransactionID": df["TransactionID"].values,
        "AccountID": df["AccountID"].values,
        "split": np.where(np.isin(np.arange(len(df)), idx_train), "train", "val"),
        "score_isolation_forest": score_if, "flag_isolation_forest_native": flag_if,
        "score_lof": score_lof, "flag_lof_native": flag_lof,
        "score_ocsvm": score_svm, "flag_ocsvm_native": flag_svm,
        "score_elliptic_envelope": score_ee, "flag_elliptic_envelope_native": flag_ee,
        "score_dbscan": score_db, "flag_dbscan_native": flag_db,
        "score_hdbscan": score_hdb, "flag_hdbscan_native": flag_hdb,
        "score_kmeans": score_km, "flag_kmeans_top5pct": flag_km,
        "score_gmm": score_gmm, "flag_gmm_top5pct": flag_gmm,
    })
    out.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_scores_classical.csv"), index=False)
    print(f"\nSaved: {os.path.join(ARTIFACTS_V2_DIR, 'model_scores_classical.csv')}")

    all_summary = {
        "isolation_forest": sum_if, "lof": sum_lof, "ocsvm": sum_svm,
        "elliptic_envelope": sum_ee, "dbscan": sum_db, "hdbscan": sum_hdb,
        "kmeans": sum_km, "gmm": sum_gmm,
    }
    with open(os.path.join(ARTIFACTS_V2_DIR, "model_summary_classical.json"), "w") as f:
        json.dump(all_summary, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_V2_DIR, 'model_summary_classical.json')}")


if __name__ == "__main__":
    main()
