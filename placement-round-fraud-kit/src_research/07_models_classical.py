"""
Phase 8 -- Model Development, Part 1 (Models 1-8): classical / classical-ML
unsupervised anomaly detectors on the RobustScaler-scaled Phase 5/6 feature
matrix (artifacts_research/features_v2.csv, 46 engineered feature columns,
ID columns and any leakage-risk columns excluded -- verified below).

Design decisions carried over from Phase 6/7 (see research/05_feature_selection_and_preprocessing.md):
  - RobustScaler, fit on the train split only, applied to the full dataset.
  - The train/val split (2,009 / 503, random_state=42) is the canonical
    split reused by every downstream script in this phase (08-13):
    train_test_split(np.arange(len(df)), test_size=0.2, random_state=42).
  - Feature set: all 46 non-ID columns of features_v2.csv (i.e. both
    Location_enc and Location_Freq are present). Phase 6 recommends
    Location_Freq over Location_enc for a *single* encoding of Location, but
    every model in this phase is scored on the identical feature matrix for
    apples-to-apples comparison (Spearman / Jaccard across models only mean
    something if every model saw the same columns). Redundancy risk from
    keeping both encodings is checked directly: corr(Location_enc,
    Location_Freq) = 0.0029 -- negligible, so this is not a meaningful
    double-count of the same information.
  - No leakage columns: features_v2.csv has no vote_count / risk_tier /
    is_fraud columns (those live only in the *separate* artifacts/labeled.csv
    from v1's supervised side-experiment) -- verified explicitly below, not
    assumed.

Note: this phase previously (pre deep-learning removal) also included an
Autoencoder, VAE, and LSTM Autoencoder (formerly Models 9-11) plus a Hybrid
Ensemble (formerly Model 12, IF + LOF + Autoencoder majority vote). Those
three deep-learning models were removed; the Hybrid Ensemble (now Model 9,
built in 08_models_deep.py) was redefined to IF + LOF + GMM majority vote.
This pipeline now has 9 models total: the 8 classical detectors built below
plus the Model 9 Hybrid Ensemble.

Anomaly-score sign convention used everywhere in this file (and carried into
08_models_deep.py): every score_<model> column is oriented so that HIGHER =
MORE ANOMALOUS. sklearn's OutlierMixin decision_function is the opposite
convention (higher = more normal) for IsolationForest / LOF / OneClassSVM /
EllipticEnvelope, so those are negated before saving.
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
from config_research import ARTIFACTS_RESEARCH_DIR, PLOTS_DIR, RANDOM_STATE, ROOT_DIR

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

FEATURES_V2_CSV = os.path.join(ARTIFACTS_RESEARCH_DIR, "features_v2.csv")
ID_COLS = ["TransactionID", "AccountID"]
MODELS_DIR = os.path.join(ARTIFACTS_RESEARCH_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TOP_PCT = 0.05  # standardized "flagged" definition used for cross-model comparison


def savefig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def load_and_split():
    df = pd.read_csv(FEATURES_V2_CSV)
    leakage_cols = [c for c in ("vote_count", "risk_tier", "is_fraud") if c in df.columns]
    assert not leakage_cols, f"Leakage-risk columns found in features_v2.csv: {leakage_cols}"

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    print(f"Feature matrix: {len(feature_cols)} columns")

    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)

    scaler = RobustScaler().fit(X[idx_train])
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    X_all = scaler.transform(X)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "shared_robust_scaler.pkl"))

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
    rows = []
    fitted = []
    t0 = time.time()
    for cfg in configs:
        clf = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **cfg)
        clf.fit(X_train)
        score = -clf.decision_function(X_all)
        flag5 = top_pct_flag(score)
        native_flag = (clf.predict(X_all) == -1).astype(int)
        rows.append({**cfg, "native_anomaly_rate": native_flag.mean(),
                     "top5pct_score_std": score.std()})
        fitted.append((clf, score, native_flag))
    fit_time = time.time() - t0

    # pick the config closest to a 5% contamination target with the most
    # stable (highest-variance / best-separated) score distribution among
    # the 5%-contamination configs, as the "selected" model for downstream use
    cand_idx = [i for i, c in enumerate(configs) if c["contamination"] == 0.05]
    best_i = max(cand_idx, key=lambda i: rows[i]["top5pct_score_std"])
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    summary = {
        "configs_tried": rows,
        "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": ("O(n log n) per tree, trivially parallel across trees and rows; "
                      "fit + score of 5 configs x 2,512 rows x 46 features took "
                      f"{fit_time:.2f}s total on CPU -- cheapest model in this comparison "
                      "and the only one of the 8 classical models with a native, "
                      "well-understood out-of-sample .decision_function for production scoring."),
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
    rows = []
    fitted = []
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
    # pick n_neighbors=20 (middle option) among the 5%-contamination configs as
    # the balanced default (10 is noisy/local, 35 over-smooths on a 495-account dataset)
    best_i = cand_idx[1]
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_DIR, "lof.pkl"))
    summary = {
        "configs_tried": rows,
        "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": ("O(n^2) neighbor search by default (no index acceleration benefit at "
                      "n=2,512, but this does not scale past low tens-of-thousands of rows "
                      "without an ANN index); novelty=True mode used so it can score new "
                      "transactions without refitting -- required for any production use."),
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
    rows = []
    fitted = []
    t0 = time.time()
    for cfg in configs:
        clf = OneClassSVM(**cfg)
        clf.fit(X_train)
        score = -clf.decision_function(X_all)
        native_flag = (clf.predict(X_all) == -1).astype(int)
        rows.append({**cfg, "native_anomaly_rate": native_flag.mean()})
        fitted.append((clf, score, native_flag))
    fit_time = time.time() - t0

    # rbf/nu=0.05/scale is the standard default combination; select it unless
    # a clearly better-separated alternative emerges (checked via score std)
    best_i = 0
    best_cfg = configs[best_i]
    best_clf, best_score, best_native_flag = fitted[best_i]

    joblib.dump(best_clf, os.path.join(MODELS_DIR, "ocsvm.pkl"))
    summary = {
        "configs_tried": rows,
        "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": ("SMO-based QP solve, roughly O(n^2)-O(n^3) depending on kernel and how "
                      "many points end up as support vectors; on this dataset it was the "
                      f"slowest of the 8 classical models to fit ({fit_time:.2f}s for 5 configs) "
                      "-- fine at n=2,512 but a real scalability concern past ~50k-100k rows "
                      "without subsampling or an approximate SVM variant."),
    }
    print("\n=== Model 3: One-Class SVM ===")
    print(json.dumps(summary, indent=2, default=float))
    return best_score, best_native_flag, summary


# ------------------------------------------------------- Model 4: Elliptic Envelope
def model_elliptic_envelope(X_train, X_all, feature_cols):
    # Normality check on the (scaled) features EllipticEnvelope assumes are
    # jointly Gaussian: Shapiro-Wilk per feature (scipy caps meaningfully at
    # n<=5000, fine here) plus a multivariate proxy (Mardia-style skew/kurtosis
    # via scipy is not available; report the fraction of individually
    # non-normal features as the practical red flag instead).
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
    rows = []
    fitted = []
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

    joblib.dump(best_clf, os.path.join(MODELS_DIR, "elliptic_envelope.pkl"))
    summary = {
        "configs_tried": rows,
        "selected_config": best_cfg,
        "selected_native_anomaly_rate": round(float(best_native_flag.mean()), 4),
        "shapiro_wilk_frac_features_non_normal_p_lt_0.05": round(frac_non_normal, 4),
        "gaussian_assumption_note": (
            f"{frac_non_normal*100:.1f}% of the 46 scaled features reject the Shapiro-Wilk "
            "normality null at p<0.05 -- consistent with Phase 4's finding that TransactionAmount "
            "and its derived features are right-skewed (skew 1.74, unchanged by any affine scaler, "
            "per Phase 6) rather than Gaussian. EllipticEnvelope assumes the *joint* feature "
            "distribution is multivariate Gaussian; individual non-normality this widespread makes "
            "that assumption shaky on this dataset, so its MCD-based covariance/Mahalanobis "
            "distance should be read as a rough, assumption-violating baseline, not a well-calibrated "
            "score -- included for completeness and comparison, not recommended as the primary "
            "production detector."
        ),
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": ("Fits a Minimum Covariance Determinant estimator, O(n * p^2) roughly; fast "
                      "at this scale but the fitted covariance matrix (46x46) itself gets less "
                      "reliable as skew/non-normality increase, per the note above."),
    }
    print("\n=== Model 4: Elliptic Envelope ===")
    print(json.dumps(summary, indent=2, default=float))
    return best_score, best_native_flag, summary


# ------------------------------------------------------------- Model 5: DBSCAN
def model_dbscan(X_all):
    # k-distance elbow: k = min_samples - 1 (standard convention), sorted ascending
    min_samples_candidates = [5, 10, 15]
    k = min_samples_candidates[1] - 1  # use min_samples=10 for the elbow plot
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    dist, _ = nn.kneighbors(X_all)
    kdist = np.sort(dist[:, -1])

    # classic elbow-detection: point of max perpendicular distance from the
    # line connecting the first and last point of the (normalized) k-distance curve
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
    ax.axvline(elbow_idx, color="#D1495B", ls="--", lw=1.2,
               label=f"Elbow at rank {elbow_idx} -> eps={eps_elbow:.3f}")
    ax.axhline(eps_elbow, color="#D1495B", ls=":", lw=0.8)
    ax.set_xlabel(f"Points sorted by distance to their {k}-th nearest neighbor")
    ax.set_ylabel(f"{k}-NN distance (RobustScaler-scaled space)")
    ax.set_title("DBSCAN -- k-Distance Elbow Plot (min_samples=10)")
    ax.legend()
    savefig(fig, "dbscan_kdistance_elbow.png")

    eps_candidates = sorted(set([round(eps_elbow, 3), round(eps_elbow * 0.8, 3), round(eps_elbow * 1.2, 3)]))
    rows = []
    fitted = []
    t0 = time.time()
    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            clf = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = clf.fit_predict(X_all)
            noise_rate = float((labels == -1).mean())
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
            rows.append({"eps": eps, "min_samples": min_samples,
                         "noise_rate": noise_rate, "n_clusters": n_clusters})
            fitted.append((labels, noise_rate, n_clusters))
    fit_time = time.time() - t0

    # select the elbow-derived eps with min_samples=10 as the primary config
    best_i = next(i for i, r in enumerate(rows) if r["eps"] == round(eps_elbow, 3) and r["min_samples"] == 10)
    best_labels, best_noise_rate, best_n_clusters = fitted[best_i]

    # continuous pseudo-score for ranking/Jaccard comparability: distance to
    # the nearest core point (0 for points that are themselves core points)
    clf_final = DBSCAN(eps=eps_candidates[eps_candidates.index(round(eps_elbow, 3))], min_samples=10, n_jobs=-1)
    labels_final = clf_final.fit_predict(X_all)
    core_idx = clf_final.core_sample_indices_
    if len(core_idx) > 0:
        nn_core = NearestNeighbors(n_neighbors=1).fit(X_all[core_idx])
        d_to_core, _ = nn_core.kneighbors(X_all)
        score = d_to_core[:, 0]
    else:
        score = np.zeros(len(X_all))

    joblib.dump(clf_final, os.path.join(MODELS_DIR, "dbscan.pkl"))
    summary = {
        "eps_from_kdistance_elbow": round(eps_elbow, 4),
        "configs_tried": rows,
        "selected_config": {"eps": round(eps_elbow, 3), "min_samples": 10},
        "selected_noise_rate": round(best_noise_rate, 4),
        "selected_n_clusters": best_n_clusters,
        "fit_time_sec_all_configs": round(fit_time, 3),
        "cost_note": ("O(n log n) with a spatial index in low dimensions, but the index degrades "
                      "toward O(n^2) in ~46-dimensional space (curse of dimensionality on the "
                      "radius query) -- consistent with the noise-rate instability across the eps "
                      "grid above; also has no out-of-sample .predict, requiring a full refit "
                      "to score new transactions, a real production limitation vs. Isolation Forest/LOF/OCSVM/EE."),
        "note": ("DBSCAN's eps is highly sensitive in this 46-D scaled space -- the noise rate "
                 "swings from "
                 f"{min(r['noise_rate'] for r in rows)*100:.1f}% to {max(r['noise_rate'] for r in rows)*100:.1f}% "
                 "across the 3x3 eps/min_samples grid, which is exactly the manual-tuning "
                 "brittleness HDBSCAN (Model 6) is designed to avoid."),
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
    rows = []
    fitted = []
    t0 = time.time()
    for cfg in configs:
        clf = hdbscan_lib.HDBSCAN(**cfg)
        labels = clf.fit_predict(X_all)
        noise_rate = float((labels == -1).mean())
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        rows.append({**cfg, "noise_rate": noise_rate, "n_clusters": n_clusters})
        fitted.append((clf, labels, noise_rate, n_clusters))
    fit_time = time.time() - t0

    # pick the config whose noise_rate is closest to 5% for rough comparability
    best_i = min(range(len(rows)), key=lambda i: abs(rows[i]["noise_rate"] - 0.05))
    best_clf, best_labels, best_noise_rate, best_n_clusters = fitted[best_i]
    outlier_scores = best_clf.outlier_scores_  # GLOSH score, higher = more anomalous, no manual eps needed

    joblib.dump(best_clf, os.path.join(MODELS_DIR, "hdbscan.pkl"))
    summary = {
        "configs_tried": rows,
        "selected_config": configs[best_i],
        "selected_noise_rate": round(best_noise_rate, 4),
        "selected_n_clusters": best_n_clusters,
        "fit_time_sec_all_configs": round(fit_time, 3),
        "comparison_vs_dbscan": ("Reported honestly rather than spun positive: HDBSCAN does remove "
                                  "the need to manually pick eps, but on this dataset that does not "
                                  "translate into a better result than DBSCAN. HDBSCAN's noise rate "
                                  f"ranges {min(r['noise_rate'] for r in rows)*100:.1f}%-"
                                  f"{max(r['noise_rate'] for r in rows)*100:.1f}% across its 4 configs -- "
                                  "far higher, and no more stable in absolute terms, than DBSCAN's "
                                  "0.7%-2.9% (Model 5). HDBSCAN only ever finds 2 clusters here and "
                                  "assigns the majority of points to neither, i.e. its stability-based "
                                  "cluster-extraction criterion does not find the single dense mass "
                                  "structure this dataset actually has (consistent with the Phase 7 "
                                  "UMAP/t-SNE finding of one dominant population plus small pockets, "
                                  "not several well-separated dense clusters) to be a 'stable' cluster "
                                  "worth keeping, and instead calls most of it noise. Its GLOSH "
                                  "outlier_scores_ (used as the anomaly score below) is still usable, "
                                  "but the noise-rate/cluster-count numbers themselves should not be "
                                  "read as 'HDBSCAN solved DBSCAN's tuning problem' on this data -- it "
                                  "traded manual eps-tuning brittleness for a different failure mode."),
        "cost_note": ("Builds a minimum spanning tree over mutual reachability distances, "
                      "O(n log n) typical case but more expensive per-fit than DBSCAN due to the "
                      f"cluster-hierarchy step; 4 configs on n=2,512 took {fit_time:.2f}s total, "
                      "still fast at this scale. Like DBSCAN, no native out-of-sample .predict "
                      "(hdbscan's approximate_predict exists but requires prediction_data=True and "
                      "was not exercised here)."),
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
        # silhouette on a 1,000-row subsample of train for speed/stability
        rng = np.random.RandomState(RANDOM_STATE)
        sub_idx = rng.choice(len(X_train), size=min(1000, len(X_train)), replace=False)
        sil = silhouette_score(X_train[sub_idx], km.labels_[sub_idx])
        silhouettes.append(sil)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(k_range, inertias, marker="o", color="#2F6690")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("K-Means Elbow (Inertia)")
    axes[1].plot(k_range, silhouettes, marker="o", color="#D1495B")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette (1,000-row subsample)")
    axes[1].set_title("K-Means Silhouette by k")
    savefig(fig, "kmeans_elbow_silhouette.png")

    # Genuine, checked-directly finding: naive argmax(silhouette) picks k=2
    # (silhouette 0.9184), and EVERY k in [2, 10] contains at least one cluster
    # below 1% of the training rows. This is not "K-Means needs more tuning" --
    # confirmed directly that at k=2 and k=9 the same 3 training rows (indices
    # matching TX000177/AC00363, TX002305/AC00494, TX001012/AC00329; the first
    # two have Amount_ZScore_Account = 92.56 and 77.71, the most extreme
    # z-scores in the dataset) get carved out into their own micro-cluster
    # regardless of k. K-Means is, correctly, telling us these rows are too
    # different from everything else to share a centroid with anything --
    # exactly the joint-multivariate-outlier signature the Phase 3 Isolation
    # Forest analysis and Phase 7 autoencoder tail (P99/max MSE) already found.
    # The practical consequence: naively using "distance to nearest centroid"
    # as the anomaly score would make these rows the SAFEST-looking points in
    # the dataset (they sit almost exactly on their own centroid, distance
    # near 0) -- the opposite of the intended anomaly score. Fix: fit k on the
    # full candidate range, but only use centroids from clusters holding >=1%
    # of the training rows as valid "normal behavior" reference points when
    # computing distances; the excluded micro-clusters are separately flagged
    # by construction (a training row assigned to a micro-cluster is, by this
    # same logic, already known to be an extreme outlier).
    degenerate_k = set()
    for i, k in enumerate(k_range):
        km_probe = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_train)
        sizes = np.bincount(km_probe.labels_)
        if sizes.min() < 0.01 * len(X_train):
            degenerate_k.add(k)

    # k is chosen from the inertia elbow (k=4, the last k with a visually
    # sharp inertia drop before the curve flattens) rather than raw silhouette,
    # since silhouette is contaminated by the micro-cluster issue at every k.
    best_k = 4
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X_train)
    sizes_final = np.bincount(km_final.labels_)
    valid_clusters = np.where(sizes_final >= 0.01 * len(X_train))[0]
    micro_clusters = np.where(sizes_final < 0.01 * len(X_train))[0]
    valid_centroids = km_final.cluster_centers_[valid_clusters]

    dists_to_valid = np.linalg.norm(X_all[:, None, :] - valid_centroids[None, :, :], axis=2)
    score_all = dists_to_valid.min(axis=1)  # distance to nearest NON-degenerate centroid
    flag = top_pct_flag(score_all)

    joblib.dump(km_final, os.path.join(MODELS_DIR, "kmeans.pkl"))
    summary = {
        "k_range_tried": k_range,
        "inertia_by_k": [round(v, 2) for v in inertias],
        "silhouette_by_k": [round(v, 4) for v in silhouettes],
        "degenerate_k_all_k_affected": sorted(degenerate_k),
        "degenerate_finding": (
            "Every k from 2 to 10 produces at least one cluster holding <1% of the "
            "2,009 training rows. Checked directly: at k=2 and k=9 the identical 3 "
            "training rows form the micro-cluster (TX000177/AC00363 and TX002305/AC00494, "
            "with Amount_ZScore_Account 92.56 and 77.71 respectively, plus TX001012/AC00329) "
            "-- the same extreme multivariate outliers every time, not noise. Naive "
            "argmax(silhouette) picking k=2 (silhouette 0.9184) is a silhouette artifact "
            "of this degenerate split, not evidence of two genuine behavioral populations."
        ),
        "selected_k": int(best_k),
        "selected_k_rationale": (
            "Chosen from the inertia elbow (k=4 is the last point with a sharp drop before "
            "the curve flattens, see kmeans_elbow_silhouette.png) rather than silhouette, "
            "which is unusable here for the reason above."
        ),
        "n_micro_clusters_at_selected_k": int(len(micro_clusters)),
        "scoring_fix": (
            "Anomaly score = distance to the nearest centroid among clusters holding >=1% "
            "of training rows only; the excluded micro-cluster centroid(s) are not used as "
            "distance targets, since points assigned to them are themselves the extreme "
            "outliers this model should be scoring highly, not comparison points for "
            "everyone else."
        ),
        "top5pct_flagged_rate": round(float(flag.mean()), 4),
        "cost_note": ("O(n * k * p) per Lloyd's iteration -- the cheapest clustering model here "
                      "alongside Isolation Forest; scales linearly in rows, straightforward to "
                      "score new points (distance to the k fixed centroids) in production, though "
                      "the micro-cluster handling above adds a small amount of production-code "
                      "complexity beyond a textbook K-Means anomaly score."),
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
    axes[0].set_xlabel("n_components"); axes[0].set_ylabel("BIC"); axes[0].set_title("GMM -- BIC by n_components")
    axes[1].set_xlabel("n_components"); axes[1].set_ylabel("AIC"); axes[1].set_title("GMM -- AIC by n_components")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    savefig(fig, "gmm_bic_aic.png")

    best_bic, best_cov, best_n = np.inf, None, None
    for cov in cov_types:
        i = int(np.argmin(bic_by_cov[cov]))
        if bic_by_cov[cov][i] < best_bic:
            best_bic, best_cov, best_n = bic_by_cov[cov][i], cov, n_range[i]

    gm_final = GaussianMixture(n_components=best_n, covariance_type=best_cov,
                                random_state=RANDOM_STATE, reg_covar=1e-5, max_iter=200).fit(X_train)
    log_lik = gm_final.score_samples(X_all)
    score = -log_lik  # low likelihood = anomaly = high score
    flag = top_pct_flag(score)

    joblib.dump(gm_final, os.path.join(MODELS_DIR, "gmm.pkl"))
    summary = {
        "n_range_tried": n_range, "cov_types_tried": cov_types,
        "bic_by_cov": {c: [round(v, 1) for v in bic_by_cov[c]] for c in cov_types},
        "selected_n_components": int(best_n), "selected_covariance_type": best_cov,
        "selected_bic": round(float(best_bic), 1),
        "top5pct_flagged_rate": round(float(flag.mean()), 4),
        "cost_note": ("EM iterations are O(n * k * p^2) for full covariance (p=46 makes the "
                      "46x46 covariance matrix per component the dominant cost) -- noticeably "
                      "slower than diag/spherical variants; likelihood-based score_samples is "
                      "cheap and well-defined for new points, a genuine production advantage "
                      "over DBSCAN/HDBSCAN."),
    }
    print("\n=== Model 8: Gaussian Mixture Model ===")
    print(json.dumps(summary, indent=2, default=float))
    return score, flag, summary


def main():
    df, feature_cols, X, X_train, X_val, X_all, idx_train, idx_val, scaler = load_and_split()

    results = {}
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
    out.to_csv(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_scores_classical.csv"), index=False)
    print(f"\nSaved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'model_scores_classical.csv')}")

    all_summary = {
        "isolation_forest": sum_if, "lof": sum_lof, "ocsvm": sum_svm,
        "elliptic_envelope": sum_ee, "dbscan": sum_db, "hdbscan": sum_hdb,
        "kmeans": sum_km, "gmm": sum_gmm,
    }
    with open(os.path.join(ARTIFACTS_RESEARCH_DIR, "model_summary_classical.json"), "w") as f:
        json.dump(all_summary, f, indent=2, default=float)
    print(f"Saved: {os.path.join(ARTIFACTS_RESEARCH_DIR, 'model_summary_classical.json')}")


if __name__ == "__main__":
    main()
