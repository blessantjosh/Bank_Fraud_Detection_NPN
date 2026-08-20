"""
Phase 9 (v2) -- Hyperparameter Optimization on the teammate's 18-feature matrix.

Mirrors src_research/09_hyperparameter_optimization.py's methodology 1:1, on
the RobustScaler-scaled 18-column matrix instead of the in-house 46-column
one. There is no fraud label anywhere in this project, so "optimize against
what" is a real design decision, stated explicitly per model rather than
left implicit:

  - Isolation Forest: objective = silhouette score between the top-5%-by-
    anomaly-score group and everyone else, on a fixed 1,000-row random
    subsample of the training split (same subsampling convention already
    used for K-Means in 06_models_classical.py), maximize.
  - Gaussian Mixture Model: objective = BIC on the training split, minimize
    -- directly comparable to the grid already computed in
    06_models_classical.py's Model 8.

No deep-learning model is tuned here (removed): the VAE hyperparameter
search block was dropped along with the VAE model itself. Only Isolation
Forest and GMM -- the two classical models this pipeline actually tunes --
remain.

Bayesian (Optuna, TPE) vs. exhaustive-grid vs. random-search baseline is run
for Isolation Forest since that 3-hyperparameter space is small enough to
grid-search completely, giving a real ground-truth best value to compare
against.

Outputs: artifacts_research_v2/hyperparameter_optimization_results.json,
artifacts_research_v2/iforest_grid_search_v2.csv.
Plots: research_v2/plots/{iforest_optuna_vs_baseline_v2.png,
gmm_optuna_search_v2.png}.
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
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import (
    ARTIFACTS_V2_DIR, FEATURE_COLS_V2, MODELS_V2_DIR, PLOTS_V2_DIR, RANDOM_STATE, load_features_v2,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

TOP_PCT = 0.05
SIL_SAMPLE_N = 1000


def savefig(fig, name):
    path = os.path.join(PLOTS_V2_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def load_common():
    df = load_features_v2()
    feature_cols = FEATURE_COLS_V2
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_V2_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    return feature_cols, X_train, X_val


def separation_silhouette(X, score, rng):
    """Internal unsupervised objective: silhouette between the top-5%-by-score
    group and everyone else, on a fixed-size random subsample for speed."""
    n = len(X)
    sample_n = min(SIL_SAMPLE_N, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    Xs, scores_s = X[idx], score[idx]
    thresh = np.percentile(score, 100 * (1 - TOP_PCT))
    labels = (scores_s >= thresh).astype(int)
    if labels.min() == labels.max():
        return -1.0  # degenerate: only one group present in the sample
    return silhouette_score(Xs, labels)


# ---------------------------------------------------- Isolation Forest search
def iforest_objective(trial, X_train, rng):
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_samples = trial.suggest_float("max_samples", 0.1, 1.0)
    max_features = trial.suggest_float("max_features", 0.3, 1.0)
    clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                           max_features=max_features, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train)
    score = -clf.decision_function(X_train)
    return separation_silhouette(X_train, score, rng)


def iforest_grid_search(X_train, rng, grid):
    rows = []
    t0 = time.time()
    for n_estimators in grid["n_estimators"]:
        for max_samples in grid["max_samples"]:
            for max_features in grid["max_features"]:
                clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                       max_features=max_features, random_state=RANDOM_STATE, n_jobs=-1)
                clf.fit(X_train)
                score = -clf.decision_function(X_train)
                sil = separation_silhouette(X_train, score, rng)
                rows.append({"n_estimators": n_estimators, "max_samples": max_samples,
                             "max_features": max_features, "silhouette": sil})
    elapsed = time.time() - t0
    grid_df = pd.DataFrame(rows)
    best_row = grid_df.loc[grid_df["silhouette"].idxmax()]
    return grid_df, best_row, elapsed


def run_iforest_search(X_train):
    rng = np.random.RandomState(RANDOM_STATE)

    # --- exhaustive grid search baseline (ground truth for this space) ---
    grid = {"n_estimators": [50, 100, 200, 300, 500],
            "max_samples": [0.3, 0.5, 0.8, 1.0],
            "max_features": [0.5, 0.7, 1.0]}
    n_grid_combos = len(grid["n_estimators"]) * len(grid["max_samples"]) * len(grid["max_features"])
    grid_df, best_grid, grid_time = iforest_grid_search(X_train, rng, grid)
    grid_df.to_csv(os.path.join(ARTIFACTS_V2_DIR, "iforest_grid_search_v2.csv"), index=False)

    # --- random search baseline, same budget as Optuna ---
    n_trials = 30
    rng_rand = np.random.RandomState(RANDOM_STATE)
    random_rows = []
    t0 = time.time()
    for _ in range(n_trials):
        n_estimators = int(rng_rand.choice(range(50, 501, 50)))
        max_samples = float(rng_rand.uniform(0.1, 1.0))
        max_features = float(rng_rand.uniform(0.3, 1.0))
        clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                               max_features=max_features, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train)
        score = -clf.decision_function(X_train)
        sil = separation_silhouette(X_train, score, rng)
        random_rows.append({"n_estimators": n_estimators, "max_samples": max_samples,
                             "max_features": max_features, "silhouette": sil})
    random_time = time.time() - t0
    random_df = pd.DataFrame(random_rows)
    best_random = random_df.loc[random_df["silhouette"].idxmax()]

    # --- Optuna (TPE, Bayesian) ---
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(lambda trial: iforest_objective(trial, X_train, rng), n_trials=n_trials)
    optuna_time = time.time() - t0

    optuna_trials_df = study.trials_dataframe()
    optuna_trials_df["best_so_far"] = optuna_trials_df["value"].cummax()
    random_df_curve = random_df.copy()
    random_df_curve["best_so_far"] = random_df_curve["silhouette"].cummax()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(range(1, n_trials + 1), optuna_trials_df["best_so_far"], marker="o",
            color="#2F6690", label=f"Optuna (TPE), {n_trials} trials")
    ax.plot(range(1, n_trials + 1), random_df_curve["best_so_far"], marker="s",
            color="#D1495B", label=f"Random search, {n_trials} trials")
    ax.axhline(best_grid["silhouette"], color="#4C956C", ls="--",
               label=f"Exhaustive grid best ({n_grid_combos} combos)")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Silhouette (top-5%-flagged vs. rest, internal objective)")
    ax.set_title("Isolation Forest (v2, 18 features) -- Optuna vs. Random Search vs. Exhaustive Grid")
    ax.legend()
    savefig(fig, "iforest_optuna_vs_baseline_v2.png")

    result = {
        "objective": "Silhouette between top-5%-by-score group and rest (1,000-row subsample of train), maximize",
        "search_space": {"n_estimators": "50-500", "max_samples": "0.1-1.0", "max_features": "0.3-1.0"},
        "exhaustive_grid": {
            "n_combinations": n_grid_combos, "elapsed_sec": round(grid_time, 2),
            "best_params": {k: (float(best_grid[k]) if k != "n_estimators" else int(best_grid[k]))
                             for k in ["n_estimators", "max_samples", "max_features"]},
            "best_silhouette": round(float(best_grid["silhouette"]), 4),
        },
        "random_search": {
            "n_trials": n_trials, "elapsed_sec": round(random_time, 2),
            "best_params": {k: (float(best_random[k]) if k != "n_estimators" else int(best_random[k]))
                             for k in ["n_estimators", "max_samples", "max_features"]},
            "best_silhouette": round(float(best_random["silhouette"]), 4),
        },
        "optuna_tpe": {
            "n_trials": n_trials, "elapsed_sec": round(optuna_time, 2),
            "best_params": study.best_params,
            "best_silhouette": round(float(study.best_value), 4),
        },
    }
    result["verdict"] = (
        f"Exhaustive grid ({n_grid_combos} combos, {grid_time:.1f}s) found silhouette="
        f"{best_grid['silhouette']:.4f}. Optuna/TPE reached {study.best_value:.4f} in only "
        f"{n_trials} trials ({optuna_time:.1f}s) -- "
        + ("matching or exceeding the grid optimum with ~"
           f"{100*(1 - n_trials/n_grid_combos):.0f}% fewer fits, a genuine efficiency win here."
           if study.best_value >= best_grid["silhouette"] - 1e-6 else
           f"{best_grid['silhouette'] - study.best_value:.4f} silhouette below the grid optimum, "
           "i.e. Optuna did not find quite as good a config within the same trial budget on this "
           "particular (small, cheap-to-grid-search) space, reported honestly rather than "
           "rounded up.")
        + f" Random search reached {best_random['silhouette']:.4f} in the same {n_trials}-trial "
          "budget -- "
        + ("essentially tied with Optuna" if abs(study.best_value - best_random["silhouette"]) < 0.01
           else "Optuna " + ("outperformed" if study.best_value > best_random["silhouette"] else "underperformed")
           + " random search")
        + f" ({study.best_value - best_random['silhouette']:+.4f} silhouette difference)."
    )
    print("\n=== Isolation Forest (v2): Optuna vs. Random Search vs. Exhaustive Grid ===")
    print(json.dumps(result, indent=2, default=float))
    return result


# --------------------------------------------------------------------- GMM search
def gmm_objective(trial, X_train):
    n_components = trial.suggest_int("n_components", 2, 10)
    covariance_type = trial.suggest_categorical("covariance_type", ["full", "diag", "tied", "spherical"])
    reg_covar = trial.suggest_float("reg_covar", 1e-6, 1e-3, log=True)
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                          reg_covar=reg_covar, random_state=RANDOM_STATE, max_iter=200).fit(X_train)
    return gm.bic(X_train)


def run_gmm_search(X_train):
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    n_trials = 40
    t0 = time.time()
    study.optimize(lambda trial: gmm_objective(trial, X_train), n_trials=n_trials)
    elapsed = time.time() - t0

    with open(os.path.join(ARTIFACTS_V2_DIR, "model_summary_classical.json")) as f:
        classical_summary = json.load(f)
    grid_best_bic = classical_summary["gmm"]["selected_bic"]
    grid_best_config = {"n_components": classical_summary["gmm"]["selected_n_components"],
                         "covariance_type": classical_summary["gmm"]["selected_covariance_type"]}

    result = {
        "objective": "BIC on the training split (minimize)",
        "search_space": {"n_components": "2-10", "covariance_type": "full/diag/tied/spherical",
                          "reg_covar": "1e-6 to 1e-3 (log-uniform)"},
        "optuna_tpe": {"n_trials": n_trials, "elapsed_sec": round(elapsed, 2),
                       "best_params": study.best_params, "best_bic": round(float(study.best_value), 1)},
        "phase8_grid_reference": {"config": grid_best_config, "bic": grid_best_bic,
                                   "note": ("Phase 8 (v2)'s Model 8 grid used reg_covar=1e-5 fixed and did not "
                                            "search it -- Optuna additionally tunes reg_covar, so a BIC "
                                            "improvement from that alone is expected, not necessarily "
                                            "evidence Optuna 'found a better n_components/covariance_type'.")},
    }
    improvement = grid_best_bic - study.best_value
    result["verdict"] = (
        f"Optuna's best BIC ({study.best_value:.1f}) vs. Phase 8 (v2)'s fixed-reg_covar grid best "
        f"({grid_best_bic:.1f}): {'an improvement of ' + format(improvement, '.1f') if improvement > 0 else 'no improvement, ' + format(-improvement, '.1f') + ' worse'}."
    )
    print("\n=== GMM (v2): Optuna Search ===")
    print(json.dumps(result, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    trials_df = study.trials_dataframe()
    ax.scatter(trials_df["number"], trials_df["value"], c="#2F6690", s=25, alpha=0.7)
    ax.plot(trials_df["number"], trials_df["value"].cummin(), color="#D1495B", lw=1.5, label="Best-so-far BIC")
    ax.axhline(grid_best_bic, color="#4C956C", ls="--", label="Phase 8 (v2) grid best (reg_covar fixed)")
    ax.set_xlabel("Trial number"); ax.set_ylabel("BIC")
    ax.set_title("GMM (v2, 18 features) -- Optuna BIC Search (n_components, covariance_type, reg_covar)")
    ax.legend()
    savefig(fig, "gmm_optuna_search_v2.png")

    return result


def main():
    feature_cols, X_train, X_val = load_common()

    print("=== Phase 9 (v2).1: Isolation Forest hyperparameter search ===")
    if_result = run_iforest_search(X_train)

    print("\n=== Phase 9 (v2).2: GMM hyperparameter search ===")
    gmm_result = run_gmm_search(X_train)

    with open(os.path.join(ARTIFACTS_V2_DIR, "hyperparameter_optimization_results.json"), "w") as f:
        json.dump({"isolation_forest": if_result, "gmm": gmm_result}, f, indent=2, default=float)
    print(f"\nSaved: {os.path.join(ARTIFACTS_V2_DIR, 'hyperparameter_optimization_results.json')}")


if __name__ == "__main__":
    main()
