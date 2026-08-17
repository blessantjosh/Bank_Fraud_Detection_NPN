"""
run_pipeline.py -- the full BAF pipeline, end to end.

    python make_sample.py --rows 60000 --out sample_train.csv   # if no real data yet
    python run_pipeline.py --train sample_train.csv

    # real competition data:
    python run_pipeline.py --train train.csv --test test.csv --submit submission.csv

What it does, in order:
  1. Preprocess correctly (sentinels -> NaN + indicators, drop constants, FE)
  2. Split TEMPORALLY (months 0-5 train / 6-7 test) -- never randomly
  3. Run the class-imbalance ABLATION so you can prove which approach wins
  4. Report ROC-AUC, PR-AUC and TPR@5%FPR -- never accuracy alone
  5. Run the predictive-equality fairness analysis on customer_age
  6. Write a submission if a test file is given
"""

from __future__ import annotations

import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

import baf

warnings.filterwarnings("ignore", category=UserWarning)

# Starting point tuned for ~1M rows at a ~1% positive rate.
# min_child_samples is deliberately LARGE: with rare positives, small leaves
# memorise noise and destroy generalisation. This is the parameter that
# matters most on this dataset.
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,

    # --- The three parameters that actually matter at a 1% positive rate ---
    # With ~8k positives in 1M rows, LightGBM's defaults are effectively
    # non-binding and let leaves form on a handful of fraud cases.
    "min_data_in_leaf": 200,          # default 20: far too permissive here
    "min_sum_hessian_in_leaf": 1.0,   # default 1e-3 ~= 0.1 samples at 1% base
                                      # rate -- i.e. no constraint at all
    "max_cat_to_onehot": 8,           # cardinalities are 5/7/7/2/5; the default
                                      # of 4 pushes them into many-vs-many
                                      # splits, which overfit on rare positives

    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,                # without this, bagging is SILENTLY OFF
    "cat_smooth": 50,
    "cat_l2": 10,
    "min_data_per_group": 200,
    "lambda_l1": 0.1,
    "lambda_l2": 5.0,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}


def feval_tpr_at_fpr(target_fpr: float = 0.05):
    """
    Early-stop on the metric you actually care about.

    Selecting checkpoints on AUC when the domain metric is TPR@5%FPR optimises
    ranking across the whole curve instead of the top of it. Pair this with
    metric="None" to disable LightGBM's built-in metric.
    """
    def _f(preds, dataset):
        y = dataset.get_label()
        return f"tpr@{int(target_fpr*100)}%fpr", baf.tpr_at_fpr(y, preds, target_fpr), True
    return _f


def train_lgbm(X_tr, y_tr, X_va, y_va, params=None, num_boost_round=2000,
               scale_pos_weight=None, early_stop_on_tpr=False, target_fpr=0.05):
    p = dict(params or PARAMS)
    if scale_pos_weight is not None:
        p["scale_pos_weight"] = scale_pos_weight

    feval = None
    if early_stop_on_tpr:
        p["metric"] = "None"          # disable built-in AUC
        feval = feval_tpr_at_fpr(target_fpr)

    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(
        p, dtr,
        num_boost_round=num_boost_round,
        valid_sets=[dva],
        feval=feval,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def undersample(X, y, ratio=10, seed=42):
    """Keep all positives, sample `ratio` negatives per positive."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y.to_numpy() == 1)
    neg = np.flatnonzero(y.to_numpy() == 0)
    keep_neg = rng.choice(neg, size=min(len(pos) * ratio, len(neg)), replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return X.iloc[idx], y.iloc[idx]


def prior_correct(p, train_rate, true_rate):
    """
    Undo the prior shift introduced by undersampling.

    After undersampling, predicted probabilities are calibrated to the
    RESAMPLED base rate, not reality. Ranking (AUC) is unaffected, but any
    probability you quote to a business is wrong until you correct it.

        p_true = p * (b / r) / ( p * (b/r) + (1-p) * ((1-b)/(1-r)) )

    where r = resampled positive rate, b = true positive rate.
    """
    a, c = true_rate / train_rate, (1 - true_rate) / (1 - train_rate)
    return (p * a) / (p * a + (1 - p) * c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--submit", default="submission.csv")
    ap.add_argument("--id-col", default=None, help="id column in test.csv for the submission")
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--skip-ablation", action="store_true")
    # VERIFIED for kaggle/1056lab-bank-account-fraud-detection: the organisers
    # split 700k/300k RANDOMLY and `month` appears in both files. So `random`
    # is the correct default HERE, even though the BAF paper (and every generic
    # guide online) uses the temporal protocol. Use --split temporal only if you
    # are reproducing the paper or your own test file is time-separated.
    ap.add_argument("--split", choices=["random", "temporal"], default="random",
                    help="random = mirror this competition's 70/30 split (default); "
                         "temporal = the BAF paper protocol, months 0-5 / 6-7")
    a = ap.parse_args()

    print("=" * 70)
    print("BANK ACCOUNT FRAUD -- PIPELINE")
    print("=" * 70)

    df = baf.load(a.train)
    print(f"\nLoaded {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Fraud rate: {df[baf.TARGET].mean():.4%}  "
          f"(predicting all-zero would score {1 - df[baf.TARGET].mean():.2%} accuracy)")

    if baf.MONTH in df.columns:
        print("\nPrevalence by month (real drift -- worth a slide either way):")
        print(df.groupby(baf.MONTH)[baf.TARGET].agg(["mean", "size"]).to_string())
        print("  If the organisers split RANDOMLY, month is a legitimate feature")
        print("  and this drift is signal. If they split TEMPORALLY, it is shift.")

    df = baf.prepare(df)

    # ---- Split ----------------------------------------------------------
    # Your validation must MIRROR the organisers' split, or your local score
    # will not track the leaderboard. This is the single highest-stakes
    # decision in the whole pipeline.
    use_temporal = (a.split == "temporal") and (baf.MONTH in df.columns)

    if use_temporal:
        tr, va = baf.temporal_split(df)
        print(f"\nTEMPORAL split -> train {len(tr):,} (months 0-5) | "
              f"valid {len(va):,} (months 6-7)")
        print("  `month` will be DROPPED as a feature (cannot generalise forward).")
    else:
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(df, test_size=0.30, stratify=df[baf.TARGET],
                                  random_state=42)
        print(f"\nRANDOM stratified split -> train {len(tr):,} | valid {len(va):,}")
        if baf.MONTH in df.columns:
            print("  `month` KEPT as a feature -- correct when the split is random.")

    # drop_month only when splitting temporally
    X_tr, y_tr = baf.xy(tr, drop_month=use_temporal)
    X_va, y_va = baf.xy(va, drop_month=use_temporal)

    # ---- Ablation: does "data balancing" actually help? ------------------
    # Your objective statement asks for data balancing. Do not assume -- MEASURE.
    # Being the team that tested the assumption beats the team that followed it.
    results = {}

    print("\n" + "=" * 70)
    print("BASELINE: no balancing, raw class distribution")
    print("=" * 70)
    m_base = train_lgbm(X_tr, y_tr, X_va, y_va)
    p_base = m_base.predict(X_va, num_iteration=m_base.best_iteration)
    results["none"] = baf.evaluate(y_va, p_base, a.fpr, "No balancing")

    if not a.skip_ablation:
        print("\n" + "=" * 70)
        print("ABLATION A: scale_pos_weight (cost-sensitive reweighting)")
        print("=" * 70)
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        print(f"  scale_pos_weight = {spw:.1f}")
        m_spw = train_lgbm(X_tr, y_tr, X_va, y_va, scale_pos_weight=spw)
        p_spw = m_spw.predict(X_va, num_iteration=m_spw.best_iteration)
        results["scale_pos_weight"] = baf.evaluate(y_va, p_spw, a.fpr, "scale_pos_weight")

        print("\n" + "=" * 70)
        print("ABLATION B: random undersampling (10:1) + prior correction")
        print("=" * 70)
        Xu, yu = undersample(X_tr, y_tr, ratio=10)
        print(f"  resampled train: {len(yu):,} rows, positive rate {yu.mean():.3%}")
        m_us = train_lgbm(Xu, yu, X_va, y_va)
        p_us_raw = m_us.predict(X_va, num_iteration=m_us.best_iteration)
        p_us = prior_correct(p_us_raw, yu.mean(), y_tr.mean())
        results["undersample_10to1"] = baf.evaluate(y_va, p_us, a.fpr, "Undersample 10:1")
        print(f"  mean predicted prob before correction: {p_us_raw.mean():.4f}")
        print(f"  mean predicted prob after correction:  {p_us.mean():.4f}  "
              f"(true rate {y_va.mean():.4f})")
        print("  -> ranking is unchanged by the correction; CALIBRATION is fixed.")

        # ---- The summary table you put on a slide ------------------------
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY  (this table is your evidence for the judges)")
        print("=" * 70)
        key = f"tpr_at_{int(a.fpr*100)}pct_fpr"
        tbl = pd.DataFrame(results).T[["roc_auc", "pr_auc", key]]
        print(tbl.round(4).to_string())
        best = tbl["roc_auc"].idxmax()
        print(f"\n  Best by ROC-AUC: {best}")
        print("  If differences are within ~0.002, they are noise -- say so, and")
        print("  prefer the simplest method. That honesty is worth more than a")
        print("  fabricated improvement.")

    # ---- Fairness --------------------------------------------------------
    if "customer_age" in X_va.columns:
        print("\n" + "=" * 70)
        print("FAIRNESS -- predictive equality (the metric BAF was built for)")
        print("=" * 70)
        # The paper's protected group is age > 50, NOT >= 50. Ages are rounded
        # to the decade, so the difference is an entire bucket of applicants
        # and it visibly moves the ratio. Get this right -- it is exactly the
        # kind of detail a judge who knows the paper will probe.
        older = (X_va["customer_age"] > 50).astype(int).to_numpy()
        baf.fairness_report(y_va, p_base, older, a.fpr, "customer_age > 50 (paper definition)")

    # ---- Feature importance ---------------------------------------------
    print("\n" + "=" * 70)
    print("TOP 20 FEATURES (gain)")
    print("=" * 70)
    imp = (pd.DataFrame({
        "feature": m_base.feature_name(),
        "gain": m_base.feature_importance("gain"),
    }).sort_values("gain", ascending=False).head(20))
    imp["share"] = imp["gain"] / imp["gain"].sum()
    print(imp.to_string(index=False))
    eng = imp[imp.feature.str.contains(
        "burst|mismatch|thin_file|no_valid|limit_to|per_risk|is_missing|"
        "emails_per|zip_density|risk_x|dob_emails|total_address|short_session",
        regex=True)]
    print(f"\n  {len(eng)} of the top 20 are engineered features "
          f"-- evidence your FE earned its place.")

    # ---- Submission ------------------------------------------------------
    if a.test:
        print("\n" + "=" * 70)
        print("BUILDING SUBMISSION")
        print("=" * 70)
        te = baf.load(a.test)
        ids = te[a.id_col].copy() if (a.id_col and a.id_col in te.columns) else None
        te_p = baf.prepare(te, verbose=False)

        # Align to training columns exactly.
        for c in X_tr.columns:
            if c not in te_p.columns:
                te_p[c] = np.nan
        X_te = te_p[X_tr.columns]
        for c in baf.CATEGORICAL_COLS:
            if c in X_te.columns:
                X_te[c] = X_te[c].astype("category")

        # Refit on ALL labelled data before predicting -- strictly more signal.
        X_all, y_all = baf.xy(df, drop_month=use_temporal)
        m_final = lgb.train(
            PARAMS, lgb.Dataset(X_all, y_all),
            num_boost_round=int(m_base.best_iteration * 1.1),
        )
        pred = m_final.predict(X_te)

        sub = pd.DataFrame({
            (a.id_col or "id"): ids if ids is not None else np.arange(len(pred)),
            "fraud_bool": pred,
        })
        sub.to_csv(a.submit, index=False)
        print(f"  wrote {a.submit}  ({len(sub):,} rows)")
        print("  NOTE: submit PROBABILITIES, not 0/1 labels -- AUC needs the ranking.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
