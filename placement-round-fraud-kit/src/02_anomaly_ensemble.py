"""
STAGE 2 -- Unsupervised anomaly detection ensemble.

There is no fraud label in this dataset, so we manufacture one by running
four independent unsupervised detectors and recording each one's binary
anomaly flag. Stage 3 turns these four votes into a confidence-tiered label.
Using four different detection principles (tree-partitioning, local density,
one-class margin, robust covariance) means a transaction only gets treated as
strong fraud evidence when multiple different mathematical views of "unusual"
agree.

  1. Isolation Forest   - isolates points via random tree partitioning;
                           anomalies need fewer splits to isolate.
  2. Local Outlier Factor - flags points whose local density is much lower
                           than their neighbors' (catches local anomalies
                           that a global method like IsolationForest can miss).
  3. One-Class SVM      - learns a soft boundary around the dense region of
                           "normal" feature space; anything outside is flagged.
  4. Minimum Covariance Determinant (via EllipticEnvelope) - robust
                           Mahalanobis distance: fits a robust covariance
                           estimate that isn't itself dragged off by the
                           outliers it's trying to detect, then flags points
                           far from the robust center.

LEAKAGE FIX (see ML_AUDIT_AFTER_FIX.md): all four detectors are FIT ON THE
TRAIN FOLD ONLY, then used to PREDICT (out-of-sample) on val and test. None
of them ever sees a val/test row during fitting.

LocalOutlierFactor's default mode (novelty=False) only supports
`fit_predict` on the exact data it was fit on -- it has no out-of-sample
`predict`. We use `novelty=True`, which is scikit-learn's documented, valid
way to fit LOF on one dataset and score genuinely different data with it
(the one documented trade-off: calling `.predict()` on the very rows used to
fit emits a benign UserWarning, which we fit once and reuse for train's own
votes too, to keep train/val/test methodology identical).
"""
import warnings
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

import config

df = pd.read_csv(config.FEATURES_SCALED_CSV)
split = df.pop("split")
X_train = df[split == "train"]
X_val = df[split == "val"]
X_test = df[split == "test"]
print(f"Fitting 4 detectors on {len(X_train)} TRAIN rows only "
      f"(val={len(X_val)}, test={len(X_test)}) x {df.shape[1]} features "
      f"(contamination={config.CONTAMINATION})")


def fit_predict_all(fit_fn, predict_fn):
    """Fit once on train, predict on train/val/test with the SAME fitted model."""
    model = fit_fn()
    return {
        "train": predict_fn(model, X_train),
        "val": predict_fn(model, X_val),
        "test": predict_fn(model, X_test),
    }


iso_out = fit_predict_all(
    lambda: IsolationForest(contamination=config.CONTAMINATION,
                             random_state=config.RANDOM_STATE, n_estimators=200).fit(X_train),
    lambda m, X: (m.predict(X) == -1).astype(int),
)

with warnings.catch_warnings():
    # novelty=True LOF warns when .predict() is called on its own fit data
    # (train) -- documented, benign; we want train scored the same way as
    # val/test so the vote methodology is identical across all three folds.
    warnings.simplefilter("ignore")
    lof_out = fit_predict_all(
        lambda: LocalOutlierFactor(n_neighbors=20, contamination=config.CONTAMINATION,
                                    novelty=True).fit(X_train),
        lambda m, X: (m.predict(X) == -1).astype(int),
    )

ocsvm_out = fit_predict_all(
    lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=config.CONTAMINATION).fit(X_train),
    lambda m, X: (m.predict(X) == -1).astype(int),
)

# DeviceNoveltyFlag is ~99.5% constant (almost every transaction is a "new"
# device for its account), which makes FastMCD's robust covariance estimate
# ill-conditioned on some resamples. This only produces benign convergence
# warnings, not invalid output -- MCD still ranks/flags by Mahalanobis
# distance correctly -- so we suppress them rather than let them drown the
# real output. Documented as a known limitation for small/sparse datasets.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mcd_out = fit_predict_all(
        lambda: EllipticEnvelope(contamination=config.CONTAMINATION, random_state=config.RANDOM_STATE,
                                  support_fraction=0.9).fit(X_train),
        lambda m, X: (m.predict(X) == -1).astype(int),
    )

parts = []
for fold in ["train", "val", "test"]:
    votes = pd.DataFrame({
        "flag_isoforest": iso_out[fold],
        "flag_lof": lof_out[fold],
        "flag_ocsvm": ocsvm_out[fold],
        "flag_mcd": mcd_out[fold],
    })
    votes["vote_count"] = votes.sum(axis=1)
    votes["split"] = fold
    parts.append(votes)
votes = pd.concat(parts, ignore_index=True)

print("\nPer-detector anomaly rate by fold:")
for fold in ["train", "val", "test"]:
    sub = votes[votes["split"] == fold]
    rates = ", ".join(f"{c}={sub[c].mean()*100:.2f}%" for c in
                       ["flag_isoforest", "flag_lof", "flag_ocsvm", "flag_mcd"])
    print(f"  {fold:5s} (n={len(sub)}): {rates}")

print("\nVote-count distribution (all folds):")
print(votes["vote_count"].value_counts().sort_index())

votes.to_csv(config.ANOMALY_VOTES_CSV, index=False)
print(f"\nSaved {config.ANOMALY_VOTES_CSV}")
