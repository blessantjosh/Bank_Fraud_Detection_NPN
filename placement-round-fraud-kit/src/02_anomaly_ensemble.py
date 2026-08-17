"""
STAGE 2 — Unsupervised anomaly detection ensemble.

There is no fraud label in this dataset, so we manufacture one by running
four independent unsupervised detectors over the scaled feature matrix and
recording each one's binary anomaly flag. Stage 3 turns these four votes
into a confidence-tiered label. Using four different detection principles
(tree-partitioning, local density, one-class margin, robust covariance)
means a transaction only gets treated as strong fraud evidence when
multiple different mathematical views of "unusual" agree.

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
"""
import warnings
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

import config

X = pd.read_csv(config.FEATURES_SCALED_CSV)
print(f"Running 4 detectors on {X.shape[0]} transactions x {X.shape[1]} features "
      f"(contamination={config.CONTAMINATION})")

iso = IsolationForest(contamination=config.CONTAMINATION, random_state=config.RANDOM_STATE, n_estimators=200)
flag_if = (iso.fit_predict(X) == -1).astype(int)

lof = LocalOutlierFactor(n_neighbors=20, contamination=config.CONTAMINATION)
flag_lof = (lof.fit_predict(X) == -1).astype(int)

ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=config.CONTAMINATION)
flag_ocsvm = (ocsvm.fit_predict(X) == -1).astype(int)

# DeviceNoveltyFlag is ~99.5% constant (almost every transaction is a "new"
# device for its account), which makes FastMCD's robust covariance estimate
# ill-conditioned on some resamples. This only produces benign convergence
# warnings, not invalid output -- MCD still ranks/flags by Mahalanobis
# distance correctly -- so we suppress them rather than let them drown the
# real output. Documented as a known limitation for small/sparse datasets.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mcd = EllipticEnvelope(contamination=config.CONTAMINATION, random_state=config.RANDOM_STATE, support_fraction=0.9)
    flag_mcd = (mcd.fit_predict(X) == -1).astype(int)

votes = pd.DataFrame({
    "flag_isoforest": flag_if,
    "flag_lof": flag_lof,
    "flag_ocsvm": flag_ocsvm,
    "flag_mcd": flag_mcd,
})
votes["vote_count"] = votes.sum(axis=1)

print("\nPer-detector anomaly rate:")
for col in ["flag_isoforest", "flag_lof", "flag_ocsvm", "flag_mcd"]:
    print(f"  {col:18s} {votes[col].mean()*100:5.2f}%  ({votes[col].sum()} txns)")
print("\nVote-count distribution:")
print(votes["vote_count"].value_counts().sort_index())

votes.to_csv(config.ANOMALY_VOTES_CSV, index=False)
print(f"\nSaved {config.ANOMALY_VOTES_CSV}")
