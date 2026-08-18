"""
STAGE 3 — Confidence-tiered labeling.

A flat majority vote would throw away information: a transaction that 3-4
detectors flag is qualitatively different evidence from one only 2 flag.
Instead we keep three tiers that mirror how a real fraud-ops team triages
alerts (auto-block / manual review / clear):

  vote_count 3-4   -> "High confidence fraud"   (near-unanimous agreement)
  vote_count 2     -> "Medium confidence / needs review"
  vote_count 0-1   -> "Normal"

For the supervised model we still need one binary target, so High + Medium
collapse into "Fraud" vs "Normal" -- but the 3-tier column is kept in
labeled.csv for the demo/output, since "needs review" is a real, useful
operational category that a binary label throws away.
"""
import pandas as pd

import config

votes = pd.read_csv(config.ANOMALY_VOTES_CSV)
features = pd.read_csv(config.FEATURES_CSV)

assert (votes["split"].values == features["split"].values).all(), \
    "votes and features rows must be in the same train/val/test row order"


def tier(v):
    if v >= 3:
        return "High confidence fraud"
    if v == 2:
        return "Medium confidence / needs review"
    return "Normal"


votes["risk_tier"] = votes["vote_count"].apply(tier)
votes["is_fraud"] = votes["risk_tier"].isin(
    ["High confidence fraud", "Medium confidence / needs review"]
).astype(int)

labeled = pd.concat(
    [features, votes[["vote_count", "risk_tier", "is_fraud"]]], axis=1
)

print("3-tier distribution (all folds):")
print(labeled["risk_tier"].value_counts())
print(f"\nBinary label distribution (is_fraud, all folds): {labeled['is_fraud'].value_counts().to_dict()}")
print(f"Fraud prevalence (all folds): {labeled['is_fraud'].mean()*100:.2f}%")
for fold in ["train", "val", "test"]:
    sub = labeled[labeled["split"] == fold]
    print(f"  {fold:5s}: {sub['is_fraud'].sum()} fraud / {len(sub)} rows "
          f"({sub['is_fraud'].mean()*100:.2f}%)")

labeled.to_csv(config.LABELED_CSV, index=False)
print(f"\nSaved {config.LABELED_CSV}")
