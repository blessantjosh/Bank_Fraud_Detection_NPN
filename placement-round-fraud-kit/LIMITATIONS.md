Slide content — limitations (say this out loud, don't bury it)

- No ground-truth fraud label exists in this dataset. "Fraud" here means
  "3+ of 4 unsupervised detectors independently flagged this transaction as
  statistically unusual" — a pattern-consistency signal, not a verified,
  investigator-confirmed fraud outcome. The 0.97 ROC-AUC measures how well
  XGBoost reproduces the anomaly ensemble's own judgment, not real-world
  fraud-catching accuracy.

- Scale mismatch with the brief: the problem statement describes ~1M rows;
  this dataset has 2,512 transactions across 495 accounts, with only ~107
  minority-class rows in the training fold. SMOTE and class-weighting both
  work here, but neither has been stress-tested at production volume or
  against adversarial/evolving fraud patterns.

- The label is circular by construction: features engineered in Stage 1 are
  the same features Stage 2's detectors scored, and the same features
  XGBoost was trained on in Stage 5. Strong SHAP attribution on
  Amount_vs_AccountAvg or DeviceNoveltyFlag confirms internal consistency,
  not that those are the features a real fraud team would find causal.

- Before production deployment: validate against a sample of real,
  investigator-labeled fraud cases from the bank's own case-management
  system; re-tune the anomaly ensemble's contamination assumption (currently
  5%, unverified) against the bank's actual historical fraud rate; and
  re-run the cost-based threshold sweep with the bank's real cost-of-fraud
  and cost-of-friction figures instead of the illustrative $5/$250 used here.
