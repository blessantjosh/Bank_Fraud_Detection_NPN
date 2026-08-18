Slide content — limitations (say this out loud, don't bury it)

- No ground-truth fraud label exists in this dataset. "Fraud" here means
  "3+ of 4 unsupervised detectors, fit only on the training period, flagged
  this transaction as statistically unusual" — a pattern-consistency signal,
  not a verified, investigator-confirmed fraud outcome. The measured
  0.83 test ROC-AUC / 0.56 test PR-AUC show how well XGBoost reproduces the
  anomaly ensemble's own (train-fit) judgment on later, unseen transactions —
  not real-world fraud-catching accuracy against a genuine label.

- The label is circular by construction: features engineered in Stage 1 are
  the same features Stage 2's detectors scored, and the same features
  XGBoost/Random Forest were trained on in Stage 5. Strong SHAP attribution
  on TransactionAmount or Amount_vs_AccountAvg confirms internal consistency,
  not that those are the features a real fraud team would find causal.

- Scale mismatch with the brief: the problem statement describes ~1M rows;
  this dataset has 2,512 transactions across 495 accounts, with only 86
  minority-class rows in the training fold. SMOTE and class-weighting both
  work here, but neither has been stress-tested at production volume or
  against adversarial/evolving fraud patterns.

- Distribution drift between train and val/test is real and visible: three
  of the four anomaly detectors, fit only on the earliest ~64% of
  transactions, flag val/test transactions at 2-4x their training-fold rate
  (see DOCUMENTATION.md Stage 2). This is an honest signal that later
  transactions look less like the training period's notion of "normal," not
  a bug -- but it means the 5% contamination assumption is only accurate for
  the training window, and would need periodic re-fitting in production.

- The cost-based threshold is not production-usable as computed. Sweeping
  the illustrative $5 (false positive) / $250 (false negative) cost ratio on
  the validation fold selects a 0.01 "review" threshold -- which flags 90%
  of the test fold for manual review. This is the mathematically
  cost-minimizing answer given the model's real (leakage-free) probability
  separation, but it is operationally unusable. Before this threshold logic
  goes anywhere near production it needs: (a) the bank's real cost figures,
  not the $5/$250 placeholders, and (b) a review-capacity constraint layered
  on top of pure cost minimization (e.g. "flag at most the top N% by risk"),
  since unconstrained cost minimization under a heavily skewed cost ratio
  will always chase recall at any FP cost unless something caps it.

- Before production deployment: validate against a sample of real,
  investigator-labeled fraud cases from the bank's own case-management
  system; re-tune the anomaly ensemble's contamination assumption (currently
  5%, unverified, and shown above to drift over time) against the bank's
  actual historical fraud rate; and re-run the cost-based threshold sweep
  with the bank's real cost-of-fraud and cost-of-friction figures, with a
  capacity constraint, instead of the illustrative $5/$250 used here.

See ML_AUDIT_AFTER_FIX.md for the full data-leakage audit this pipeline was
rebuilt against, and why every number above is now measured on transactions
the model never touched during training or threshold selection.
