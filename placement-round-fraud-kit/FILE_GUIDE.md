# File Guide — Placement Round Fraud Kit

Every file in this project, what it is, and why it's there. Dataset throughout: `data/bank_transactions_data_2.csv` — 2,512 transactions, 495 accounts, **no fraud label** (this is an unsupervised anomaly-detection project, not a supervised classifier).

---

## Root files

| File | What it is |
|---|---|
| `README.md` | Top-level project overview |
| `DOCUMENTATION.md` | Technical writeup of the **v1 pipeline** (the original, smaller build in `src/`) — stage-by-stage explanation with real measured numbers, from the leakage-fixed pipeline |
| `LIMITATIONS.md` | Honest caveats for v1: no real fraud label exists (the label is 4 detectors' own consensus), the dataset is far smaller than a real production brief would need, the label is circular by construction, the cost-optimal threshold is not directly production-usable |
| `ML_AUDIT_AFTER_FIX.md` | The full data-leakage audit: what leaked in the original v1 pipeline, exactly how each issue was fixed, and the before/after methodology |
| `PRESENTATION_MODEL_SUMMARY.md` | Judge-facing summary of the v1 approach and results, in plain language |
| `requirements.txt` | Every Python package this whole project needs (pandas, sklearn, xgboost, shap, torch, umap-learn, hdbscan, optuna, fastapi, uvicorn, etc.) |
| `FILE_GUIDE.md` | This file |

---

## `data/` — the dataset

| File | What it is |
|---|---|
| `bank_transactions_data_2.csv` | The raw dataset. 16 columns: TransactionID, AccountID, TransactionAmount, TransactionDate, TransactionType, Location, DeviceID, IP Address, MerchantID, Channel, CustomerAge, CustomerOccupation, TransactionDuration, LoginAttempts, AccountBalance, PreviousTransactionDate (this last one is broken — a data-export artifact, dropped everywhere) |

---

## `src/` — the v1 pipeline (leakage-fixed)

A working, leakage-free pipeline: engineer features (train-only stats) → 4 unsupervised detectors, fit on train
only, vote → confidence-tiered labels → train XGBoost (x2) + Random Forest on the train fold → compare all three →
pick a cost-based threshold on validation → evaluate once on test → explain with SHAP → serve live scoring from the
Bank Transaction Fraud & Anomaly Detection dashboard's "Upload & Predict" page (`dashboard/backend/api_server.py`, no standalone demo app). See
`ML_AUDIT_AFTER_FIX.md` for the full before/after leakage audit.

| File | What it is |
|---|---|
| `config.py` | Shared file paths and constants for the whole v1 pipeline (one place to edit, not per-script) |
| `fe_utils.py` | The feature-engineering logic, split into fit-on-train / apply-to-any-fold steps — used both to build the training set AND to score brand-new transactions live (single-row `transform_new`, batch-CSV `transform_batch_new`) from the Bank Transaction Fraud & Anomaly Detection dashboard, so training and live-scoring never drift apart |
| `01_feature_engineering.py` | Chronological train/val/test split, then builds the 20-feature matrix (every fitted statistic/encoder/scaler comes from the train fold only) |
| `02_anomaly_ensemble.py` | Fits 4 unsupervised detectors (Isolation Forest, LOF via `novelty=True`, One-Class SVM, Elliptic Envelope) on the train fold only, then predicts out-of-sample on val/test |
| `03_confidence_labeling.py` | Turns the vote count into High/Medium/Normal risk tiers and a binary fraud label, per fold |
| `04_balancing.py` | Reads the chronological split (already made in Stage 1) and balances only the training fold with SMOTE (also computes class-weighting as a comparison) |
| `04b_cross_validation.py` | Robust 5-fold stratified CV on the training fold only, for all 3 models' baseline hyperparameters — run BEFORE any fine-tuning, so a future hyperparameter search has a variance-aware pre-tuning benchmark and a ready-made train-fold-only harness to plug into |
| `05_train_model.py` | Trains XGBoost+SMOTE, XGBoost+class-weight, and Random Forest (new, for the model comparison), plus a small decision tree for explainability — all on the train fold only |
| `06_evaluation.py` | Compares all 3 models on val + test, picks the primary XGBoost variant from measured test PR-AUC, sweeps the cost-based threshold on val only, applies it once to test, and runs SHAP on the selected model |

---

## `artifacts/` — v1's saved outputs

| File | What it is |
|---|---|
| `features.csv` / `features_scaled.csv` | The engineered feature matrix (unscaled / StandardScaler-scaled), each row tagged with a `split` column (train/val/test) and a `TransactionID` identity key |
| `reference.pkl` | Lookup tables (per-account history, train-fit encoders, full-dataset stats) needed to score a brand-new transaction the same way training did |
| `scaler.pkl` | The StandardScaler, fit on the training fold only |
| `anomaly_votes.csv` | Which of the 4 detectors flagged each transaction, per fold |
| `labeled.csv` | The full engineered dataset + risk tier + binary label + split |
| `split.pkl` | The chronological train/val/test split (features + resampled training data) |
| `cv_per_fold.csv` | Per-fold precision/recall/F1/ROC-AUC/PR-AUC/confusion-matrix counts for all 3 models across the 5-fold CV, train fold only |
| `cv_summary.csv` / `.json` | Mean ± std of each metric per model across the 5 CV folds — the pre-tuning robust baseline |
| `xgb_model.json` | XGBoost trained with SMOTE |
| `xgb_model_classweight.json` | XGBoost trained with class-weighting (no synthetic data) |
| `random_forest_model.pkl` | Random Forest (class_weight="balanced") — the new third model added for comparison |
| `xgb_model_best.json` | Copy of whichever XGBoost variant Stage 6 measured as primary — this is what the Bank Transaction Fraud & Anomaly Detection dashboard's "Upload & Predict" page loads |
| `best_model_choice.json` | Which XGBoost variant was picked as primary, and the measured PR-AUC reasoning |
| `model_comparison.csv` / `.json` | Precision/recall/F1/ROC-AUC/PR-AUC/FP/FN/TP/TN for all 3 models, on both val and test |
| `final_test_evaluation.json` | The one-time final test-set numbers at both the default and VAL-selected thresholds, plus Approve/Review/Block counts |
| `shap_global_importance.csv` | Mean \|SHAP\| ranking for every feature, on the primary model |
| `decision_tree.pkl` / `decision_tree_rules.txt` | A shallow, human-readable decision tree trained on the training fold, for an easy explanation slide |
| `thresholds.json` | The two operating thresholds (review/block), selected on the VALIDATION fold, never on test |
| `plots/` | Saved PNGs: cost-vs-threshold curve, model comparison bar chart, decision tree diagram, SHAP summary/waterfall plots |

---

## `research/` — the main deliverable: 15 reports, one per phase

This is the full 17-phase build (some phases share a report file). **Start with #15 if you only read one.**

| File | Phase | What's inside |
|---|---|---|
| `01_business_understanding.md` | 1 | What "normal" vs "suspicious" banking behavior means for this exact schema; a fraud-scenario table (account takeover, transaction bursts, mule accounts, money laundering, synthetic identity, unusual spending) honestly scored on whether this dataset can actually detect each one |
| `02_data_understanding.md` | 2 | Every one of the 16 raw columns profiled: type, range, mean/median/skew/kurtosis for numerics, cardinality/frequency for categoricals, temporal patterns for dates (including the finding that all transactions fall on weekdays in a 3-hour window — an export artifact) |
| `03_data_quality_and_eda.md` | 3–4 | Missing values (0%), duplicates (0), 5 different outlier-detection methods compared side by side; full EDA — univariate distributions, correlation/VIF/chi-square, PCA/t-SNE |
| `04_feature_engineering.md` | 5 | The 46 engineered features: velocity counts, rolling stats, deviation/z-scores, cyclical time encoding, cross-account device/IP/merchant sharing (the mule-account proxy) |
| `05_feature_selection_and_preprocessing.md` | 6–7 | Why RobustScaler beats StandardScaler/MinMaxScaler/QuantileTransformer here (measured, not assumed); frequency vs. label vs. one-hot encoding compared for `Location`; UMAP vs. PCA vs. t-SNE; the autoencoder's architecture and reconstruction error |
| `06_model_development.md` | 8 | All 12 anomaly models built, tuned, and compared against each other (agreement/disagreement matrix) |
| `07_hyperparameter_optimization.md` | 9 | Optuna (Bayesian search) vs. grid/random search, for Isolation Forest, GMM, and the VAE |
| `08_evaluation.md` | 10 | Internal validity metrics (silhouette etc.), a bootstrap-stability check (how consistent flagging is across retrains), and concrete top-1%/2%/5%/10% business-evaluation examples with real transaction IDs |
| `09_explainability.md` | 11 | SHAP explanations for Isolation Forest and the Autoencoder — global feature importance and specific flagged-transaction walkthroughs |
| `10_ensemble_scoring.md` | 12 | 4 ways to combine all 12 models into one score (weighted average, rank aggregation, percentile aggregation, a stacking proxy) — compared and one recommended |
| `11_threshold_optimization.md` | 13 | Percentile thresholds (95th/97th/99th/99.5th) and statistical thresholds (mean+3σ, IQR) — with resulting flagged counts |
| `12_final_model_selection.md` | 14 | A full decision matrix scoring every model + ensemble strategy on detection quality, stability, interpretability, cost, scalability, deployment readiness — with the final recommendation and reasoning |
| `13_deployment_architecture.md` | 15 | The real production pipeline design (ingestion → feature engineering → scoring → alerting → the Bank Transaction Fraud & Anomaly Detection dashboard), referencing this project's actual code, not generic architecture |
| `14_monitoring_framework.md` | 16 | Drift monitoring plan — PSI/KS-statistic thresholds, concept/feature/model/alert-volume drift, concrete trigger rules |
| `15_final_research_report.md` | 17 | **The standalone executive summary** — reads on its own, covers everything, this is the one to hand a stakeholder or judge |

`research/plots/` — 42 PNG charts referenced throughout the reports above (distributions, correlation heatmaps, UMAP/t-SNE/PCA projections, SHAP plots, threshold curves, etc.)

---

## `src_research/` — the actual code behind every number in `research/`

| File | What it does |
|---|---|
| `config_research.py` | Shared paths/constants for this whole research pipeline |
| `01_data_understanding.py` | Produces the Phase 2 report's numbers |
| `02_data_quality.py` | Missing/duplicate/outlier analysis (5 methods) |
| `03_eda.py` | All univariate/bivariate/multivariate analysis + plots |
| `04_feature_engineering.py` | Builds all 46 features |
| `05_preprocessing.py` | Scaler and encoding comparisons |
| `06_dim_reduction.py` | PCA/UMAP/t-SNE + trains the Autoencoder |
| `07_models_classical.py` | Models 1–8 (Isolation Forest, LOF, OCSVM, Elliptic Envelope, DBSCAN, HDBSCAN, K-Means, GMM) |
| `08_models_deep.py` | Models 9–12 (Autoencoder, VAE, LSTM-AE, Hybrid Ensemble) + the cross-model comparison |
| `09_hyperparameter_optimization.py` | Optuna tuning + baseline search comparison |
| `10_evaluation.py` | Internal metrics, stability bootstrap, business evaluation |
| `11_explainability.py` | SHAP for Isolation Forest + Autoencoder |
| `12_ensemble_scoring.py` | The 4 ensemble-combination strategies |
| `13_threshold_optimization.py` | Threshold sweep + business-impact framing |
| `autoencoder_utils.py` | Reusable Autoencoder class + `load_autoencoder()` function |
| `vae_utils.py` | Reusable VAE class (mirrors `autoencoder_utils.py`) |

---

## `artifacts_research/` — every real output number/file behind the reports (~55 files)

Grouped by what they support (all CSV/JSON unless noted):

- **Data understanding/quality**: `dataset_facts.json`, `numeric_summary.csv`, `categorical_summary.json`, `datetime_summary.json`, `data_quality_summary.json`, `outlier_comparison.csv`, `outlier_method_overlap.csv`
- **EDA**: `correlation_pearson.csv` / `correlation_spearman.csv` / `correlation_matrix.csv`, `mutual_information_matrix.csv`, `vif.csv`, `pca_explained_variance.csv` / `pca_loadings.csv`, `crosstab_channel_transactiontype.csv`, `chisq_channel_transactiontype.json`, `amount_by_channel.csv` / `amount_by_transactiontype.csv`, `tsne_embedding.npy`, `umap_embedding_5feat.npy` / `umap_embedding_full.npy`
- **Feature engineering**: `features_v2.csv` (the final 46-feature matrix everything downstream uses), `phase5_diagnostics.json`
- **Preprocessing**: `scaler_comparison.csv`, `scaler_comparison_transactionamount.json`, `scaler_sensitivity_to_outliers.csv`, `encoding_comparison.json`
- **Autoencoder / VAE**: `autoencoder_config.json`, `autoencoder_training_history.json`, `autoencoder_reconstruction_errors.csv`, `vae_config.json`, `vae_training_history.json`, `lstm_ae_config.json`
- **12-model comparison**: `model_scores_classical.csv` / `model_summary_classical.json` (models 1–8), `model_scores_all.csv` (all 12, this is the master file), `model_comparison_summary.json`, `model_pairwise_spearman.csv` / `model_pairwise_jaccard.csv`, `iforest_grid_search.csv`
- **Hyperparameter tuning**: `hyperparameter_optimization_results.json`
- **Evaluation**: `internal_validity_metrics.csv`, `stability_bootstrap_jaccard.csv`, `reconstruction_metrics_summary.json`, `business_evaluation_examples.json`
- **Explainability**: `shap_isolation_forest.csv`, `shap_autoencoder.csv`, `shap_global_importance_comparison.csv`, `shap_local_explanations.json`
- **Ensemble scoring**: `ensemble_scores.csv`, `ensemble_weights.json`, `ensemble_pairwise_comparison.csv`, `ensemble_vs_v1_crosscheck.json`
- **Threshold optimization**: `threshold_analysis.json`, `threshold_flagged_counts.csv`
- **`models/` subfolder**: the actual fitted/trained model files — `isolation_forest.pkl`, `lof.pkl`, `ocsvm.pkl`, `elliptic_envelope.pkl`, `dbscan.pkl`, `hdbscan.pkl`, `kmeans.pkl`, `gmm.pkl`, `vae.pt`, `lstm_ae.pt`, `shared_robust_scaler.pkl` (the Autoencoder's own weights live one level up: `artifacts_research/autoencoder.pt` + `autoencoder_scaler.pkl`)

---

## `notebooks/` — the Colab-runnable notebook

| File | What it is |
|---|---|
| `Fraud_Anomaly_Detection_Pipeline.ipynb` | A single Jupyter notebook covering all 17 phases as real, runnable code — no dashboard, no UI. Runs end-to-end in ~3 minutes with real inline outputs (tables, plots, metrics). Detects if it's running in actual Google Colab and prompts for a data upload; otherwise finds the data automatically. Some expensive steps (full Optuna search, full VAE/LSTM-AE retraining) load the real saved results from `artifacts_research/` instead of retraining from scratch — clearly labeled either way. |

---

## `dashboard/` — Bank Transaction Fraud & Anomaly Detection, the web dashboard

| File | What it is |
|---|---|
| `README.md` | How to run it (`python -m uvicorn backend.api_server:app --reload`, then open the browser) |
| `backend/api_server.py` | The FastAPI server — loads the v1 pipeline's trained model, scores all 2,512 transactions, precomputes SHAP once, serves the API |
| `backend/cache/shap_values.npy` | Cached SHAP values so the server doesn't recompute them every restart |
| `backend/queue_state.json` | Saved state of the Investigation Queue (Approve/Escalate/Block actions persist here) |
| `frontend/index.html` | The single-page app shell |
| `frontend/css/style.css` | All styling — dark/light theme, animations |
| `frontend/js/app.js` | Page navigation, state management |
| `frontend/js/charts.js` | Hand-drawn SVG charts (bar, line, SHAP bars) |
| `frontend/js/api.js` | Talks to the backend API |
| `frontend/js/format.js` | Number/date formatting helpers |
| `frontend/js/icons.js` | Inline SVG icon set |

Pages inside: Overview dashboard, Transaction Explorer (browse/search/click — no manual typing), Investigation Queue, Model Comparison, Explainability, and a secondary What-if Simulator.
