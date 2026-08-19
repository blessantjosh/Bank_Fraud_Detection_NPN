# Placement-round fraud detection pipeline

Dataset: `data/bank_transactions_data_2.csv` (2,512 transactions, 495 accounts,
no fraud label). Run in order from `src/`:

```
pip install -r ../requirements.txt

python 01_feature_engineering.py
python 02_anomaly_ensemble.py
python 03_confidence_labeling.py
python 04_balancing.py
python 04b_cross_validation.py   # robust CV baseline, train fold only, before any fine-tuning
python 05_train_model.py
python 06_evaluation.py
```

Each stage reads the previous stage's output from `artifacts/` and writes
its own — you can re-run any single stage without repeating earlier ones.
`fe_utils.py` and `config.py` are shared by every stage and by the live
scoring path (`transform_new`/`transform_batch_new`, used by the Argus
dashboard's "Upload & Predict" page — see `dashboard/README.md`), so
training-time and live-scoring feature engineering can never drift apart.

See `LIMITATIONS.md` for the honest caveats to state alongside the results,
`ML_AUDIT_AFTER_FIX.md` for the data-leakage audit and fix methodology, and
`PRESENTATION_MODEL_SUMMARY.md` for a judge-facing summary of the approach
and results.
