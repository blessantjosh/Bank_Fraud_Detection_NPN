# Placement-round fraud detection pipeline

Dataset: `data/bank_transactions_data_2.csv` (2,512 transactions, 495 accounts,
no fraud label). Run in order from `src/`:

```
pip install -r ../requirements.txt

python 01_feature_engineering.py
python 02_anomaly_ensemble.py
python 03_confidence_labeling.py
python 04_balancing.py
python 05_train_model.py
python 06_evaluation.py
streamlit run app_streamlit.py
```

Each stage reads the previous stage's output from `artifacts/` and writes
its own — you can re-run any single stage without repeating earlier ones.
`fe_utils.py` and `config.py` are shared by every stage and by the
Streamlit app, so training-time and live-scoring feature engineering can
never drift apart.

See `LIMITATIONS.md` for the honest caveats to state alongside the results.
