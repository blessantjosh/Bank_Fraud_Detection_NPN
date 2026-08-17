import numpy as np

from src import models
from src.prediction import predict_dataframe, risk_level
from src.preprocessing import Preprocessor


def test_risk_level_boundaries(cfg):
    prob = np.array([0.0, 0.019, 0.02, 0.09, 0.1, 0.29, 0.3, 0.9])
    levels = risk_level(prob, cfg)
    assert list(levels) == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "CRITICAL", "CRITICAL"]


def test_predict_dataframe_end_to_end_with_logistic_regression(synthetic_baf_df, cfg):
    train = synthetic_baf_df.iloc[:1500].reset_index(drop=True)
    new_apps = synthetic_baf_df.iloc[1500:].drop(columns=["fraud_bool"]).reset_index(drop=True)

    pre = Preprocessor(cfg)
    pre.fit(train)
    X_tr = pre.transform_dense(train)
    y_tr = pre.get_target(train)

    model = models.train_logistic_regression(X_tr, y_tr, cfg, {"class_weight": "balanced"}, seed=42)

    meta = {
        "model_type": "logistic_regression",
        "strategy": "class_weight",
        "threshold": 0.5,
        "feature_columns": list(X_tr.columns),
    }
    out = predict_dataframe(new_apps, model, pre, meta, cfg)

    assert "fraud_probability" in out.columns
    assert "fraud_prediction" in out.columns
    assert "risk_level" in out.columns
    assert out["fraud_probability"].between(0, 1).all()
    assert set(out["fraud_prediction"].unique()).issubset({0, 1})
    assert set(out["risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})
    assert len(out) == len(new_apps)
