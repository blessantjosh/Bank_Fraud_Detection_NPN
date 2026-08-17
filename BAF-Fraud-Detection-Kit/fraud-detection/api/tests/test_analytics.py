"""
api/tests/test_analytics.py -- GET /model/metrics and GET /model/figures/{name}.

These read the real reports/ artifacts already produced by evaluate.py (see
api/routers/analytics.py), so this test proves the endpoint against the
actual, checked-in report files rather than a mock.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.tests.conftest import make_user


def test_model_metrics_requires_auth(client: TestClient):
    r = client.get("/model/metrics")
    assert r.status_code == 401


def test_model_metrics_basic_for_analyst(client: TestClient, admin_token: str):
    analyst_token = make_user(client, admin_token, "analytics-analyst@example.com", "AnalystPassword123", "FRAUD_ANALYST")
    r = client.get("/model/metrics", headers={"Authorization": f"Bearer {analyst_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["full_detail"] is False
    assert isinstance(body["roc_auc"], float)
    assert isinstance(body["pr_auc"], float)
    assert isinstance(body["tpr_at_5pct_fpr"], float)
    # Basic roles never get fairness / feature importance / confusion detail.
    assert body["fairness"] is None
    assert body["confusion_at_threshold"] is None
    assert body["feature_importance"] == []


def test_model_metrics_full_detail_for_risk_manager(client: TestClient, admin_token: str):
    rm_token = make_user(client, admin_token, "analytics-rm@example.com", "RiskMgrPassword123", "RISK_MANAGER")
    r = client.get("/model/metrics", headers={"Authorization": f"Bearer {rm_token}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["full_detail"] is True
    assert body["fairness"]["protected_attribute"] == "customer_age"
    assert 0.0 <= body["fairness"]["age_gt_threshold"]["fpr"] <= 1.0
    assert 0.0 <= body["fairness"]["age_le_threshold"]["fpr"] <= 1.0
    assert body["confusion_at_threshold"]["tp"] >= 0
    assert len(body["feature_importance"]) > 0
    assert {"feature", "gain", "share"} <= body["feature_importance"][0].keys()


def test_model_metrics_full_detail_for_viewer_denied_detail_not_endpoint(client: TestClient, admin_token: str):
    """Viewer can view predictions, so can call /model/metrics, but only gets
    basic fields -- never full explainability detail."""
    viewer_token = make_user(client, admin_token, "analytics-viewer@example.com", "ViewerPassword123", "VIEWER")
    r = client.get("/model/metrics", headers={"Authorization": f"Bearer {viewer_token}"})
    assert r.status_code == 200
    assert r.json()["full_detail"] is False


def test_model_figures_whitelisted_names_only(client: TestClient, admin_token: str):
    r = client.get("/model/figures/roc_curve.png", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0


def test_model_figures_rejects_unknown_name(client: TestClient, admin_token: str):
    r = client.get("/model/figures/not_a_real_figure.png", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 404


def test_model_figures_rejects_path_traversal(client: TestClient, admin_token: str):
    r = client.get("/model/figures/..%2f..%2fmain.py", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code in (403, 404)


def test_model_figures_gated_to_full_explainability_roles(client: TestClient, admin_token: str):
    analyst_token = make_user(client, admin_token, "analytics-figs-analyst@example.com", "AnalystPassword123", "FRAUD_ANALYST")
    r = client.get("/model/figures/roc_curve.png", headers={"Authorization": f"Bearer {analyst_token}"})
    assert r.status_code == 403

    auditor_token = make_user(client, admin_token, "analytics-figs-auditor@example.com", "AuditorPassword123", "AUDITOR")
    r = client.get("/model/figures/roc_curve.png", headers={"Authorization": f"Bearer {auditor_token}"})
    assert r.status_code == 403
