"""
api/schemas.py -- Pydantic request/response models.

Every model uses `extra="forbid"` so unexpected fields are rejected rather
than silently ignored, and every numeric field has an explicit type + range
matching the BAF dataset bible (../../01-DATASET-BIBLE.md), with a small
margin beyond the observed training extremes (documented per-field) so a
legitimate edge-case application isn't falsely rejected, while garbage/
out-of-domain input still is.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from api.models_db import Role


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- auth -------------------------------------------------------------------

class LoginRequest(StrictModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(StrictModel):
    """Either a full token pair, or (if MFA is required) an mfa_token that
    must be redeemed at /auth/mfa/verify. Never both at once."""
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "bearer"
    mfa_required: bool = False
    mfa_token: str | None = None


class RefreshRequest(StrictModel):
    refresh_token: str


class TokenPairResponse(StrictModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MfaVerifyRequest(StrictModel):
    mfa_token: str
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class MfaSetupResponse(StrictModel):
    secret: str
    provisioning_uri: str


class MfaEnableRequest(StrictModel):
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


# --- admin / users ------------------------------------------------------

class UserCreateRequest(StrictModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=256)
    role: Role


class UserResponse(StrictModel):
    id: str
    email: str
    role: Role
    mfa_enabled: bool
    is_active: bool
    created_at: str


class UserUpdateRequest(StrictModel):
    role: Role | None = None
    is_active: bool | None = None


# --- prediction input row (one BAF application) -----------------------------
# Ranges per 01-DATASET-BIBLE.md, widened slightly beyond training extremes
# (documented inline) to avoid false rejections on legitimate edge cases.

class ApplicationRow(StrictModel):
    income: float = Field(ge=0.0, le=1.0, description="decile-form income, 0.1-0.9 in training data")
    customer_age: int = Field(ge=10, le=100)
    employment_status: str = Field(min_length=1, max_length=8)
    housing_status: str = Field(min_length=1, max_length=8)
    name_email_similarity: float = Field(ge=0.0, le=1.0)
    prev_address_months_count: int = Field(ge=-1, le=500)
    current_address_months_count: int = Field(ge=-1, le=500)
    bank_months_count: int = Field(ge=-1, le=60)
    days_since_request: float = Field(ge=0.0, le=200.0)
    velocity_6h: float = Field(ge=-1000.0, le=30000.0)
    velocity_24h: float = Field(ge=-1000.0, le=30000.0)
    velocity_4w: float = Field(ge=-1000.0, le=30000.0)
    zip_count_4w: int = Field(ge=0, le=10000)
    bank_branch_count_8w: int = Field(ge=0, le=5000)
    date_of_birth_distinct_emails_4w: int = Field(ge=0, le=100)
    phone_home_valid: int = Field(ge=0, le=1)
    phone_mobile_valid: int = Field(ge=0, le=1)
    email_is_free: int = Field(ge=0, le=1)
    foreign_request: int = Field(ge=0, le=1)
    device_os: str = Field(min_length=1, max_length=16)
    device_distinct_emails_8w: int = Field(ge=-1, le=5)
    session_length_in_minutes: float = Field(ge=-1.0, le=300.0)
    keep_alive_session: int = Field(ge=0, le=1)
    source: str = Field(min_length=1, max_length=16)
    credit_risk_score: float = Field(ge=-500.0, le=1000.0)
    proposed_credit_limit: float = Field(ge=0.0, le=10000.0)
    payment_type: str = Field(min_length=1, max_length=8)
    has_other_cards: int = Field(ge=0, le=1)
    intended_balcon_amount: float = Field(ge=-100.0, le=500.0)
    month: int = Field(ge=0, le=11)
    device_fraud_count: int = Field(ge=0, le=1)

    @field_validator("employment_status", "housing_status", "device_os", "source", "payment_type")
    @classmethod
    def _no_control_chars(cls, v: str) -> str:
        if any(ord(ch) < 32 for ch in v):
            raise ValueError("control characters are not allowed")
        return v


class PredictRequest(StrictModel):
    rows: list[ApplicationRow] = Field(min_length=1, max_length=1000)


class PredictionItemBasic(StrictModel):
    """Returned to all roles permitted to call /predict (Fraud Analyst,
    Risk Manager, Admin)."""
    fraud_probability: float
    fraud_prediction: int
    risk_level: str


class FeatureContribution(StrictModel):
    feature: str
    value: float | str | None = None
    contribution: float
    method: str  # "shap" (tree models) or "linear_contribution" (approximation, see README)


class PredictionItemFull(PredictionItemBasic):
    """Returned only to Admin / Risk Manager (full explainability detail)."""
    top_features: list[FeatureContribution] = Field(default_factory=list)


class PredictResponse(StrictModel):
    model_version: str
    threshold_used: float
    row_count: int
    predictions: list[PredictionItemBasic] | list[PredictionItemFull]


# --- audit / prediction history -----------------------------------------

class AuditLogEntry(StrictModel):
    id: str
    timestamp: str
    user_id: str | None
    role: str | None
    action: str
    resource: str
    result: str
    request_id: str
    detail: str | None = None


class PredictionRecordEntry(StrictModel):
    id: str
    timestamp: str
    user_id: str | None
    model_version: str
    input_row_count: int
    risk_level_counts: dict
    threshold_used: float


class ThresholdUpdateRequest(StrictModel):
    threshold: float = Field(ge=0.0, le=1.0)


class HealthResponse(StrictModel):
    status: str
    model_loaded: bool
    model_checksum_verified: bool | None
    mfa_required: bool


# --- model analytics (reports/metrics + reports/figures, read-only) ---------

class ConfusionAtThreshold(StrictModel):
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float


class FairnessGroup(StrictModel):
    n: int
    fpr: float
    tpr: float
    prevalence: float


class FairnessSummary(StrictModel):
    """Predictive-equality finding (BAF paper), reproduced on the real held-out
    test set: customer_age > 50 vs <= 50, both measured at the same global
    threshold (see src/evaluation.py fairness_report)."""
    protected_attribute: str
    protected_threshold: float
    age_le_threshold: FairnessGroup
    age_gt_threshold: FairnessGroup
    fpr_ratio: float
    fairness_eval_threshold: float


class FeatureImportanceItem(StrictModel):
    feature: str
    gain: float
    share: float


class ModelMetricsResponse(StrictModel):
    """GET /model/metrics. `available=False` (with everything else null/empty)
    means the report files have not been generated yet -- never a fabricated
    number. Fields beyond the headline four are only populated when the
    caller's role passes full_explainability_allowed()."""
    available: bool
    full_detail: bool
    model: str | None = None
    strategy: str | None = None
    threshold: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    tpr_at_5pct_fpr: float | None = None
    n: int | None = None
    n_positive: int | None = None
    positive_rate: float | None = None
    confusion_at_threshold: ConfusionAtThreshold | None = None
    fairness: FairnessSummary | None = None
    feature_importance: list[FeatureImportanceItem] = Field(default_factory=list)
