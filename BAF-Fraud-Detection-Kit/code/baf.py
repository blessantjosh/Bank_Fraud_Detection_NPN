"""
baf.py — Bank Account Fraud toolkit.

Everything you need for the BAF dataset in one importable module:
sentinel handling, feature engineering, temporal splitting, the
domain-correct metrics, and the fairness analysis.

Designed for the NeurIPS 2022 Feedzai BAF suite (Base variant).

Tested on: pandas 3.0.5, numpy 2.4.6, scikit-learn 1.9.0, lightgbm 4.7.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

TARGET = "fraud_bool"
MONTH = "month"

# Columns where a negative value means MISSING, per the official Feedzai datasheet.
# Leaving these as -1 lets a tree half-recover the signal by accident;
# median-imputing them destroys it. We do neither -- see to_nan_and_flag().
SENTINEL_COLS = [
    "prev_address_months_count",     # -1 missing, range [-1, 380]
    "current_address_months_count",  # -1 missing, range [-1, 429]
    "bank_months_count",             # -1 missing, range [-1, 32]
    "session_length_in_minutes",     # -1 missing, range [-1, 107]
    "device_distinct_emails_8w",     # -1 missing, range [-1, 2]
    "intended_balcon_amount",        # negatives missing, range [-16, 114]
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]


# --------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------

MAX_LOAD_BYTES = 2 * 1024 ** 3  # 2 GB -- Base.csv is ~200MB; this just catches runaway/malformed input


def load(path: str) -> pd.DataFrame:
    """Read a BAF csv. Handles the unnamed index column some exports carry."""
    import os
    size = os.path.getsize(path)
    if size > MAX_LOAD_BYTES:
        raise ValueError(
            f"{path} is {size / 1e9:.1f} GB, over the {MAX_LOAD_BYTES / 1e9:.0f} GB sanity limit. "
            "If this is genuinely expected, raise baf.MAX_LOAD_BYTES."
        )
    df = pd.read_csv(path)
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    return df.drop(columns=junk) if junk else df


def drop_constant(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop zero-variance columns. In BAF Base this removes device_fraud_count."""
    const = [c for c in df.columns if c != TARGET and df[c].nunique(dropna=False) <= 1]
    if verbose and const:
        print(f"  dropping constant columns: {const}")
    return df.drop(columns=const)


def to_nan_and_flag(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convert sentinel negatives to NaN AND add an explicit _is_missing indicator.

    This is the single highest-value preprocessing step on this dataset.
    Missingness is *itself* predictive: a synthetic identity has no previous
    address precisely because it was invented last week. Keeping both the
    indicator and a real NaN gives the model both signals; GBDTs learn
    optimal NaN routing natively, so no imputation is needed or wanted.
    """
    df = df.copy()
    for col in SENTINEL_COLS:
        if col not in df.columns:
            continue
        miss = df[col] < 0
        df[f"{col}_is_missing"] = miss.astype("int8")
        df.loc[miss, col] = np.nan
        if verbose:
            print(f"  {col}: {miss.mean():.1%} missing -> NaN + indicator")
    return df


def set_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast object/string columns to pandas 'category'.

    LightGBM consumes category dtype natively and splits on subsets of levels,
    which beats one-hot encoding for anonymised categoricals like these.
    Note: pandas 3.0 reads text as 'str' dtype, not 'object', so we check both.
    """
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineered features, each tied to a concrete account-opening fraud pattern.

    Three fraud archetypes drive every feature here:
      (A) synthetic identity  -- fabricated person, thin file, incoherent attributes
      (B) identity theft      -- real person, wrong human, contactability fails
      (C) mule farming        -- bulk applications, shared attributes, bursts
    """
    df = df.copy()
    eps = 1e-6

    # --- (C) Velocity ratios -------------------------------------------------
    # The three velocity_* columns share units (applications/hour) over 6h/24w/4w
    # windows, so their ratios measure ACCELERATION against the long-run baseline.
    # A burst is far more informative than an absolute level, and a divisive
    # relationship is expensive for axis-aligned tree splits to approximate --
    # which is exactly why handing it over as a feature pays.
    #
    # velocity_6h ranges to -175 (generator artefact). Clip at 0 before forming
    # ratios, or a negative numerator flips the sign and makes a burst look calm.
    # velocity_24h and velocity_4w have strictly positive ranges -- safe denominators.
    if "velocity_6h" in df.columns:
        v6 = df["velocity_6h"].clip(lower=0)
        if "velocity_4w" in df.columns:
            df["velocity_burst_6h_4w"] = v6 / (df["velocity_4w"] + eps)
        if "velocity_24h" in df.columns:
            df["velocity_ratio_6h_24h"] = v6 / (df["velocity_24h"] + eps)
    if {"velocity_24h", "velocity_4w"}.issubset(df.columns):
        df["velocity_burst_24h_4w"] = df["velocity_24h"] / (df["velocity_4w"] + eps)

    # --- (A) Synthetic identity: the email/name coherence cluster ------------
    # A real person's email usually resembles their name. A generated identity's
    # does not. Combine with a free provider and the signal sharpens.
    if {"name_email_similarity", "email_is_free"}.issubset(df.columns):
        df["email_mismatch_free"] = (
            (1.0 - df["name_email_similarity"]) * df["email_is_free"]
        )
    # One date of birth attached to many distinct emails is close to a working
    # definition of automated synthetic-identity generation.
    if {"date_of_birth_distinct_emails_4w", "name_email_similarity"}.issubset(df.columns):
        df["dob_emails_x_mismatch"] = (
            df["date_of_birth_distinct_emails_4w"] * (1.0 - df["name_email_similarity"])
        )

    # --- (A) Thin file: no history because the identity is new ---------------
    if {"prev_address_months_count", "current_address_months_count"}.issubset(df.columns):
        df["total_address_history"] = (
            df["prev_address_months_count"].fillna(0) + df["current_address_months_count"].fillna(0)
        )
    thin_parts = [c for c in ["prev_address_months_count_is_missing",
                              "bank_months_count_is_missing"] if c in df.columns]
    if thin_parts:
        # How many independent history checks came back empty.
        df["thin_file_score"] = df[thin_parts].sum(axis=1).astype("int8")

    # The CROSS-COLUMN aggregate is the feature with real added value here.
    # A GBDT can already isolate any single -1 with one split, so the per-column
    # indicators are largely redundant on their own -- their true job is to let
    # us NaN the sentinels without losing information, which in turn stops -1
    # from poisoning every ratio above. But "how many checks came back empty",
    # summed across columns, is a genuinely new signal: a tree would need one
    # split per column to reconstruct it.
    miss_cols = [f"{c}_is_missing" for c in SENTINEL_COLS if f"{c}_is_missing" in df.columns]
    if len(miss_cols) >= 2:
        df["n_missing"] = df[miss_cols].sum(axis=1).astype("int8")

    # --- (B) Contactability: can anyone actually reach this human ------------
    phones = [c for c in ["phone_home_valid", "phone_mobile_valid"] if c in df.columns]
    if len(phones) == 2:
        df["n_valid_phones"] = df["phone_home_valid"] + df["phone_mobile_valid"]
        # Both invalid is a much stronger signal than either alone.
        df["no_valid_phone"] = (df["n_valid_phones"] == 0).astype("int8")

    # --- (A) Financial coherence: does the ask match the profile -------------
    # Fraudsters maximise take. The RATIO to income is the signal, not the level.
    if {"proposed_credit_limit", "income"}.issubset(df.columns):
        df["limit_to_income"] = df["proposed_credit_limit"] / (df["income"] + eps)
    if {"proposed_credit_limit", "credit_risk_score"}.issubset(df.columns):
        # A high requested limit on a weak internal score is incoherent.
        df["limit_per_risk"] = df["proposed_credit_limit"] / (df["credit_risk_score"] + 200.0)
    if {"credit_risk_score", "income"}.issubset(df.columns):
        df["risk_x_income"] = df["credit_risk_score"] * df["income"]

    # --- (C) Device & session behaviour: humans vs scripts --------------------
    if {"session_length_in_minutes", "device_distinct_emails_8w"}.issubset(df.columns):
        df["emails_per_session_min"] = (
            df["device_distinct_emails_8w"] / (df["session_length_in_minutes"] + 1.0)
        )
    if {"keep_alive_session", "session_length_in_minutes"}.issubset(df.columns):
        # Bots do not fiddle with session preferences.
        df["short_session_no_keepalive"] = (
            (df["session_length_in_minutes"] < 5) & (df["keep_alive_session"] == 0)
        ).astype("int8")

    # --- (C) Geographic / branch clustering ----------------------------------
    if {"zip_count_4w", "velocity_4w"}.issubset(df.columns):
        df["zip_density_vs_velocity"] = df["zip_count_4w"] / (df["velocity_4w"] + eps)

    return df


def prepare(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full preprocessing chain in the correct order."""
    if verbose:
        print("Preparing data...")
    df = drop_constant(df, verbose=verbose)
    df = to_nan_and_flag(df, verbose=verbose)
    df = add_features(df)
    df = set_categoricals(df)
    if verbose:
        print(f"  final shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, train_months=(0, 1, 2, 3, 4, 5), test_months=(6, 7)):
    """
    The official Feedzai protocol: first six months train, last two test.

    Random splitting on this dataset leaks the future into the past and
    produces validation scores that will not survive the leaderboard.
    """
    if MONTH not in df.columns:
        raise KeyError(
            "No 'month' column -- cannot split temporally. "
            "If your competition file has no month, the organisers split randomly; "
            "use stratified KFold instead and say so in your write-up."
        )
    tr = df[df[MONTH].isin(train_months)].copy()
    te = df[df[MONTH].isin(test_months)].copy()
    return tr, te


def xy(df: pd.DataFrame, drop_month: bool = True):
    """Split into features/target. `month` is a splitting key, never a feature."""
    drop = [TARGET] + ([MONTH] if drop_month and MONTH in df.columns else [])
    return df.drop(columns=drop), df[TARGET]


# --------------------------------------------------------------------------
# Metrics -- the ones that actually matter here
# --------------------------------------------------------------------------

def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    """
    Recall at a fixed false-positive budget -- the metric the BAF paper uses.

    Chosen by Feedzai because in fraud detection every false positive is a
    genuine customer wrongly rejected. A bank fixes its tolerable FPR first,
    then asks how much fraud you catch within it.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    """The score cutoff that yields the requested FPR. Use for deployment."""
    fpr, _, thr = roc_curve(y_true, y_score)
    idx = int(np.searchsorted(fpr, target_fpr, side="right")) - 1
    idx = max(0, min(idx, len(thr) - 1))
    return float(thr[idx])


def evaluate(y_true, y_score, target_fpr: float = 0.05, label: str = "") -> dict:
    """Report the full picture, not one flattering number."""
    y_true = np.asarray(y_true)
    res = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        f"tpr_at_{int(target_fpr*100)}pct_fpr": tpr_at_fpr(y_true, y_score, target_fpr),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    if label:
        print(f"\n--- {label} ---")
        for k, v in res.items():
            print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")
        # The line that shows you understand the problem.
        print(f"  {'accuracy_if_predict_all_0':<24} {1 - res['positive_rate']:.4f}  <- the do-nothing baseline")
    return res


# --------------------------------------------------------------------------
# Fairness -- predictive equality, the BAF paper's metric
# --------------------------------------------------------------------------

def fairness_report(y_true, y_score, group, target_fpr: float = 0.05,
                    label: str = "") -> dict:
    """
    Predictive equality across a protected group, measured correctly.

    The whole point: pick ONE global threshold that achieves the target FPR
    overall, then measure each group's FPR at THAT SAME threshold. Comparing
    groups at different thresholds is the most common way people get this wrong.

    A false positive here means a real person is denied a bank account.

    `group` should be a boolean/int array. Use (customer_age > 50) -- the BAF
    paper's protected group is strictly GREATER than 50, and because ages are
    rounded to the decade, using >= 50 silently moves a whole bucket of
    applicants across the line and changes the reported ratio.

    Returns per-group FPR/TPR and the FPR ratio as min/max, so 1.0 means
    parity and smaller is worse. (The naive max/min convention plots upside
    down and is a common way to misreport this.)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    group = np.asarray(group)

    thr = threshold_at_fpr(y_true, y_score, target_fpr)
    pred = (y_score >= thr).astype(int)

    out = {}
    for g in np.unique(group):
        m = group == g
        neg, pos = m & (y_true == 0), m & (y_true == 1)
        out[f"group_{g}"] = {
            "n": int(m.sum()),
            "fpr": float(pred[neg].mean()) if neg.sum() else float("nan"),
            "tpr": float(pred[pos].mean()) if pos.sum() else float("nan"),
            "prevalence": float(y_true[m].mean()) if m.sum() else float("nan"),
        }

    fprs = [v["fpr"] for v in out.values() if not np.isnan(v["fpr"])]
    ratio = (min(fprs) / max(fprs)) if fprs and max(fprs) > 0 else float("nan")
    out["fpr_ratio"] = ratio
    out["threshold"] = thr

    if label:
        print(f"\n--- Fairness: {label} (at global FPR={target_fpr:.0%}) ---")
        for k, v in out.items():
            if isinstance(v, dict):
                print(f"  {k}: n={v['n']:>7,}  FPR={v['fpr']:.4f}  "
                      f"TPR={v['tpr']:.4f}  prevalence={v['prevalence']:.4f}")
        print(f"  predictive equality (FPR ratio, 1.0 = parity): {ratio:.3f}")
        if not np.isnan(ratio) and ratio < 0.8:
            worse = 1 / ratio if ratio > 0 else float("inf")
            print(f"  -> one group is falsely flagged {worse:.1f}x more often. "
                  f"Report this honestly; it is the paper's own finding.")
    return out
