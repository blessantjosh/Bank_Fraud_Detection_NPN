"""
STAGE 7 -- Streamlit demo.

Two modes, both backed by the real pipeline artifacts -- nothing on either
screen is fabricated:

  Score New Transaction   -- engineers a new transaction's features the same
                             way Stage 1 did (fe_utils.transform_new, reusing
                             the saved account-history reference), scores it
                             with the Stage 5 XGBoost model, maps probability
                             to a tier using the Stage 6 cost-optimal
                             thresholds, and shows the top-3 SHAP drivers.

  Search Identifier History -- looks up a real AccountID / DeviceID /
                             IP Address / MerchantID against the actual
                             confidence-tiered labels produced in Stage 3
                             (labeled.csv), joined back to the raw
                             transaction rows it came from. If an identifier
                             never appears in the data, it says so -- it
                             does not invent a result.

Run with:  streamlit run app_streamlit.py
"""
import json
from datetime import datetime

import joblib
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

import config
import fe_utils as fe

st.set_page_config(page_title="Fraud Risk Console", layout="wide")

# ---------------------------------------------------------------------------
# Visual theme: an official, institutional look (deep blue header/nav, white
# card panels, a formal disclaimer block, a dark footer) -- a styling choice,
# not a claim of affiliation with any government or regulator.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
  .stApp { background: #eef1f5; }
  .block-container { padding-top: 0 !important; max-width: 980px; }

  .gov-topbar {
    background: #0d3b66; color: #d9e6f2; font-size: 0.78rem;
    padding: 0.35rem 1.1rem; margin: 0 -1rem; letter-spacing: 0.02em;
  }
  .gov-header {
    background: #ffffff; border-bottom: 3px solid #0d3b66;
    padding: 1.1rem 1.1rem 0.9rem; margin: 0 -1rem 0;
  }
  .gov-header h1 { margin: 0; font-size: 1.55rem; color: #14243b; font-weight: 700; }
  .gov-header p { margin: 0.2rem 0 0; color: #4a5a6a; font-size: 0.92rem; }

  .gov-nav { background: #14487a; margin: 0 -1rem 1.4rem; padding: 0 0.6rem; }
  .gov-breadcrumb { color: #5a6b7a; font-size: 0.82rem; margin: 0.9rem 0 1.1rem; }
  .gov-breadcrumb b { color: #14243b; }

  div[data-testid="stTabs"] [data-testid="stTab"] {
    background: #14487a; border-radius: 0 !important; font-weight: 600;
  }
  div[data-testid="stTabs"] [data-testid="stTab"] p { color: #cfe0f2 !important; margin: 0.3rem 0.5rem; }
  div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] { background: #ffffff; }
  div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p { color: #0d3b66 !important; }
  div[data-testid="stTabs"] .react-aria-SelectionIndicator { background: #0d3b66 !important; }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #d6dde4 !important; border-radius: 3px;
  }
  .gov-disclaimer {
    background: #fbfbf6; border: 1px solid #e3e0c8; border-left: 4px solid #a68b1f;
    padding: 0.9rem 1.1rem; font-size: 0.85rem; color: #4a4530; margin-top: 1.2rem;
  }
  .gov-footer {
    background: #14243b; color: #93a3b5; font-size: 0.76rem;
    margin: 2rem -1rem 0; padding: 0.9rem 1.1rem;
  }
  .stButton>button {
    background: #1e7d4a; color: white; border: none; border-radius: 3px;
    font-weight: 600; padding: 0.45rem 1.4rem;
  }
  .stButton>button:hover { background: #166138; color: white; }
  .risk-badge {
    display: inline-block; padding: 0.25rem 0.8rem; border-radius: 3px;
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    reference = joblib.load(config.REFERENCE_PKL)
    model = XGBClassifier()
    model.load_model(config.MODEL_JSON)
    with open(config.THRESHOLDS_JSON) as f:
        thresholds = json.load(f)
    explainer = shap.TreeExplainer(model)
    raw = fe.load_raw(config.RAW_CSV)
    labeled = pd.read_csv(config.LABELED_CSV)

    # labeled.csv was built by fit_engineer() sorting on (AccountID,
    # TransactionDate, TransactionID) before engineering -- re-apply the
    # identical sort to the raw rows so row i here lines up with row i of
    # labeled.csv. This is the ONLY way to recover which real identifiers
    # a given tier belongs to, since fit_engineer drops the raw ID columns
    # from the modeling matrix itself.
    raw_sorted = raw.sort_values(["AccountID", "TransactionDate", "TransactionID"]).reset_index(drop=True)
    ledger = pd.concat(
        [raw_sorted, labeled[["vote_count", "risk_tier", "is_fraud"]]], axis=1
    )
    return reference, model, thresholds, explainer, raw, ledger


reference, model, thresholds, explainer, raw, ledger = load_artifacts()

st.markdown('<div class="gov-topbar">Internal fraud-analytics tool &nbsp;|&nbsp; Placement-round hackathon build &nbsp;|&nbsp; Not a production system</div>', unsafe_allow_html=True)
st.markdown("""
<div class="gov-header">
  <h1>Fraud Risk Console</h1>
  <p>Transaction scoring and identifier history, backed by the confidence-tiered anomaly pipeline documented alongside this tool.</p>
</div>
""", unsafe_allow_html=True)

tab_score, tab_search = st.tabs(["Score New Transaction", "Search Identifier History"])

# ============================================================================
# TAB 1 -- score a new transaction
# ============================================================================
with tab_score:
    st.markdown('<div class="gov-breadcrumb">Home &nbsp;›&nbsp; <b>Score New Transaction</b></div>', unsafe_allow_html=True)

    known_accounts = sorted(reference["account_history"].keys())
    known_devices = sorted(raw["DeviceID"].unique())
    known_locations = sorted(raw["Location"].unique())
    known_merchants = sorted(raw["MerchantID"].unique())
    known_occupations = sorted(raw["CustomerOccupation"].unique())

    with st.form("txn_form", border=True):
        col1, col2 = st.columns(2)
        with col1:
            account_id = st.selectbox("Account ID", ["<new account>"] + known_accounts)
            amount = st.number_input("Transaction Amount", min_value=0.01, value=150.0, step=10.0)
            txn_type = st.selectbox("Transaction Type", ["Debit", "Credit"])
            channel = st.selectbox("Channel", ["ATM", "Online", "Branch"])
            location = st.selectbox("Location", known_locations)
            occupation = st.selectbox("Customer Occupation", known_occupations)
        with col2:
            device_id = st.selectbox("Device ID", known_devices + ["<new device>"])
            ip_address = st.text_input("IP Address", "10.0.0.1")
            merchant_id = st.selectbox("Merchant ID", known_merchants)
            age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
            duration = st.number_input("Transaction Duration (seconds)", min_value=1, value=90)
            login_attempts = st.number_input("Login Attempts", min_value=1, max_value=10, value=1)
            balance = st.number_input("Account Balance", min_value=0.0, value=5000.0, step=100.0)

        submitted = st.form_submit_button("Score transaction")

    if submitted:
        txn = {
            "AccountID": None if account_id == "<new account>" else account_id,
            "TransactionAmount": amount,
            "TransactionType": txn_type,
            "Location": location,
            "DeviceID": device_id,
            "IP Address": ip_address,
            "MerchantID": merchant_id,
            "Channel": channel,
            "CustomerAge": age,
            "CustomerOccupation": occupation,
            "TransactionDuration": duration,
            "LoginAttempts": login_attempts,
            "AccountBalance": balance,
            "TransactionDate": datetime.now(),
        }

        row = fe.transform_new(txn, reference)
        proba = float(model.predict_proba(row)[:, 1][0])

        if proba >= thresholds["block_threshold"]:
            tier, bg, fg = "BLOCK", "#f9d7d3", "#8a2b20"
        elif proba >= thresholds["review_threshold"]:
            tier, bg, fg = "MANUAL REVIEW", "#f7e6c4", "#8a5a10"
        else:
            tier, bg, fg = "AUTO-APPROVE", "#d7ecdd", "#1e6b3c"

        with st.container(border=True):
            st.markdown(
                f'<span class="risk-badge" style="background:{bg};color:{fg}">{tier}</span>'
                f'&nbsp;&nbsp; Fraud probability: <b>{proba*100:.1f}%</b>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Thresholds -- review at {thresholds['review_threshold']*100:.1f}%, "
                f"block at {thresholds['block_threshold']*100:.1f}% "
                f"(cost-optimal thresholds from Stage 6, not fixed 50/50 rules of thumb)."
            )

            shap_values = explainer(row)
            contributions = pd.Series(shap_values.values[0], index=row.columns).sort_values(
                key=lambda s: s.abs(), ascending=False
            )
            top3 = contributions.head(3)

            st.markdown("**Top 3 features driving this decision**")
            for feat, val in top3.items():
                direction = "increases" if val > 0 else "decreases"
                st.write(f"- **{feat}** = {row[feat].iloc[0]:.3g} -- {direction} fraud risk "
                         f"(SHAP contribution {val:+.3f})")

# ============================================================================
# TAB 2 -- search an identifier's real history in the labeled dataset
# ============================================================================
with tab_search:
    st.markdown('<div class="gov-breadcrumb">Home &nbsp;›&nbsp; <b>Search Identifier History</b></div>', unsafe_allow_html=True)

    with st.container(border=True):
        id_type = st.radio(
            "Identifier type",
            ["Account ID", "Device ID", "IP Address", "Merchant ID"],
            horizontal=True,
        )
        id_column = {
            "Account ID": "AccountID",
            "Device ID": "DeviceID",
            "IP Address": "IP Address",
            "Merchant ID": "MerchantID",
        }[id_type]

        query = st.text_input(f"Enter {id_type}", "")
        search_clicked = st.button("Search")

    if search_clicked and query.strip():
        matches = ledger[ledger[id_column].astype(str).str.strip().str.lower() == query.strip().lower()]

        with st.container(border=True):
            if matches.empty:
                st.markdown(
                    f'<span class="risk-badge" style="background:#e4e7ea;color:#4a5a6a">NOT FOUND</span>'
                    f'&nbsp;&nbsp; No transactions with {id_type} <b>{query}</b> exist in this dataset.',
                    unsafe_allow_html=True,
                )
            else:
                tier_counts = matches["risk_tier"].value_counts()
                n_high = int(tier_counts.get("High confidence fraud", 0))
                n_medium = int(tier_counts.get("Medium confidence / needs review", 0))
                n_normal = int(tier_counts.get("Normal", 0))

                if n_high > 0:
                    badge, bg, fg = "HIGH-CONFIDENCE FLAGS FOUND", "#f9d7d3", "#8a2b20"
                elif n_medium > 0:
                    badge, bg, fg = "REVIEW FLAGS FOUND", "#f7e6c4", "#8a5a10"
                else:
                    badge, bg, fg = "NO FLAGS FOUND", "#d7ecdd", "#1e6b3c"

                st.markdown(
                    f'<span class="risk-badge" style="background:{bg};color:{fg}">{badge}</span>'
                    f'&nbsp;&nbsp; {id_type} <b>{query}</b> -- {len(matches)} transaction(s) on record',
                    unsafe_allow_html=True,
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("High confidence", n_high)
                c2.metric("Needs review", n_medium)
                c3.metric("Normal", n_normal)
                c4.metric("Avg. amount", f"{matches['TransactionAmount'].mean():.2f}")

                st.markdown("**Matching transactions** (most recent first)")
                display_cols = ["TransactionID", "TransactionDate", "TransactionAmount",
                                 "TransactionType", "Channel", "risk_tier", "vote_count"]
                st.dataframe(
                    matches.sort_values("TransactionDate", ascending=False)[display_cols].head(50),
                    hide_index=True, use_container_width=True,
                )
    elif search_clicked:
        st.warning(f"Enter a {id_type} to search.")

    st.markdown(
        '<div class="gov-disclaimer"><b>Disclaimer.</b> Risk tiers shown on this page come from an '
        'unsupervised anomaly-detection ensemble (Isolation Forest, LOF, One-Class SVM, MCD) applied to '
        'this dataset -- they are a pattern-consistency signal, not a verified, investigator-confirmed '
        'fraud outcome, because this dataset carries no ground-truth fraud label. See the accompanying '
        'documentation for the full methodology and known limitations before treating any result here as '
        'a real fraud determination.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="gov-footer">Fraud Risk Console -- built for a placement-round hackathon exercise. '
    'Source pipeline, methodology, and limitations documented separately. Best viewed in a modern browser.</div>',
    unsafe_allow_html=True,
)
