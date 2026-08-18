"""
Phase 8 (v2), Part 2 -- Models 9-12 (deep learning + ensemble) plus the
cross-model comparison across all 12 models, on the teammate's 18-feature
matrix.

Model 9  -- Autoencoder: REUSED directly from Phase 7
            (artifacts_research_v2/autoencoder.pt / autoencoder_scaler.pkl /
            autoencoder_config.json). Not retrained.
Model 10 -- Variational Autoencoder: trained fresh in PyTorch, same
            train/val row-level split as everything else.
Model 11 -- LSTM Autoencoder: re-verified sequence-length distribution on
            THIS file (04_feature_verification.py already confirmed it is
            identical to the in-house pipeline's finding: 495 accounts,
            mean 5.075 txns, 428/495 accounts >=3 txns covering 2,402/2,512
            rows) -- same scoped-coverage approach applied here.
Model 12 -- Hybrid Ensemble: Isolation Forest + LOF + Autoencoder majority
            vote (>=2 of 3).
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_research_v2 import (
    ARTIFACTS_V2_DIR, FEATURE_COLS_V2, MODELS_V2_DIR, PLOTS_V2_DIR, RANDOM_STATE, load_features_v2,
)
from autoencoder_utils import load_autoencoder, reconstruction_errors
from vae_utils import VAE, train_vae, vae_reconstruction_errors

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
})

TOP_PCT = 0.05


def savefig(fig, name):
    path = os.path.join(PLOTS_V2_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")
    return path


def top_pct_flag(score, pct=TOP_PCT):
    thresh = np.percentile(score, 100 * (1 - pct))
    return (score >= thresh).astype(int)


def load_common():
    df = load_features_v2()
    feature_cols = FEATURE_COLS_V2
    X = df[feature_cols].astype(float).values
    idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE)
    scaler = joblib.load(os.path.join(MODELS_V2_DIR, "shared_robust_scaler.pkl"))
    X_train = scaler.transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])
    X_all = scaler.transform(X)
    return df, feature_cols, X, X_train, X_val, X_all, idx_train, idx_val, scaler


# ------------------------------------------------------ Model 9: Autoencoder (reused)
def model_autoencoder_reused(feature_cols, X_all):
    with open(os.path.join(ARTIFACTS_V2_DIR, "autoencoder_config.json")) as f:
        ae_config = json.load(f)
    assert ae_config["feature_cols"] == feature_cols, "Feature schema mismatch vs trained autoencoder"
    model = load_autoencoder(os.path.join(ARTIFACTS_V2_DIR, "autoencoder.pt"),
                              input_dim=ae_config["input_dim"], bottleneck_dim=ae_config["bottleneck_dim"])
    mse, mae, bottleneck, _ = reconstruction_errors(model, X_all)
    flag = top_pct_flag(mse)
    summary = {
        "reused_from": "artifacts_research_v2/autoencoder.pt (Phase 7, not retrained)",
        "architecture": ae_config["architecture"],
        "train_mse_mean_at_training_time": ae_config["train_mse_mean"],
        "val_mse_mean_at_training_time": ae_config["val_mse_mean"],
        "val_mse_p99_at_training_time": ae_config["val_mse_p99"],
        "top5pct_flagged_rate": round(float(flag.mean()), 4),
    }
    print("\n=== Model 9: Autoencoder (reused) ===")
    print(json.dumps(summary, indent=2, default=float))
    return mse, flag, summary


# ---------------------------------------------------------------- Model 10: VAE
def model_vae(feature_cols, X_train, X_val, X_all):
    print("\nTraining VAE (beta=0.1, latent_dim=3, 200 epochs)...")
    t0 = time.time()
    model, history = train_vae(X_train, X_val, latent_dim=3, epochs=200, lr=1e-3,
                                batch_size=64, beta=0.1, random_state=RANDOM_STATE)
    fit_time = time.time() - t0

    train_mse, train_kl = vae_reconstruction_errors(model, X_train)
    val_mse, val_kl = vae_reconstruction_errors(model, X_val)
    all_mse, all_kl = vae_reconstruction_errors(model, X_all)

    score = all_mse
    flag = top_pct_flag(score)

    torch.save(model.state_dict(), os.path.join(MODELS_V2_DIR, "vae.pt"))
    final_metrics = {
        "architecture": "input(18) -> 8 -> 4 -> [mu(3), logvar(3)] -> reparameterize -> 4 -> 8 -> output(18)",
        "beta_kl_weight": 0.1,
        "epochs": 200,
        "fit_time_sec": round(fit_time, 3),
        "train_recon_mse_mean": round(float(train_mse.mean()), 6),
        "val_recon_mse_mean": round(float(val_mse.mean()), 6),
        "val_recon_mse_p95": round(float(np.percentile(val_mse, 95)), 6),
        "val_recon_mse_p99": round(float(np.percentile(val_mse, 99)), 6),
        "val_recon_mse_max": round(float(val_mse.max()), 6),
        "val_kl_mean": round(float(val_kl.mean()), 6),
        "top5pct_flagged_rate": round(float(flag.mean()), 4),
    }
    with open(os.path.join(ARTIFACTS_V2_DIR, "vae_config.json"), "w") as f:
        json.dump({"feature_cols": feature_cols, **final_metrics}, f, indent=2)
    with open(os.path.join(ARTIFACTS_V2_DIR, "vae_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["train_recon_mse"], label="Train recon MSE", color="#2F6690")
    axes[0].plot(history["val_recon_mse"], label="Val recon MSE", color="#D1495B")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].set_title("VAE Training Curve (Reconstruction), v2")
    axes[0].legend()
    axes[1].plot(history["train_kl"], label="Train KL", color="#4C956C")
    axes[1].plot(history["val_kl"], label="Val KL", color="#EDAE49")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("KL divergence"); axes[1].set_title("VAE Training Curve (KL term), v2")
    axes[1].legend()
    savefig(fig, "vae_training_curve_v2.png")

    print("\n=== Model 10: Variational Autoencoder ===")
    print(json.dumps(final_metrics, indent=2, default=float))
    return score, flag, final_metrics


def plot_ae_vs_vae(ae_all_mse, vae_all_mse):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.linspace(0, np.percentile(np.concatenate([ae_all_mse, vae_all_mse]), 99), 60)
    ax.hist(ae_all_mse, bins=bins, alpha=0.55, label="Autoencoder (Model 9)", color="#2F6690")
    ax.hist(vae_all_mse, bins=bins, alpha=0.55, label="VAE (Model 10)", color="#D1495B")
    ax.set_xlabel("Reconstruction MSE (RobustScaler-scaled space, clipped at P99)")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder vs. VAE -- Reconstruction Error Distribution (v2, all 2,512 rows)")
    ax.legend()
    savefig(fig, "ae_vs_vae_reconstruction_error_v2.png")


# ------------------------------------------------------------ Model 11: LSTM-AE
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=12, latent_dim=6):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.encoder_lstm(packed)
        z = self.to_latent(h_n[-1])
        h0 = self.from_latent(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        dec_in = torch.zeros_like(x)
        dec_in[:, 1:, :] = x[:, :-1, :]
        dec_out, _ = self.decoder_lstm(dec_in, (h0, c0))
        recon = self.out_proj(dec_out)
        return recon, z


def build_account_sequences(df, feature_cols, scaler, min_len=3):
    grouped = df.groupby("AccountID", sort=False)
    seqs, row_idx_lists, acc_ids, lengths = [], [], [], []
    for acc_id, g in grouped:
        if len(g) < min_len:
            continue
        row_positions = g.index.values
        X_acc = scaler.transform(g[feature_cols].astype(float).values)
        seqs.append(X_acc)
        row_idx_lists.append(row_positions)
        acc_ids.append(acc_id)
        lengths.append(len(g))
    return seqs, row_idx_lists, acc_ids, lengths


def model_lstm_autoencoder(df, feature_cols, scaler):
    counts = df.groupby("AccountID").size()
    n_accounts_total = len(counts)
    n_accounts_qualify = int((counts >= 3).sum())
    rows_qualify = int(df["AccountID"].isin(counts[counts >= 3].index).sum())

    seq_len_dist = {
        "n_accounts_total": int(n_accounts_total),
        "n_accounts_ge3_txns": n_accounts_qualify,
        "pct_accounts_ge3_txns": round(100 * n_accounts_qualify / n_accounts_total, 2),
        "rows_covered": rows_qualify,
        "pct_rows_covered": round(100 * rows_qualify / len(df), 2),
        "seq_len_min": int(counts.min()), "seq_len_max": int(counts.max()),
        "seq_len_mean": round(float(counts.mean()), 3), "seq_len_median": float(counts.median()),
        "seq_len_value_counts": {int(k): int(v) for k, v in counts.value_counts().sort_index().items()},
    }
    print("\n=== Model 11: LSTM Autoencoder -- feasibility check (re-verified on features_teammate_merged.csv) ===")
    print(json.dumps(seq_len_dist, indent=2))
    print(
        "Verdict (matches the in-house Phase 8 finding, re-confirmed here rather than assumed): "
        "sequences are short (median 5, max 12), 86.5% of accounts have >=3 transactions covering "
        "95.6% of rows -- enough to build a minimal, scoped LSTM-AE on that subset. The remaining "
        "110 rows (4.4%, accounts with 1-2 transactions) get no LSTM-AE score."
    )

    seqs, row_idx_lists, acc_ids, lengths = build_account_sequences(df, feature_cols, scaler, min_len=3)
    max_len = max(lengths)
    n_seq = len(seqs)
    n_features = len(feature_cols)

    X_pad = np.zeros((n_seq, max_len, n_features), dtype=np.float32)
    for i, s in enumerate(seqs):
        X_pad[i, :len(s), :] = s
    lengths_arr = np.array(lengths)

    acc_idx = np.arange(n_seq)
    acc_train_idx, acc_val_idx = train_test_split(acc_idx, test_size=0.2, random_state=RANDOM_STATE)

    Xtr = torch.tensor(X_pad[acc_train_idx], dtype=torch.float32)
    Ltr = torch.tensor(lengths_arr[acc_train_idx], dtype=torch.long)
    Xva = torch.tensor(X_pad[acc_val_idx], dtype=torch.float32)
    Lva = torch.tensor(lengths_arr[acc_val_idx], dtype=torch.long)

    torch.manual_seed(RANDOM_STATE)
    model = LSTMAutoencoder(input_dim=n_features, hidden_dim=12, latent_dim=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    def masked_mse(recon, target, lengths_t):
        mask = torch.zeros_like(target)
        for i, L in enumerate(lengths_t):
            mask[i, :L, :] = 1.0
        se = (recon - target) ** 2 * mask
        return se.sum() / mask.sum()

    epochs = 150
    history = {"train_mse": [], "val_mse": []}
    n_train = Xtr.shape[0]
    batch_size = 32
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        running = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, lb = Xtr[idx], Ltr[idx]
            opt.zero_grad()
            recon, _ = model(xb, lb)
            loss = masked_mse(recon, xb, lb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        train_mse = running / n_train

        model.eval()
        with torch.no_grad():
            recon_va, _ = model(Xva, Lva)
            val_mse = masked_mse(recon_va, Xva, Lva).item()
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:>4}/{epochs}  train_mse={train_mse:.5f}  val_mse={val_mse:.5f}")
    fit_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        X_all_t = torch.tensor(X_pad, dtype=torch.float32)
        L_all_t = torch.tensor(lengths_arr, dtype=torch.long)
        recon_all, _ = model(X_all_t, L_all_t)
        se_all = ((recon_all - X_all_t) ** 2).mean(dim=2).numpy()

    per_row_scores = {}
    for i, row_positions in enumerate(row_idx_lists):
        L = lengths[i]
        for t in range(L):
            per_row_scores[int(row_positions[t])] = float(se_all[i, t])

    score_col = np.full(len(df), np.nan)
    applicable = np.zeros(len(df), dtype=int)
    for pos, sc in per_row_scores.items():
        score_col[pos] = sc
        applicable[pos] = 1

    applicable_scores = score_col[applicable == 1]
    thresh = np.percentile(applicable_scores, 100 * (1 - TOP_PCT))
    flag_col = np.zeros(len(df), dtype=int)
    flag_col[applicable == 1] = (score_col[applicable == 1] >= thresh).astype(int)

    torch.save(model.state_dict(), os.path.join(MODELS_V2_DIR, "lstm_ae.pt"))
    summary = {
        "sequence_length_distribution": seq_len_dist,
        "scope": "428/495 accounts (86.5%), 2,402/2,512 rows (95.6%) -- accounts with >=3 transactions only",
        "architecture": "LSTM(18->12) encoder -> Linear(12->6 latent) -> Linear(6->12) -> LSTM(18->12) decoder -> Linear(12->18), teacher-forced, masked MSE over real (non-padded) timesteps only",
        "split": "account-level 80/20, random_state=42 (NOT the row-level split used elsewhere)",
        "n_accounts_train": int(len(acc_train_idx)), "n_accounts_val": int(len(acc_val_idx)),
        "max_seq_len_padded_to": int(max_len),
        "epochs": epochs, "fit_time_sec": round(fit_time, 3),
        "train_mse_mean": round(float(history["train_mse"][-1]), 6),
        "val_mse_mean": round(float(history["val_mse"][-1]), 6),
        "applicable_rows": int(applicable.sum()),
        "applicable_pct": round(100 * applicable.sum() / len(df), 2),
        "top5pct_flagged_rate_within_applicable_rows": round(float(flag_col[applicable == 1].mean()), 4),
    }
    with open(os.path.join(ARTIFACTS_V2_DIR, "lstm_ae_config.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["train_mse"], label="Train MSE", color="#2F6690")
    ax.plot(history["val_mse"], label="Validation MSE", color="#D1495B")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Masked MSE"); ax.set_title("LSTM Autoencoder Training Curve, v2")
    ax.legend()
    savefig(fig, "lstm_ae_training_curve_v2.png")

    print("\n=== Model 11: LSTM Autoencoder ===")
    print(json.dumps(summary, indent=2, default=float))
    return score_col, flag_col, applicable, summary


# --------------------------------------------------------- Model 12: Hybrid Ensemble
def model_hybrid_ensemble(flag_if, flag_lof, flag_ae):
    vote_count = flag_if.astype(int) + flag_lof.astype(int) + flag_ae.astype(int)
    flag_majority = (vote_count >= 2).astype(int)
    summary = {
        "components": "Isolation Forest (Model 1, native flag) + LOF (Model 2, native flag) + Autoencoder (Model 9, top-5% reconstruction-MSE flag)",
        "rule": "flagged if >=2 of the 3 component flags fire (majority vote)",
        "vote_distribution": {int(k): int(v) for k, v in pd.Series(vote_count).value_counts().sort_index().items()},
        "majority_flagged_rate": round(float(flag_majority.mean()), 4),
        "pairwise_agreement_if_lof": round(float((flag_if == flag_lof).mean()), 4),
        "pairwise_agreement_if_ae": round(float((flag_if == flag_ae).mean()), 4),
        "pairwise_agreement_lof_ae": round(float((flag_lof == flag_ae).mean()), 4),
    }
    print("\n=== Model 12: Hybrid Ensemble (IF + LOF + AE, majority vote) ===")
    print(json.dumps(summary, indent=2, default=float))
    return vote_count, flag_majority, summary


def main():
    df, feature_cols, X, X_train, X_val, X_all, idx_train, idx_val, scaler = load_common()
    classical = pd.read_csv(os.path.join(ARTIFACTS_V2_DIR, "model_scores_classical.csv"))
    assert (classical["TransactionID"].values == df["TransactionID"].values).all()

    ae_score, ae_flag, ae_summary = model_autoencoder_reused(feature_cols, X_all)
    vae_score, vae_flag, vae_summary = model_vae(feature_cols, X_train, X_val, X_all)
    plot_ae_vs_vae(ae_score, vae_score)
    lstm_score, lstm_flag, lstm_applicable, lstm_summary = model_lstm_autoencoder(df, feature_cols, scaler)

    hybrid_votes, hybrid_flag, hybrid_summary = model_hybrid_ensemble(
        classical["flag_isolation_forest_native"].values,
        classical["flag_lof_native"].values,
        ae_flag,
    )

    all_scores = pd.DataFrame({
        "TransactionID": df["TransactionID"].values,
        "AccountID": df["AccountID"].values,
        "score_isolation_forest": classical["score_isolation_forest"].values,
        "score_lof": classical["score_lof"].values,
        "score_ocsvm": classical["score_ocsvm"].values,
        "score_elliptic_envelope": classical["score_elliptic_envelope"].values,
        "score_dbscan": classical["score_dbscan"].values,
        "score_hdbscan": classical["score_hdbscan"].values,
        "score_kmeans": classical["score_kmeans"].values,
        "score_gmm": classical["score_gmm"].values,
        "score_autoencoder": ae_score,
        "score_vae": vae_score,
        "score_lstm_ae": lstm_score,
        "lstm_ae_applicable": lstm_applicable,
        "hybrid_vote_count": hybrid_votes,
        "flag_isolation_forest": classical["flag_isolation_forest_native"].values,
        "flag_lof": classical["flag_lof_native"].values,
        "flag_ocsvm": classical["flag_ocsvm_native"].values,
        "flag_elliptic_envelope": classical["flag_elliptic_envelope_native"].values,
        "flag_dbscan": classical["flag_dbscan_native"].values,
        "flag_hdbscan": classical["flag_hdbscan_native"].values,
        "flag_kmeans": classical["flag_kmeans_top5pct"].values,
        "flag_gmm": classical["flag_gmm_top5pct"].values,
        "flag_autoencoder": ae_flag,
        "flag_vae": vae_flag,
        "flag_lstm_ae": lstm_flag,
        "flag_hybrid_ensemble": hybrid_flag,
    })
    all_scores.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_scores_all.csv"), index=False)
    print(f"\nSaved: {os.path.join(ARTIFACTS_V2_DIR, 'model_scores_all.csv')}")

    model_names = ["isolation_forest", "lof", "ocsvm", "elliptic_envelope", "dbscan",
                    "hdbscan", "kmeans", "gmm", "autoencoder", "vae", "lstm_ae", "hybrid_ensemble"]
    score_cols = {n: (f"score_{n}" if n != "hybrid_ensemble" else "hybrid_vote_count") for n in model_names}
    flag_cols = {n: f"flag_{n}" for n in model_names}

    n_models = len(model_names)
    spearman_mat = np.full((n_models, n_models), np.nan)
    applicable_mask = (all_scores["lstm_ae_applicable"] == 1).values
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            s1 = all_scores[score_cols[m1]].values
            s2 = all_scores[score_cols[m2]].values
            mask = applicable_mask if (m1 == "lstm_ae" or m2 == "lstm_ae") else np.ones(len(df), dtype=bool)
            rho, _ = spearmanr(s1[mask], s2[mask])
            spearman_mat[i, j] = rho

    spearman_df = pd.DataFrame(spearman_mat, index=model_names, columns=model_names)
    spearman_df.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_pairwise_spearman.csv"))

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(spearman_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Spearman rho"}, vmin=-1, vmax=1)
    ax.set_title("Pairwise Spearman Rank Correlation Between All 12 Models' Anomaly Scores (v2)")
    savefig(fig, "model_pairwise_spearman_heatmap_v2.png")

    jaccard_mat = np.full((n_models, n_models), np.nan)
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            f1 = all_scores[flag_cols[m1]].values.astype(bool)
            f2 = all_scores[flag_cols[m2]].values.astype(bool)
            mask = applicable_mask if (m1 == "lstm_ae" or m2 == "lstm_ae") else np.ones(len(df), dtype=bool)
            f1m, f2m = f1[mask], f2[mask]
            union = (f1m | f2m).sum()
            inter = (f1m & f2m).sum()
            jaccard_mat[i, j] = inter / union if union > 0 else np.nan

    jaccard_df = pd.DataFrame(jaccard_mat, index=model_names, columns=model_names)
    jaccard_df.to_csv(os.path.join(ARTIFACTS_V2_DIR, "model_pairwise_jaccard.csv"))

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": "Jaccard overlap"})
    ax.set_title("Pairwise Jaccard Overlap on Each Model's Flagged Set (v2)")
    savefig(fig, "model_pairwise_jaccard_heatmap_v2.png")

    rates = {n: all_scores[flag_cols[n]].mean() for n in model_names}
    fig, ax = plt.subplots(figsize=(9, 5))
    names_sorted = sorted(rates, key=lambda n: rates[n])
    ax.barh(names_sorted, [rates[n] * 100 for n in names_sorted], color="#2F6690")
    ax.set_xlabel("Flagged rate (%)")
    ax.set_title("Anomaly / Flagged Rate by Model, v2 (native contamination where defined, else top-5%)")
    ax.axvline(5.0, color="#D1495B", ls="--", lw=1, label="5% reference line")
    ax.legend()
    savefig(fig, "model_anomaly_rate_comparison_v2.png")

    with open(os.path.join(ARTIFACTS_V2_DIR, "model_comparison_summary.json"), "w") as f:
        json.dump({
            "anomaly_rates_pct": {n: round(float(rates[n] * 100), 2) for n in model_names},
        }, f, indent=2, default=float)

    print("\n=== Cross-model comparison summary (v2) ===")
    print("Anomaly rates (%):", json.dumps({n: round(rates[n]*100, 2) for n in model_names}, indent=2))
    print(f"\nSaved: model_scores_all.csv, model_pairwise_spearman.csv, model_pairwise_jaccard.csv, "
          f"model_comparison_summary.json (all in {ARTIFACTS_V2_DIR})")


if __name__ == "__main__":
    main()
