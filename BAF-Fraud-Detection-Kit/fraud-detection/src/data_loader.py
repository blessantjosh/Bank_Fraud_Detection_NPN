"""
data_loader.py -- read the raw BAF csv and perform the stratified random split.

VERIFIED split protocol (see README.md / 01-DATASET-BIBLE.md "Trap 2"):
the organiser split for this dataset's leaderboard is a RANDOM stratified
70/30 split, NOT the NeurIPS paper's temporal (month 0-5 / 6-7) protocol.
`month` therefore stays IN the feature set as an ordinary column, and we
split with sklearn's stratified train_test_split rather than a month cutoff.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config, resolve_path

logger = logging.getLogger("fraud_detection.data_loader")


def load_raw(cfg: Config) -> pd.DataFrame:
    """Load the raw BAF csv, dropping any stray index column some exports carry."""
    path = resolve_path(cfg.data.raw_path)
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)
    junk = [c for c in df.columns if c.lower().startswith("unnamed")]
    if junk:
        logger.info("Dropping junk index columns: %s", junk)
        df = df.drop(columns=junk)
    logger.info("Loaded %d rows x %d columns", df.shape[0], df.shape[1])
    return df


def stratified_split(df: pd.DataFrame, cfg: Config):
    """
    70/15/15 stratified random train/val/test split, mirroring the verified
    organiser protocol (random, not temporal). `month` is kept as a feature.
    """
    target = cfg.data.target_col
    seed = cfg.seed
    train_size = cfg.split.train_size
    val_size = cfg.split.val_size
    test_size = cfg.split.test_size
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9, "split sizes must sum to 1"

    train_df, rest_df = train_test_split(
        df, train_size=train_size, stratify=df[target], random_state=seed
    )
    rel_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        rest_df, train_size=rel_val, stratify=rest_df[target], random_state=seed
    )

    logger.info(
        "Stratified random split -> train=%d (%.4f%% fraud) | val=%d (%.4f%% fraud) | "
        "test=%d (%.4f%% fraud)",
        len(train_df), 100 * train_df[target].mean(),
        len(val_df), 100 * val_df[target].mean(),
        len(test_df), 100 * test_df[target].mean(),
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_and_split(cfg: Config, cache: bool = True):
    """
    Load raw data and split it, caching the three splits as parquet in
    data/processed/ so repeated runs (training, notebooks, tests) don't
    re-read and re-split the full 1M-row csv every time.
    """
    processed_dir = resolve_path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_p = processed_dir / "train_raw.parquet"
    val_p = processed_dir / "val_raw.parquet"
    test_p = processed_dir / "test_raw.parquet"

    if cache and train_p.exists() and val_p.exists() and test_p.exists():
        logger.info("Loading cached raw splits from %s", processed_dir)
        return (
            pd.read_parquet(train_p),
            pd.read_parquet(val_p),
            pd.read_parquet(test_p),
        )

    df = load_raw(cfg)
    train_df, val_df, test_df = stratified_split(df, cfg)

    if cache:
        train_df.to_parquet(train_p, index=False)
        val_df.to_parquet(val_p, index=False)
        test_df.to_parquet(test_p, index=False)
        logger.info("Cached raw splits to %s", processed_dir)

    return train_df, val_df, test_df
