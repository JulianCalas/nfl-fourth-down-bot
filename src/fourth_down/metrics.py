"""Scoring rules for probabilities.

Log loss and the Brier score are proper scoring rules: a model that reports
its true probabilities scores better, in expectation, than a model that
hedges or exaggerates. Accuracy (did we call win or loss?) is a poor fit for
win probability, because a 60% favorite and a 90% favorite can both "win" the
classification and still be very different forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    """Mean log loss. Lower is better. A coin flip scores ln(2) ≈ 0.693."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p must have the same shape")
    if len(y_arr) == 0:
        raise ValueError("log loss needs at least one row")
    return float(-np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of a probability. Lower is better. A coin flip scores 0.25."""
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if len(y_arr) == 0:
        raise ValueError("Brier score needs at least one row")
    return float(np.mean((p_arr - y_arr) ** 2))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability versus the observed win rate.

    Equal count (rather than equal width) keeps the extreme bins from being
    almost empty. A calibrated model has `actual_rate` close to `pred_mean`
    in every bin.
    """
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if len(y_arr) < n_bins:
        raise ValueError("need at least one row per bin")
    order = np.argsort(p_arr)
    y_arr = y_arr[order]
    p_arr = p_arr[order]
    chunks = np.array_split(np.arange(len(y_arr)), n_bins)
    rows = []
    for i, idx in enumerate(chunks, start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "pred_mean": float(p_arr[idx].mean()),
                "actual_rate": float(y_arr[idx].mean()),
                "pred_min": float(p_arr[idx].min()),
                "pred_max": float(p_arr[idx].max()),
            }
        )
    return pd.DataFrame(rows)


def mean_absolute_calibration_error(table: pd.DataFrame) -> float:
    """Average gap between predicted and actual rate, weighted by bin size."""
    w = table["n"].to_numpy(dtype=float)
    gap = np.abs(table["pred_mean"].to_numpy(dtype=float) - table["actual_rate"].to_numpy(dtype=float))
    return float(np.sum(w * gap) / np.sum(w))


def season_folds(seasons: list[int], first_test: int) -> list[dict]:
    """Expanding-window folds that never put the test season in training.

    For test season T the model is trained on seasons through T-2 and
    calibrated on T-1. Splitting by season, instead of by random plays,
    stops the same game from appearing on both sides of the score.
    """
    seasons = sorted(int(s) for s in seasons)
    folds = []
    for test in seasons:
        if test < first_test:
            continue
        calibrate = test - 1
        train = [s for s in seasons if s <= test - 2]
        if calibrate not in seasons or len(train) < 2:
            raise ValueError(f"not enough history to test season {test}")
        folds.append({"test": test, "calibrate": calibrate, "train": train})
    if not folds:
        raise ValueError("no test folds were produced")
    return folds
