"""Figures for the README and the notebook.

Colors are consistent across charts: green is go, navy is punt, gold is a
field goal. The decision chart uses the same three.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

GO = "#2E7D4F"
PUNT = "#1F4E79"
FG = "#C48A1A"
INK = "#1C1917"
MUTED = "#57534E"
OURS = "#9A3412"
PANEL = "#FBF8F3"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": PANEL,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.titlecolor": INK,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "regular",
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    }
)


def plot_decision_chart(grid: pl.DataFrame, path: Path, title: str, subtitle: str) -> None:
    """Yard line by yards to go, colored by the recommended choice."""
    ytg_vals = list(range(1, 11))
    # Column 0 is our own 1-yard line (yardline_100 = 99). Column 98 is the
    # opponent's 1 (yardline_100 = 1). Left to right is a drive toward the goal.
    calls = np.full((len(ytg_vals), 99), np.nan)
    edges = np.full((len(ytg_vals), 99), np.nan)
    code = {"go": 0.0, "fg": 1.0, "punt": 2.0}
    for row in grid.iter_rows(named=True):
        ytg = int(row["ydstogo"])
        yl = int(round(row["yardline_100"]))
        if ytg not in ytg_vals or not 1 <= yl <= 99:
            continue
        calls[ytg - 1, 99 - yl] = code[row["recommended"]]
        edges[ytg - 1, 99 - yl] = row["wp_edge"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    cmap = ListedColormap([GO, FG, PUNT])
    axes[0].imshow(calls, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    axes[0].set_title(title)
    _field_axis(axes[0])
    # Above the axes, so the legend does not cover the goal line.
    fig.legend(
        handles=[
            Patch(facecolor=GO, edgecolor="none", label="Go for it"),
            Patch(facecolor=FG, edgecolor="none", label="Field goal"),
            Patch(facecolor=PUNT, edgecolor="none", label="Punt"),
        ],
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.42, 1.02),
    )

    image = axes[1].imshow(edges, origin="lower", aspect="auto", cmap="YlOrBr", vmin=0, vmax=max(0.08, np.nanmax(edges)))
    axes[1].set_title("Edge of the best choice over the runner-up")
    _field_axis(axes[1])
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Win probability")
    fig.suptitle(subtitle, fontsize=10, color=MUTED, y=0.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(table: pl.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    pred = table["pred_mean"].to_numpy()
    actual = table["actual_rate"].to_numpy()
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--", label="Perfect calibration")
    ax.plot(pred, actual, color=OURS, marker="o", lw=2, label="This model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_logloss(metrics: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    season = metrics["season"].to_numpy()
    ax.plot(season, metrics["logloss_ours"].to_numpy(), color=OURS, marker="o", lw=2, label="This model")
    ax.plot(season, metrics["logloss_nflfastr"].to_numpy(), color=PUNT, marker="o", lw=2, label="nflfastR wp")
    ax.plot(season, metrics["logloss_vegas"].to_numpy(), color=FG, marker="o", lw=2, label="nflfastR vegas_wp")
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_xlabel("Held-out season")
    ax.set_title("Win probability on seasons the model did not train on")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_vs_nflfastr(sample: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(
        sample["nflfastr_wp"].to_numpy(),
        sample["our_wp"].to_numpy(),
        s=8,
        alpha=0.25,
        color=PUNT,
        linewidths=0,
    )
    ax.plot([0, 1], [0, 1], color=OURS, lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nflfastR wp")
    ax.set_ylabel("This model's win probability")
    ax.set_title("Same plays, two win-probability models")
    fig.savefig(path)
    plt.close(fig)


def plot_fg_curve(curve: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(curve["distance"], curve["league_mix"], color=FG, lw=2.4, label="Distance curve")
    if "actual_rate" in curve.columns:
        observed = curve.filter(pl.col("n") >= 20)
        ax.scatter(
            observed["distance"].to_numpy(),
            observed["actual_rate"].to_numpy(),
            s=np.clip(observed["n"].to_numpy() / 8, 12, 80),
            color=INK,
            alpha=0.75,
            label="Actual make rate (dot size = attempts)",
        )
    ax.set_xlabel("Kick distance (yards)")
    ax.set_ylabel("Make probability")
    ax.set_ylim(0, 1)
    ax.set_title("Field goals get less likely as the kick gets longer")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_conversion(table: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for yl, label, color in (
        (25, "Opponent's 25", GO),
        (50, "Midfield", FG),
        (75, "Own 25", PUNT),
    ):
        sl = table.filter(pl.col("yardline_100") == yl).sort("ydstogo")
        ax.plot(sl["ydstogo"], sl["p_success"], color=color, lw=2.2, label=label)
    ax.set_xlabel("Yards to go")
    ax.set_ylabel("P(first down or touchdown)")
    ax.set_ylim(0, 1)
    ax.set_title("Fourth-down conversion probability")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_trend(trend: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        trend["season"].to_numpy(),
        trend["go_rate_when_should"].to_numpy(),
        color=GO,
        marker="o",
        lw=2.2,
        label="Go rate when the model says go",
    )
    ax.plot(
        trend["season"].to_numpy(),
        trend["go_rate_fourth_and_one"].to_numpy(),
        color=PUNT,
        marker="o",
        lw=2.2,
        label="Go rate on 4th and 1 between the 40s",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of plays that went for it")
    ax.set_xlabel("Season")
    ax.set_title("Are coaches going for it more often?")
    ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_bars(frame: pl.DataFrame, path: Path, label_col: str, value_col: str, title: str, xlabel: str) -> None:
    """Horizontal bars. The frame should already be in display order, top row at the top."""
    if frame.height == 0:
        return
    labels = frame[label_col].to_list()
    values = frame[value_col].to_numpy() * 100  # percentage points of win probability
    fig_h = max(4.2, 0.38 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=OURS, height=0.72)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def _field_axis(ax) -> None:
    ax.set_yticks(range(0, 10), [str(i) for i in range(1, 11)])
    ax.set_ylabel("Yards to go")
    # Tick marks at our 10, 20, 30, 40, midfield, their 40, 30, 20, 10.
    # x=0 is yardline_100 99 (our 1). x position of a yardline y is 99 - y.
    tick_yl = [90, 80, 70, 60, 50, 40, 30, 20, 10]
    tick_x = [99 - yl for yl in tick_yl]
    tick_labels = ["Own 10", "20", "30", "40", "50", "40", "30", "20", "Opp 10"]
    ax.set_xticks(tick_x, tick_labels, rotation=0)
    ax.set_xlabel("Line of scrimmage")
