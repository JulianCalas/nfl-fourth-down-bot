"""Build the small situation tables the chart and the app both need."""

from __future__ import annotations

import polars as pl

from fourth_down.config import LAST_FULL_SEASON

DEFAULTS = {
    "timeouts_off": 3.0,
    "timeouts_def": 3.0,
    "is_home": 1.0,
    "expected_margin": 0.0,
    "receive_2h": 1.0,
    "spread_missing": 0.0,
    "indoor": 0.0,
    "wind_mph": 0.0,
    "cold": 0.0,
    "altitude": 0.0,
    "season": float(LAST_FULL_SEASON),
    "score_diff": 0.0,
    "game_seconds_remaining": 3600.0,
    "yardline_100": 60.0,
    "ydstogo": 1.0,
}


def make_situations(records: list[dict]) -> pl.DataFrame:
    """One row per hypothetical fourth down. Missing keys take neutral defaults."""
    rows = []
    for record in records:
        row = dict(DEFAULTS)
        row.update(record)
        rows.append(row)
    frame = pl.DataFrame(rows)
    # Keep numeric columns as floats so a one-row app request matches a 990-row chart.
    casts = []
    for name, value in DEFAULTS.items():
        if isinstance(value, float) and name in frame.columns:
            casts.append(pl.col(name).cast(pl.Float64))
    return frame.with_columns(casts)


def decision_grid(scenario: dict, yards_to_go: range = range(1, 11)) -> pl.DataFrame:
    """Every yard line and distance-to-go for one fixed game script."""
    records = []
    for yardline in range(1, 100):
        for ytg in yards_to_go:
            records.append({**scenario, "yardline_100": float(yardline), "ydstogo": float(ytg)})
    return make_situations(records)
