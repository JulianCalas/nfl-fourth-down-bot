"""Punt outcomes as a distribution over the next snap, by yard line.

A punt is not a single expected net yardage. Touchbacks pile up on one yard
line, punts downed inside the 10 pile up on another, and a handful come back
the other way. Because win probability is curved, the average of win
probability across those outcomes is not the win probability of the average
outcome. The model keeps the distribution.

Samples are quantiles, not a random draw, so the same data always produces
the same recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class PuntSpot:
    """Outcome mix for punts snapped at one yard line (nflverse yardline_100)."""

    p_opp: float = 1.0
    p_keep: float = 0.0
    p_we_score: float = 0.0
    p_they_score: float = 0.0
    opp_yl: np.ndarray = field(default_factory=lambda: np.array([75.0]))
    keep_yl: np.ndarray = field(default_factory=lambda: np.array([50.0]))
    # Score change on non-touchdown outcomes (a safety is -2). Usually 0.
    opp_delta: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    keep_delta: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    we_score_points: float = 7.0
    they_score_points: float = 7.0
    n: int = 0


@dataclass
class PuntModel:
    spots: dict[int, PuntSpot] = field(default_factory=dict)
    elapsed: float = 8.0
    n_punts: int = 0

    def fit(self, plays: pl.DataFrame) -> "PuntModel":
        punts = plays.filter(
            (pl.col("play_type") == "punt")
            & pl.col("transition_ok")
            & pl.col("yardline_100").is_between(1, 99)
            & pl.col("qtr").is_between(1, 4)
        )
        self.n_punts = int(punts.height)
        if punts.height < 50:
            raise RuntimeError(f"only {punts.height} punts with a usable next snap")
        self.elapsed = float(punts.select(pl.col("elapsed").median()).item())
        frame = punts.select(
            [
                pl.col("yardline_100").round(0).cast(pl.Int64).alias("from_yl"),
                pl.col("kept_ball"),
                pl.col("n_yardline_100"),
                (pl.col("n_score_for_us") - pl.col("score_diff")).alias("delta"),
            ]
        ).drop_nulls()
        from_yl = frame["from_yl"].to_numpy()
        kept = frame["kept_ball"].to_numpy().astype(bool)
        nyl = frame["n_yardline_100"].to_numpy().astype(float)
        delta = frame["delta"].to_numpy().astype(float)
        self.spots = {
            yl: _spot_from_pool(from_yl, kept, nyl, delta, yl) for yl in range(1, 100)
        }
        return self

    def spot(self, yardline_100: float) -> PuntSpot:
        yl = int(np.clip(round(float(yardline_100)), 1, 99))
        if yl not in self.spots:
            raise RuntimeError("punt model is not fit")
        return self.spots[yl]


def _spot_from_pool(
    from_yl: np.ndarray,
    kept: np.ndarray,
    nyl: np.ndarray,
    delta: np.ndarray,
    yl: int,
) -> PuntSpot:
    window = 2
    pool = _window_mask(from_yl, yl, window)
    while int(pool.sum()) < 80 and window < 18:
        window += 2
        pool = _window_mask(from_yl, yl, window)
    if int(pool.sum()) < 30:
        pool = np.ones(len(from_yl), dtype=bool)
    kept_p = kept[pool]
    nyl_p = nyl[pool]
    delta_p = delta[pool]
    n = int(pool.sum())
    they_score = delta_p <= -6
    we_score = delta_p >= 6
    ordinary = ~they_score & ~we_score
    keep = ordinary & kept_p
    opp = ordinary & ~kept_p
    spot = PuntSpot(n=n)
    spot.p_they_score = float(they_score.mean()) if n else 0.0
    spot.p_we_score = float(we_score.mean()) if n else 0.0
    spot.p_keep = float(keep.mean()) if n else 0.0
    spot.p_opp = float(max(0.0, 1.0 - spot.p_they_score - spot.p_we_score - spot.p_keep))
    if they_score.any():
        spot.they_score_points = float(np.median(-delta_p[they_score]))
    if we_score.any():
        spot.we_score_points = float(np.median(delta_p[we_score]))
    # Quantiles are taken on yard line, and the score change that actually
    # happened on that punt comes along with it. Pairing them at random would
    # attach a safety to an ordinary touchback.
    if opp.any():
        spot.opp_yl, spot.opp_delta = _paired_quantiles(nyl_p[opp], delta_p[opp])
    else:
        spot.opp_yl = np.array([_default_opp_yl(yl)])
        spot.opp_delta = np.array([0.0])
    if keep.any():
        spot.keep_yl, spot.keep_delta = _paired_quantiles(nyl_p[keep], delta_p[keep])
    else:
        spot.keep_yl = np.array([float(np.clip(yl - 15, 1, 99))])
        spot.keep_delta = np.array([0.0])
    # Renormalize in case rounding left the ordinary-opponent share short.
    total = spot.p_opp + spot.p_keep + spot.p_we_score + spot.p_they_score
    if total > 0 and abs(total - 1) > 1e-6:
        spot.p_opp /= total
        spot.p_keep /= total
        spot.p_we_score /= total
        spot.p_they_score /= total
    return spot


def _window_mask(from_yl: np.ndarray, yl: int, window: int) -> np.ndarray:
    return (from_yl >= yl - window) & (from_yl <= yl + window)


def _default_opp_yl(from_yl: int) -> float:
    """A boring touchback-or-net guess used only when a yard line has no punts."""
    # Net of ~40 yards, then flip to the opponent's yardline_100.
    traveled = min(40.0, max(10.0, from_yl - 20.0))
    raw = 100.0 - (from_yl - traveled)
    return float(np.clip(raw, 1, 99))


def _paired_quantiles(yardline: np.ndarray, delta: np.ndarray, m: int = 21) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(yardline, dtype=float))
    yardline = np.asarray(yardline, dtype=float)[order]
    delta = np.asarray(delta, dtype=float)[order]
    idx = np.linspace(0, len(yardline) - 1, min(m, len(yardline))).astype(int)
    return yardline[idx], delta[idx]
