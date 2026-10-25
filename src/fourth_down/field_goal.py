"""Field-goal make probability by distance, then a small weather adjustment.

Distance does almost all of the work, and it should only go one direction:
a longer kick is never more likely than a shorter one. Isotonic regression
enforces that. It is a step function that best matches the observed make
rate without ever stepping up as distance grows.

Weather, roof, altitude, and year are a second stage. They enter as a
binomial GLM whose offset is the log-odds of the distance curve, and whose
features are centered on the training means. At average conditions the
adjustment is zero and the isotonic curve is returned unchanged. A calm
outdoor kick in Denver is allowed to move off that average.

Blocked kicks count as misses for the make probability. The miss model in
`decide.py` still gives blocks their own field position, via the residual
distribution stored here, so a block returned for a touchdown is rare in
the average rather than impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import statsmodels.api as sm

from fourth_down.football import expit, logit, missed_fg_opponent_yardline


WEATHER_NAMES = ("indoor", "wind_mph", "cold", "altitude", "years_since_2014")


@dataclass
class FieldGoalModel:
    iso_x: np.ndarray | None = None
    iso_y: np.ndarray | None = None
    distance_min: float = 18.0
    distance_max: float = 62.0
    tail_slope: float = -0.12
    centers: dict[str, float] = field(default_factory=dict)
    weather_coef: dict[str, float] = field(default_factory=dict)
    weather_pvalue: dict[str, float] = field(default_factory=dict)
    weather_summary: str = ""
    miss_residual: np.ndarray = field(default_factory=lambda: np.zeros(1))
    n_kicks: int = 0
    n_made: int = 0

    def fit(self, plays: pl.DataFrame) -> "FieldGoalModel":
        kicks = _kick_frame(plays)
        if kicks.height < 200:
            raise RuntimeError(f"only {kicks.height} field goals in the training window")
        distance = kicks["kick_distance"].to_numpy().astype(float)
        made = (kicks["field_goal_result"] == "made").to_numpy().astype(float)
        order = np.argsort(distance)
        distance = distance[order]
        made = made[order]

        # Pool to integer yards so a single 64-yard make cannot create a
        # one-play step. Isotonic is then fit on one row per attempt, which
        # weights distances by how often they are attempted.
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(distance, made)
        grid = np.arange(np.floor(distance.min()), np.ceil(distance.max()) + 1)
        curve = iso.predict(grid)
        self.iso_x = grid.astype(float)
        self.iso_y = curve.astype(float)
        self.distance_min = float(grid.min())
        self.distance_max = float(grid.max())
        self.tail_slope = _tail_slope(self, lookback=8.0)
        self.n_kicks = int(len(made))
        self.n_made = int(made.sum())

        p0 = np.clip(self.distance_probability(distance), 1e-3, 1 - 1e-3)
        weather = _weather_matrix(kicks, centers=None)
        self.centers = weather["centers"]
        x = weather["x"]
        try:
            glm = sm.GLM(
                made,
                x,
                family=sm.families.Binomial(),
                offset=logit(p0),
            ).fit(disp=0, maxiter=100)
            coef = np.asarray(glm.params, dtype=float)
            pvalues = np.asarray(glm.pvalues, dtype=float)
            if not np.all(np.isfinite(coef)):
                raise RuntimeError("weather coefficients were not finite")
            self.weather_coef = {name: float(coef[i]) for i, name in enumerate(WEATHER_NAMES)}
            self.weather_pvalue = {name: float(pvalues[i]) for i, name in enumerate(WEATHER_NAMES)}
            try:
                self.weather_summary = glm.summary(xname=list(WEATHER_NAMES)).as_text()
            except Exception as exc:  # noqa: BLE001
                self.weather_summary = f"Weather coefficients saved. Summary text failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - distance curve still works
            self.weather_coef = {name: 0.0 for name in WEATHER_NAMES}
            self.weather_pvalue = {name: 1.0 for name in WEATHER_NAMES}
            self.weather_summary = f"Weather GLM failed ({exc}). Distance curve only."

        self.miss_residual = _miss_residuals(kicks)
        return self

    def distance_probability(self, distance: np.ndarray | float) -> np.ndarray:
        """Make probability from distance alone, decreasing by construction."""
        d = np.asarray(distance, dtype=float)
        scalar = np.isscalar(distance)
        d = np.atleast_1d(d)
        if self.iso_x is None or self.iso_y is None:
            raise RuntimeError("field-goal model is not fit")
        inside = np.interp(d, self.iso_x, self.iso_y)
        # Past the longest kick in the training window, extend the logit
        # slope from the last several yards instead of freezing the rate.
        # A 70-yarder should not inherit the make rate of a 62-yarder.
        far = d > self.distance_max
        if np.any(far):
            p_edge = float(np.clip(self.iso_y[-1], 1e-3, 1 - 1e-3))
            extra = d[far] - self.distance_max
            inside[far] = expit(logit(p_edge) + self.tail_slope * extra)
        inside = np.clip(inside, 0.005, 0.995)
        if scalar:
            return float(inside[0])
        return inside

    def predict(
        self,
        distance: np.ndarray,
        indoor: np.ndarray,
        wind_mph: np.ndarray,
        cold: np.ndarray,
        altitude: np.ndarray,
        season: np.ndarray,
    ) -> np.ndarray:
        """Make probability for each kick, including conditions."""
        d = np.asarray(distance, dtype=float)
        p0 = self.distance_probability(d)
        if not self.weather_coef:
            return p0
        years = np.asarray(season, dtype=float) - 2014.0
        raw = {
            "indoor": np.asarray(indoor, dtype=float),
            "wind_mph": np.asarray(wind_mph, dtype=float),
            "cold": np.asarray(cold, dtype=float),
            "altitude": np.asarray(altitude, dtype=float),
            "years_since_2014": years,
        }
        adjustment = np.zeros(len(d), dtype=float)
        for name in WEATHER_NAMES:
            centered = raw[name] - self.centers.get(name, 0.0)
            adjustment = adjustment + self.weather_coef.get(name, 0.0) * centered
        return np.clip(expit(logit(p0) + adjustment), 0.005, 0.995)

    def curve_table(self, season: int = 2025) -> list[dict]:
        """Distance curve at average conditions and at a calm outdoor kick."""
        rows = []
        for dist in range(18, 69):
            rows.append(
                {
                    "distance": dist,
                    "league_mix": float(self.distance_probability(dist)),
                    "calm_outdoor": float(
                        self.predict(
                            np.array([dist]),
                            np.array([0.0]),
                            np.array([0.0]),
                            np.array([0.0]),
                            np.array([0.0]),
                            np.array([season]),
                        )[0]
                    ),
                }
            )
        return rows


def _tail_slope(model: FieldGoalModel, lookback: float) -> float:
    anchor = max(model.distance_min, model.distance_max - lookback)
    p_far = float(np.clip(np.interp(model.distance_max, model.iso_x, model.iso_y), 1e-3, 1 - 1e-3))
    p_near = float(np.clip(np.interp(anchor, model.iso_x, model.iso_y), 1e-3, 1 - 1e-3))
    span = model.distance_max - anchor
    if span <= 0:
        return -0.12
    slope = (logit(p_far) - logit(p_near)) / span
    if slope >= -0.01:
        return -0.12
    return float(slope)


def _kick_frame(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.filter(
        (pl.col("play_type") == "field_goal")
        & pl.col("field_goal_result").is_in(["made", "missed", "blocked"])
        & pl.col("kick_distance").is_between(18, 76)
        & pl.col("qtr").is_between(1, 4)
    )


def _weather_matrix(kicks: pl.DataFrame, centers: dict | None) -> dict:
    years = kicks["season"].to_numpy().astype(float) - 2014.0
    columns = {
        "indoor": kicks["indoor"].to_numpy().astype(float),
        "wind_mph": kicks["wind_mph"].to_numpy().astype(float),
        "cold": kicks["cold"].to_numpy().astype(float),
        "altitude": kicks["altitude"].to_numpy().astype(float),
        "years_since_2014": years,
    }
    if centers is None:
        centers = {name: float(np.nanmean(values)) for name, values in columns.items()}
    x = np.column_stack([columns[name] - centers[name] for name in WEATHER_NAMES])
    return {"x": x, "centers": centers}


def _miss_residuals(kicks: pl.DataFrame, m: int = 17) -> np.ndarray:
    """How many yards the actual miss spot differs from the rule-of-thumb spot.

    Most misses land on the formula. The left tail is blocked kicks that
    were returned. Quantiles keep that tail without letting one return
    dominate the average.
    """
    misses = kicks.filter(
        pl.col("field_goal_result").is_in(["missed", "blocked"])
        & pl.col("transition_ok")
        & ~pl.col("kept_ball")
    )
    if misses.height < 20:
        return np.zeros(m)
    formula = missed_fg_opponent_yardline(misses["yardline_100"].to_numpy())
    residual = misses["n_yardline_100"].to_numpy().astype(float) - np.asarray(formula, dtype=float)
    residual = residual[np.isfinite(residual)]
    residual = residual[(residual > -40) & (residual < 25)]
    if len(residual) < 10:
        return np.zeros(m)
    residual = np.sort(residual)
    idx = np.linspace(0, len(residual) - 1, m).astype(int)
    return residual[idx]
