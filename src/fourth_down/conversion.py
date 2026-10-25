"""How often a fourth-down attempt converts, and what the next snap looks like.

Two binomial GLMs, not a black-box classifier:

* P(success): a first down or an offensive touchdown
* P(touchdown): scored on the play

Both are logistic regressions on yards to go, field position, and a goal-to-go
flag. The coefficients are the thing to read in an interview. Yards to go are
scaled by 10 and field position by 100 so a coefficient is a change in
log-odds for a meaningful chunk of the field, not for a single yard squared.

Given a failure, the split into turnover on downs, a giveaway, a defensive
touchdown, and a safety is a set of rates, not another regression. Those
events are too rare for a stable coefficient on field position. The rates
are estimated on the same prior seasons as the GLMs.

The yard line *after* the play comes from the empirical distribution of
past plays of that type. Win probability is curved in field position, so
the decision uses a spread of outcomes (a stuff, a gain to the sticks, a
breakaway) rather than only the average gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
import statsmodels.api as sm

from fourth_down.football import expit

SUCCESS_NAMES = (
    "intercept",
    "ydstogo_per_10",
    "ydstogo_per_10_sq",
    "yardline_per_100",
    "yardline_per_100_sq",
    "goal_to_go",
    "ydstogo_x_goal",
    "years_centered",
)
POINT_VALUES = np.array([6.0, 7.0, 8.0])
BUCKETS = (1, 2, 3, 4, 7, 11)


@dataclass
class TransitionBook:
    """Empirical 'what happens next' pieces that sit beside the GLMs."""

    elapsed_go: float = 6.0
    elapsed_score: float = 5.0
    elapsed_fg: float = 5.0
    elapsed_punt: float = 8.0
    td_point_weights: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.88, 0.04]))
    safety_opp_yl: float = 60.0
    fd_extra: dict[int, np.ndarray] = field(default_factory=dict)
    downs_shift: np.ndarray = field(default_factory=lambda: np.zeros(21))
    giveaway_shift: np.ndarray = field(default_factory=lambda: np.zeros(21))
    # Share of *failures* (not of all attempts) that land in each bucket.
    fail_downs: float = 0.90
    fail_giveaway: float = 0.08
    fail_def_td: float = 0.015
    fail_safety: float = 0.005
    n_go: int = 0
    n_success: int = 0
    n_td: int = 0


@dataclass
class ConversionModel:
    success_coef: np.ndarray | None = None
    success_pvalue: dict[str, float] = field(default_factory=dict)
    success_summary: str = ""
    td_coef: np.ndarray | None = None
    td_pvalue: dict[str, float] = field(default_factory=dict)
    td_summary: str = ""
    year_center: float = 2018.0
    backend: str = "statsmodels"
    transitions: TransitionBook = field(default_factory=TransitionBook)

    def fit(self, plays: pl.DataFrame) -> "ConversionModel":
        goes = _go_frame(plays)
        if goes.height < 300:
            raise RuntimeError(f"only {goes.height} fourth-down attempts to fit")
        ytg = goes["ydstogo"].to_numpy().astype(float)
        yl = goes["yardline_100"].to_numpy().astype(float)
        season = goes["season"].to_numpy().astype(float)
        self.year_center = float(season.mean())
        goal = (yl <= ytg).astype(float)
        success = _success_flag(goes)
        td = _td_flag(goes)
        x = _design(ytg, yl, goal, season - self.year_center)

        self.success_coef, self.success_pvalue, self.success_summary, ok_s = _fit_binomial(
            success, x, "success"
        )
        self.td_coef, self.td_pvalue, self.td_summary, ok_t = _fit_binomial(td, x, "touchdown")
        self.backend = "statsmodels" if (ok_s and ok_t) else "fallback"
        self.transitions = _fit_transitions(plays, goes)
        self.transitions.n_go = int(goes.height)
        self.transitions.n_success = int(success.sum())
        self.transitions.n_td = int(td.sum())
        return self

    def predict(self, ydstogo: np.ndarray, yardline_100: np.ndarray, season: np.ndarray) -> dict[str, np.ndarray]:
        """Class probabilities for each fourth-down attempt. They sum to 1."""
        ytg = np.clip(np.asarray(ydstogo, dtype=float), 1.0, 20.0)
        yl = np.clip(np.asarray(yardline_100, dtype=float), 1.0, 99.0)
        years = np.asarray(season, dtype=float) - self.year_center
        goal = (yl <= ytg).astype(float)
        x = _design(ytg, yl, goal, years)
        p_success = np.clip(expit(x @ self.success_coef), 0.01, 0.97)
        p_td = np.clip(expit(x @ self.td_coef), 0.0, 1.0)
        # A touchdown is a kind of success. The two models are fit separately,
        # so clip rather than letting P(TD) exceed P(success).
        p_td = np.minimum(p_td, p_success - 0.005)
        p_fd = p_success - p_td
        p_fail = 1.0 - p_success
        book = self.transitions
        return {
            "p_td": p_td,
            "p_fd": p_fd,
            "p_downs": p_fail * book.fail_downs,
            "p_giveaway": p_fail * book.fail_giveaway,
            "p_def_td": p_fail * book.fail_def_td,
            "p_safety": p_fail * book.fail_safety,
        }

    def probability_table(self, season: int = 2025) -> list[dict]:
        """Success and TD rates on a small grid, for the notebook and a figure."""
        rows = []
        for ytg in range(1, 16):
            for yl in (5, 25, 50, 75, 95):
                pred = self.predict(np.array([ytg]), np.array([yl]), np.array([season]))
                rows.append(
                    {
                        "ydstogo": ytg,
                        "yardline_100": yl,
                        "p_success": float(pred["p_td"][0] + pred["p_fd"][0]),
                        "p_td": float(pred["p_td"][0]),
                    }
                )
        return rows


def _design(ytg: np.ndarray, yl: np.ndarray, goal: np.ndarray, years_c: np.ndarray) -> np.ndarray:
    ytg_s = np.clip(ytg, 1.0, 20.0) / 10.0
    yl_s = np.clip(yl, 1.0, 99.0) / 100.0
    return np.column_stack(
        [
            np.ones(len(ytg_s)),
            ytg_s,
            ytg_s**2,
            yl_s,
            yl_s**2,
            goal,
            ytg_s * goal,
            years_c,
        ]
    )


def _fit_binomial(y: np.ndarray, x: np.ndarray, label: str) -> tuple[np.ndarray, dict, str, bool]:
    """Fit a binomial GLM. Fall back to L2 logistic if the MLE is unstable."""
    from sklearn.linear_model import LogisticRegression

    try:
        glm = sm.GLM(y, x, family=sm.families.Binomial()).fit(disp=0, maxiter=100)
        coef = np.asarray(glm.params, dtype=float)
        pvalues = np.asarray(glm.pvalues, dtype=float)
        try:
            summary = glm.summary(xname=list(SUCCESS_NAMES)).as_text()
        except Exception as exc:  # noqa: BLE001
            summary = f"Coefficients saved. Summary text failed: {exc}"
        stable = bool(np.all(np.isfinite(coef)) and np.all(np.abs(coef) < 25))
        if stable:
            named = {name: float(pvalues[i]) for i, name in enumerate(SUCCESS_NAMES)}
            return coef, named, summary, True
        summary = summary + f"\n\nCoefficients for {label} were unstable; L2 logistic used instead."
    except Exception as exc:  # noqa: BLE001
        summary = f"{label} GLM failed ({exc}). L2 logistic used instead."
        pvalues = np.ones(x.shape[1])

    fallback = LogisticRegression(C=1.0, max_iter=400)
    fallback.fit(x[:, 1:], y.astype(int))
    coef = np.concatenate([fallback.intercept_, fallback.coef_.ravel()])
    named = {name: float(pvalues[i]) for i, name in enumerate(SUCCESS_NAMES)}
    return coef, named, summary, False


def _go_frame(plays: pl.DataFrame) -> pl.DataFrame:
    # `choice == "go"` is every pass or run. The conversion model is only
    # about fourth down. First-down "success" is a different question.
    return plays.filter(
        (pl.col("down") == 4)
        & (pl.col("choice") == "go")
        & pl.col("qtr").is_between(1, 4)
        & pl.col("ydstogo").is_between(1, 25)
        & pl.col("yardline_100").is_between(1, 99)
    )


def _td_flag(goes: pl.DataFrame) -> np.ndarray:
    return (
        (goes["touchdown"].fill_null(0).to_numpy() == 1)
        & (goes["td_team"].to_numpy() == goes["posteam"].to_numpy())
    ).astype(float)


def _success_flag(goes: pl.DataFrame) -> np.ndarray:
    converted = goes["fourth_down_converted"].fill_null(0).to_numpy() == 1
    return np.maximum(converted.astype(float), _td_flag(goes))


def _fit_transitions(plays: pl.DataFrame, goes: pl.DataFrame) -> TransitionBook:
    book = TransitionBook()
    book.elapsed_go = _median_elapsed(goes.filter(~_td_expr()))
    book.elapsed_score = _median_elapsed(
        plays.filter(
            (
                ((pl.col("touchdown") == 1) & (pl.col("td_team") == pl.col("posteam")))
                | ((pl.col("play_type") == "field_goal") & (pl.col("field_goal_result") == "made"))
            )
            & pl.col("transition_ok")
        )
    )
    book.elapsed_fg = _median_elapsed(plays.filter(pl.col("play_type") == "field_goal"))
    book.elapsed_punt = _median_elapsed(plays.filter(pl.col("play_type") == "punt"))
    book.td_point_weights = _td_point_weights(plays)
    book.safety_opp_yl = _safety_yardline(goes)
    book.fd_extra = _fd_extras(goes)
    book.downs_shift = _shift_samples(goes, kind="downs")
    book.giveaway_shift = _shift_samples(goes, kind="giveaway")
    _set_fail_rates(book, goes)
    return book


def _td_expr() -> pl.Expr:
    return (pl.col("touchdown") == 1) & (pl.col("td_team") == pl.col("posteam"))


def _median_elapsed(frame: pl.DataFrame, default: float = 6.0) -> float:
    ok = frame.filter(pl.col("transition_ok") & pl.col("elapsed").is_not_null())
    if ok.height < 30:
        return default
    return float(ok.select(pl.col("elapsed").median()).item())


def _td_point_weights(plays: pl.DataFrame) -> np.ndarray:
    """How many points an offensive TD is worth by the next snap, including the try.

    Estimated from every offensive touchdown, not only fourth downs, because
    the extra point does not care which down the score came on. Fourth-down
    samples alone are small and tilted toward late-game two-point tries.
    """
    tds = plays.filter(
        (pl.col("touchdown") == 1)
        & (pl.col("td_team") == pl.col("posteam"))
        & pl.col("transition_ok")
        & pl.col("n_score_for_us").is_not_null()
        & pl.col("score_diff").is_not_null()
    )
    if tds.height < 50:
        return np.array([0.08, 0.88, 0.04])
    delta = (tds["n_score_for_us"] - tds["score_diff"]).to_numpy()
    weights = np.array([(delta == points).mean() for points in POINT_VALUES], dtype=float)
    if weights.sum() <= 0:
        return np.array([0.08, 0.88, 0.04])
    # Renormalize over {6, 7, 8}. Other deltas (a defensive score on the
    # try, a bad join) are dropped rather than forced into a bucket.
    return weights / weights.sum()


def _safety_yardline(goes: pl.DataFrame) -> float:
    safeties = goes.filter(
        (pl.col("safety") == 1) & pl.col("transition_ok") & ~pl.col("kept_ball")
    )
    if safeties.height < 8:
        return 60.0
    return float(safeties.select(pl.col("n_yardline_100").median()).item())


def _bucket_id(ydstogo: float) -> int:
    ytg = int(ydstogo)
    if ytg <= 1:
        return 1
    if ytg == 2:
        return 2
    if ytg == 3:
        return 3
    if ytg <= 6:
        return 4
    if ytg <= 10:
        return 7
    return 11


def _fd_extras(goes: pl.DataFrame, m: int = 21) -> dict[int, np.ndarray]:
    """Yards gained past the line to gain on conversions that were not touchdowns.

    Positive means the offense got more than it needed. The decision samples
    these residuals so a breakaway and a fall-forward are both possible.
    """
    fd = goes.filter(
        (pl.col("fourth_down_converted") == 1)
        & ~_td_expr()
        & pl.col("transition_ok")
        & pl.col("kept_ball")
    ).with_columns(
        (pl.col("yardline_100") - pl.col("n_yardline_100") - pl.col("ydstogo")).alias("extra")
    )
    fd = fd.filter(pl.col("extra").is_between(-3, 40))
    if fd.height == 0:
        global_extra = np.full(m, 2.0)
        return {bucket: global_extra.copy() for bucket in BUCKETS}
    extra = fd["extra"].to_numpy().astype(float)
    buckets = np.array([_bucket_id(value) for value in fd["ydstogo"].to_numpy()], dtype=int)
    global_extra = _quantiles(extra, m)
    out = {bucket: global_extra.copy() for bucket in BUCKETS}
    for bucket in BUCKETS:
        sample = extra[buckets == bucket]
        if len(sample) >= 25:
            out[bucket] = _quantiles(sample, m)
    return out


def _shift_samples(goes: pl.DataFrame, kind: str, m: int = 21) -> np.ndarray:
    """Opponent yardline minus the line-of-scrimmage flip.

    Zero means they took over exactly where the offense was standing.
    Negative means they took over closer to a touchdown (a sack, a completed
    pass that came up short, a returned interception).
    """
    base = goes.filter(pl.col("transition_ok") & ~pl.col("kept_ball") & ~_td_expr())
    if kind == "giveaway":
        base = base.filter((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
    else:
        base = base.filter(
            (pl.col("interception").fill_null(0) != 1)
            & (pl.col("fumble_lost").fill_null(0) != 1)
            & (pl.col("safety").fill_null(0) != 1)
            & ~((pl.col("touchdown") == 1) & (pl.col("td_team") == pl.col("defteam")))
        )
    if base.height < 20:
        return np.zeros(m)
    shift = base["n_yardline_100"].to_numpy() - (100.0 - base["yardline_100"].to_numpy())
    shift = shift[np.isfinite(shift)]
    shift = shift[(shift > -30) & (shift < 30)]
    if len(shift) < 10:
        return np.zeros(m)
    return _quantiles(shift, m)


def _set_fail_rates(book: TransitionBook, goes: pl.DataFrame) -> None:
    success = _success_flag(goes).astype(bool)
    td_team = goes["td_team"].to_numpy()
    posteam = goes["posteam"].to_numpy()
    defteam = goes["defteam"].to_numpy()
    touchdown = goes["touchdown"].fill_null(0).to_numpy() == 1
    def_td = touchdown & (td_team == defteam) & ~success
    safety = (goes["safety"].fill_null(0).to_numpy() == 1) & ~success & ~def_td
    giveaway = (
        (
            (goes["interception"].fill_null(0).to_numpy() == 1)
            | (goes["fumble_lost"].fill_null(0).to_numpy() == 1)
        )
        & ~success
        & ~def_td
        & ~safety
    )
    fail = ~success
    n_fail = int(fail.sum())
    if n_fail < 30:
        return
    book.fail_def_td = float(def_td.sum() / n_fail)
    book.fail_safety = float(safety.sum() / n_fail)
    book.fail_giveaway = float(giveaway.sum() / n_fail)
    book.fail_downs = float(max(0.0, 1.0 - book.fail_def_td - book.fail_safety - book.fail_giveaway))


def _quantiles(values: np.ndarray, m: int) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros(m)
    idx = np.linspace(0, len(values) - 1, m).astype(int)
    return values[idx]
