"""Expected win probability of going for it, punting, and kicking.

Each option is a probability-weighted average of the win-probability model
at the snap that would follow. The weights come from the conversion model,
the field-goal model, and the punt distribution. The win-probability model
never sees which button the coach pressed. It only answers "if the game
were in this state, how often would the team with the ball win?"

Clock rule, applied the same way to every option:

* if the play would run out the half in the first half, the next state is
  the second-half kickoff (timeouts reset, score updated if someone scored)
* if the play would run out the game, win probability is 1, 0, or 0.5 from
  the final score
* otherwise the next snap keeps the remaining clock

A failed fourth down with three seconds left therefore does not hand the
opponent a full drive. That matters, and it is a common way to accidentally
make "go for it" look reckless late in a half.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fourth_down.conversion import BUCKETS, ConversionModel
from fourth_down.features import pack_wp_features
from fourth_down.field_goal import FieldGoalModel
from fourth_down.football import (
    kickoff_yardline_array,
    missed_fg_opponent_yardline,
    terminal_win_prob,
)
from fourth_down.punt import PuntModel
from fourth_down.wp import WPModel

SITUATION_COLUMNS = [
    "score_diff",
    "game_seconds_remaining",
    "yardline_100",
    "ydstogo",
    "timeouts_off",
    "timeouts_def",
    "is_home",
    "expected_margin",
    "receive_2h",
    "spread_missing",
    "season",
    "indoor",
    "wind_mph",
    "cold",
    "altitude",
]


def choice_win_probs(
    situations: pl.DataFrame,
    wp: WPModel,
    fg: FieldGoalModel,
    conv: ConversionModel,
    punt: PuntModel,
) -> pl.DataFrame:
    """Add wp_go, wp_punt, wp_fg, recommended, and wp_edge to each situation."""
    missing = [col for col in SITUATION_COLUMNS if col not in situations.columns]
    if missing:
        raise ValueError(f"situations are missing columns: {missing}")
    if situations.height == 0:
        return situations.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("wp_go"),
                pl.lit(None).cast(pl.Float64).alias("wp_punt"),
                pl.lit(None).cast(pl.Float64).alias("wp_fg"),
                pl.lit(None).cast(pl.Utf8).alias("recommended"),
                pl.lit(None).cast(pl.Float64).alias("wp_edge"),
            ]
        )

    sit = _Sit(situations)
    engine = _Engine(wp, fg, conv, punt, sit)
    wp_go = engine.go()
    wp_punt = engine.punt()
    wp_fg = engine.field_goal()
    stacked = np.column_stack([wp_go, wp_punt, wp_fg])
    order = np.argsort(-stacked, axis=1, kind="mergesort")
    best = stacked[np.arange(len(stacked)), order[:, 0]]
    second = stacked[np.arange(len(stacked)), order[:, 1]]
    names = np.array(["go", "punt", "fg"])
    recommended = names[order[:, 0]]
    out = situations.with_columns(
        [
            pl.Series("wp_go", wp_go),
            pl.Series("wp_punt", wp_punt),
            pl.Series("wp_fg", wp_fg),
            pl.Series("recommended", recommended),
            pl.Series("wp_edge", best - second),
            pl.Series("wp_best", best),
        ]
    )
    if "choice" in situations.columns:
        actual = situations["choice"].to_numpy()
        wp_actual = np.where(
            actual == "go",
            wp_go,
            np.where(actual == "punt", wp_punt, np.where(actual == "fg", wp_fg, np.nan)),
        )
        out = out.with_columns(
            [
                pl.Series("wp_actual", wp_actual),
                pl.Series("wp_left", best - wp_actual),
                pl.Series("followed", actual == recommended),
            ]
        )
    return out


class _Sit:
    """Situation columns as numpy arrays. One row is one fourth down."""

    def __init__(self, frame: pl.DataFrame):
        self.n = frame.height
        self.score = frame["score_diff"].to_numpy().astype(float)
        self.gsr = frame["game_seconds_remaining"].to_numpy().astype(float)
        self.yl = frame["yardline_100"].to_numpy().astype(float)
        self.ytg = frame["ydstogo"].to_numpy().astype(float)
        self.to_off = frame["timeouts_off"].to_numpy().astype(float)
        self.to_def = frame["timeouts_def"].to_numpy().astype(float)
        self.is_home = frame["is_home"].to_numpy().astype(float)
        self.margin = frame["expected_margin"].to_numpy().astype(float)
        self.receive = frame["receive_2h"].to_numpy().astype(float)
        self.spread_missing = frame["spread_missing"].to_numpy().astype(float)
        self.season = frame["season"].to_numpy().astype(float)
        self.indoor = frame["indoor"].to_numpy().astype(float)
        self.wind = frame["wind_mph"].to_numpy().astype(float)
        self.cold = frame["cold"].to_numpy().astype(float)
        self.altitude = frame["altitude"].to_numpy().astype(float)

    def take(self, mask: np.ndarray) -> "_Sit":
        child = _Sit.__new__(_Sit)
        for name, value in self.__dict__.items():
            if name == "n":
                continue
            setattr(child, name, value[mask])
        child.n = int(mask.sum())
        return child


class _Engine:
    def __init__(self, wp: WPModel, fg: FieldGoalModel, conv: ConversionModel, punt: PuntModel, sit: _Sit):
        self.wp = wp
        self.fg = fg
        self.conv = conv
        # Named punt_model so it does not shadow the punt() method below.
        self.punt_model = punt
        self.sit = sit
        self._cache: dict[bytes, np.ndarray] = {}

    def go(self) -> np.ndarray:
        sit = self.sit
        probs = self.conv.predict(sit.ytg, sit.yl, sit.season)
        book = self.conv.transitions
        after_td = self._points_then_kickoff(
            points=float("nan"),
            point_values=np.array([6.0, 7.0, 8.0]),
            weights=book.td_point_weights,
            kicking_team_receives_kickoff=False,
            elapsed=book.elapsed_score,
        )
        after_def_td = self._points_then_kickoff(
            points=float("nan"),
            point_values=np.array([-6.0, -7.0, -8.0]),
            weights=book.td_point_weights,
            kicking_team_receives_kickoff=True,
            elapsed=book.elapsed_score,
        )
        fd = self._first_down_wp(after_td, book.elapsed_go)
        downs = self._turnover_wp(book.downs_shift, book.elapsed_go)
        give = self._turnover_wp(book.giveaway_shift, book.elapsed_go)
        # A safety is rare. The next snap is a free kick, not "they take over
        # at the line of scrimmage," so the yard line is an absolute spot.
        safety = self._absolute_opponent_wp(
            np.array([book.safety_opp_yl]), book.elapsed_go, points=-2.0
        )
        return (
            probs["p_td"] * after_td
            + probs["p_fd"] * fd
            + probs["p_downs"] * downs
            + probs["p_giveaway"] * give
            + probs["p_def_td"] * after_def_td
            + probs["p_safety"] * safety
        )

    def field_goal(self) -> np.ndarray:
        sit = self.sit
        # NFL field goals are spotted 17 yards behind the line of scrimmage
        # (the end zone plus the hold). Kicks past 68 yards are treated as
        # unavailable. The make rate out there is not identified, and the
        # miss spot (often the offense's own 20) would otherwise look like a
        # free punt that also has a tiny chance of three points.
        distance = sit.yl + 17.0
        p_make = self.fg.predict(distance, sit.indoor, sit.wind, sit.cold, sit.altitude, sit.season)
        made = self._points_then_kickoff(
            points=3.0,
            point_values=None,
            weights=None,
            kicking_team_receives_kickoff=False,
            elapsed=self.conv.transitions.elapsed_fg,
        )
        residual = np.asarray(self.fg.miss_residual, dtype=float)
        formula = np.asarray(missed_fg_opponent_yardline(sit.yl), dtype=float)
        miss = self._absolute_opponent_wp(formula, self.conv.transitions.elapsed_fg, points=0.0, residual=residual)
        value = p_make * made + (1.0 - p_make) * miss
        return np.where(distance <= 68.0, value, 0.0)

    def punt(self) -> np.ndarray:
        sit = self.sit
        out = np.zeros(sit.n, dtype=float)
        rounded = np.clip(np.rint(sit.yl), 1, 99).astype(int)
        for yl in np.unique(rounded):
            mask = rounded == yl
            out[mask] = self._punt_at(int(yl), mask)
        return out

    def _punt_at(self, yl: int, mask: np.ndarray) -> np.ndarray:
        spot = self.punt_model.spot(yl)
        sub = self.sit.take(mask)
        # Evaluate this yard line's distribution with a temporary engine so
        # the arrays stay the length of the subgroup.
        sub_engine = _Engine(self.wp, self.fg, self.conv, self.punt_model, sub)
        sub_engine._cache = self._cache
        total = np.zeros(sub.n, dtype=float)
        elapsed = self.punt_model.elapsed
        if spot.p_opp > 0:
            total += spot.p_opp * sub_engine._paired_opponent_wp(spot.opp_yl, spot.opp_delta, elapsed)
        if spot.p_keep > 0:
            total += spot.p_keep * sub_engine._paired_keep_wp(spot.keep_yl, spot.keep_delta, elapsed)
        if spot.p_we_score > 0:
            total += spot.p_we_score * sub_engine._points_then_kickoff(
                points=spot.we_score_points,
                point_values=None,
                weights=None,
                kicking_team_receives_kickoff=False,
                elapsed=elapsed,
            )
        if spot.p_they_score > 0:
            total += spot.p_they_score * sub_engine._points_then_kickoff(
                points=-spot.they_score_points,
                point_values=None,
                weights=None,
                kicking_team_receives_kickoff=True,
                elapsed=elapsed,
            )
        return total

    def _first_down_wp(self, wp_if_score: np.ndarray, elapsed: float) -> np.ndarray:
        """Win probability after a conversion that was not ruled a touchdown up front.

        Some samples still cross the goal line (a fourth-and-1 from the 4 that
        gains 6). Those are scored as touchdowns so the residual cannot turn
        a breakaway into "first and goal from the 1."
        """
        sit = self.sit
        out = np.zeros(sit.n, dtype=float)
        buckets = _ytg_bucket(sit.ytg)
        for bucket in BUCKETS:
            mask = buckets == bucket
            if not np.any(mask):
                continue
            extras = self.conv.transitions.fd_extra[bucket]
            sub = self.sit.take(mask)
            raw = sub.yl[:, None] - sub.ytg[:, None] - extras[None, :]
            scored = raw <= 0
            new_yl = np.clip(raw, 1.0, 99.0)
            new_ytg = np.minimum(10.0, new_yl)
            sample_wp = np.broadcast_to(wp_if_score[mask][:, None], raw.shape).copy()
            live_pairs = ~scored
            if np.any(live_pairs):
                row, col = np.where(live_pairs)
                sample_wp[row, col] = self._next_own_ball(
                    sub,
                    row,
                    new_yl[row, col],
                    new_ytg[row, col],
                    elapsed,
                    points=0.0,
                )
            out[mask] = sample_wp.mean(axis=1)
        return out

    def _turnover_wp(self, shifts: np.ndarray, elapsed: float, points: float = 0.0) -> np.ndarray:
        sit = self.sit
        shifts = np.asarray(shifts, dtype=float)
        opp_yl = np.clip((100.0 - sit.yl)[:, None] + shifts[None, :], 1.0, 99.0)
        return self._sample_opponent(opp_yl, elapsed, points)

    def _absolute_opponent_wp(
        self,
        opp_yl: np.ndarray,
        elapsed: float,
        points: float,
        residual: np.ndarray | None = None,
    ) -> np.ndarray:
        """Opponent yard line is already in yardline_100, not a shift from the LOS.

        A length-1 yard line (the safety spot) is broadcast to every situation.
        A length-n array is one spot per fourth down, which is what a missed
        field goal uses before the residual is added.
        """
        base = np.atleast_1d(np.asarray(opp_yl, dtype=float)).reshape(-1)
        if base.shape[0] == 1 and self.sit.n != 1:
            base = np.full(self.sit.n, float(base[0]))
        if base.shape[0] != self.sit.n:
            raise ValueError(
                f"opponent yard line has length {base.shape[0]} and there are {self.sit.n} situations"
            )
        if residual is None:
            grid = base.reshape(-1, 1)
        else:
            grid = np.clip(base[:, None] + np.asarray(residual, dtype=float)[None, :], 1.0, 99.0)
        return self._sample_opponent(grid, elapsed, points)

    def _sample_opponent(self, opp_yl: np.ndarray, elapsed: float, points: float) -> np.ndarray:
        sit = self.sit
        n, m = opp_yl.shape
        live, half, over, new_gsr = _clock(sit.gsr, elapsed)
        sample = np.empty((n, m), dtype=float)
        if np.any(over):
            sample[over, :] = terminal_win_prob(sit.score[over] + points)[:, None]
        if np.any(half):
            sample[half, :] = self._halftime_wp(sit.score[half] + points, half)[:, None]
        if np.any(live):
            row = np.repeat(np.where(live)[0], m)
            col = np.tile(np.arange(m), int(live.sum()))
            state_score = sit.score[row] + points
            sample[row, col] = self._defense_wp(
                state_score,
                new_gsr[row],
                opp_yl[row, col],
                sit.to_off[row],
                sit.to_def[row],
                sit.is_home[row],
                sit.margin[row],
                sit.receive[row],
                sit.spread_missing[row],
            )
        return sample.mean(axis=1)

    def _paired_opponent_wp(self, opp_yl: np.ndarray, delta: np.ndarray, elapsed: float) -> np.ndarray:
        sit = self.sit
        opp_yl = np.asarray(opp_yl, dtype=float)
        delta = np.asarray(delta, dtype=float)
        m = len(opp_yl)
        grid = np.broadcast_to(opp_yl, (sit.n, m)).copy()
        # Score change is part of the historical punt, so each quantile carries
        # its own delta instead of one number for the whole yard line.
        n = sit.n
        live, half, over, new_gsr = _clock(sit.gsr, elapsed)
        sample = np.empty((n, m), dtype=float)
        if np.any(over):
            # Game over: the return can still score before the clock dies.
            sample[over, :] = terminal_win_prob(sit.score[over, None] + delta[None, :])
        if np.any(half):
            # Half over: a return TD still counts; an ordinary punt does not
            # give the opponent a drive. Approximate with the delta only.
            sample[half, :] = self._halftime_wp_delta(sit.score[half], delta, half)
        if np.any(live):
            rows = np.where(live)[0]
            row = np.repeat(rows, m)
            col = np.tile(np.arange(m), len(rows))
            sample[row, col] = self._defense_wp(
                sit.score[row] + delta[col],
                new_gsr[row],
                grid[row, col],
                sit.to_off[row],
                sit.to_def[row],
                sit.is_home[row],
                sit.margin[row],
                sit.receive[row],
                sit.spread_missing[row],
            )
        return sample.mean(axis=1)

    def _paired_keep_wp(self, keep_yl: np.ndarray, delta: np.ndarray, elapsed: float) -> np.ndarray:
        sit = self.sit
        keep_yl = np.clip(np.asarray(keep_yl, dtype=float), 1.0, 99.0)
        delta = np.asarray(delta, dtype=float)
        m = len(keep_yl)
        n = sit.n
        live, half, over, new_gsr = _clock(sit.gsr, elapsed)
        sample = np.empty((n, m), dtype=float)
        if np.any(over):
            sample[over, :] = terminal_win_prob(sit.score[over, None] + delta[None, :])
        if np.any(half):
            sample[half, :] = self._halftime_wp_delta(sit.score[half], delta, half)
        if np.any(live):
            rows = np.where(live)[0]
            row = np.repeat(rows, m)
            col = np.tile(np.arange(m), len(rows))
            yl = np.broadcast_to(keep_yl, (n, m))[row, col]
            sample[row, col] = self._offense_wp(
                sit.score[row] + delta[col],
                new_gsr[row],
                yl,
                np.ones(len(row)),
                np.minimum(10.0, yl),
                sit.to_off[row],
                sit.to_def[row],
                sit.is_home[row],
                sit.margin[row],
                sit.receive[row],
                sit.spread_missing[row],
            )
        return sample.mean(axis=1)

    def _next_own_ball(self, sub: _Sit, row: np.ndarray, yl: np.ndarray, ytg: np.ndarray, elapsed: float, points: float) -> np.ndarray:
        """Win probability for a subset of (situation, sample) pairs that kept the ball."""
        gsr = sub.gsr[row]
        score = sub.score[row] + points
        live, half, over, new_gsr = _clock(gsr, elapsed)
        out = np.empty(len(row), dtype=float)
        if np.any(over):
            out[over] = terminal_win_prob(score[over])
        if np.any(half):
            # Map half rows back to the subgroup, then to the original receive flag.
            out[half] = self._halftime_from_sub(sub, row[half], score[half])
        if np.any(live):
            out[live] = self._offense_wp(
                score[live],
                new_gsr[live],
                yl[live],
                np.ones(int(live.sum())),
                ytg[live],
                sub.to_off[row[live]],
                sub.to_def[row[live]],
                sub.is_home[row[live]],
                sub.margin[row[live]],
                sub.receive[row[live]],
                sub.spread_missing[row[live]],
            )
        return out

    def _points_then_kickoff(
        self,
        points: float,
        point_values: np.ndarray | None,
        weights: np.ndarray | None,
        kicking_team_receives_kickoff: bool,
        elapsed: float,
    ) -> np.ndarray:
        """WP after points go on the board and the other team kicks off.

        `kicking_team_receives_kickoff` is from the original offense's view.
        False means we scored and they receive. True means they scored and
        we receive. If the clock runs out, nobody kicks off.
        """
        if point_values is None:
            return self._one_kickoff(points, kicking_team_receives_kickoff, elapsed)
        total = np.zeros(self.sit.n, dtype=float)
        for value, weight in zip(point_values, weights, strict=True):
            if weight <= 0:
                continue
            total += weight * self._one_kickoff(float(value), kicking_team_receives_kickoff, elapsed)
        return total

    def _one_kickoff(self, points: float, we_receive_kickoff: bool, elapsed: float) -> np.ndarray:
        sit = self.sit
        live, half, over, new_gsr = _clock(sit.gsr, elapsed)
        out = np.empty(sit.n, dtype=float)
        new_score = sit.score + points
        if np.any(over):
            out[over] = terminal_win_prob(new_score[over])
        if np.any(half):
            out[half] = self._halftime_wp(new_score[half], half)
        if np.any(live):
            kick_yl = kickoff_yardline_array(sit.season[live])
            if we_receive_kickoff:
                out[live] = self._offense_wp(
                    new_score[live],
                    new_gsr[live],
                    kick_yl,
                    np.ones(int(live.sum())),
                    np.full(int(live.sum()), 10.0),
                    sit.to_off[live],
                    sit.to_def[live],
                    sit.is_home[live],
                    sit.margin[live],
                    sit.receive[live],
                    sit.spread_missing[live],
                )
            else:
                out[live] = self._defense_wp(
                    new_score[live],
                    new_gsr[live],
                    kick_yl,
                    sit.to_off[live],
                    sit.to_def[live],
                    sit.is_home[live],
                    sit.margin[live],
                    sit.receive[live],
                    sit.spread_missing[live],
                )
        return out

    def _halftime_wp(self, new_score: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Second-half kickoff for the rows where `mask` is true on the full situation list."""
        sit = self.sit
        return self._halftime_rows(
            new_score,
            sit.receive[mask],
            sit.is_home[mask],
            sit.margin[mask],
            sit.spread_missing[mask],
            sit.season[mask],
        )

    def _halftime_from_sub(self, sub: _Sit, sub_rows: np.ndarray, new_score: np.ndarray) -> np.ndarray:
        return self._halftime_rows(
            new_score,
            sub.receive[sub_rows],
            sub.is_home[sub_rows],
            sub.margin[sub_rows],
            sub.spread_missing[sub_rows],
            sub.season[sub_rows],
        )

    def _halftime_wp_delta(self, score: np.ndarray, delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Halftime win probability when each sample adds its own score delta.

        `score` is already restricted to the halftime rows. `delta` is the
        sample axis. A return touchdown counts; the yard line of an ordinary
        punt does not, because the half is over.
        """
        m = len(delta)
        out = np.empty((len(score), m), dtype=float)
        sit = self.sit
        for j, change in enumerate(delta):
            out[:, j] = self._halftime_rows(
                score + change,
                sit.receive[mask],
                sit.is_home[mask],
                sit.margin[mask],
                sit.spread_missing[mask],
                sit.season[mask],
            )
        return out

    def _halftime_rows(
        self,
        new_score: np.ndarray,
        receive: np.ndarray,
        is_home: np.ndarray,
        margin: np.ndarray,
        spread_missing: np.ndarray,
        season: np.ndarray,
    ) -> np.ndarray:
        n = len(new_score)
        kick_yl = kickoff_yardline_array(season)
        gsr = np.full(n, 1800.0)
        timeouts = np.full(n, 3.0)
        we_receive = receive >= 0.5
        out = np.empty(n, dtype=float)
        if np.any(we_receive):
            out[we_receive] = self._offense_wp(
                new_score[we_receive],
                gsr[we_receive],
                kick_yl[we_receive],
                np.ones(int(we_receive.sum())),
                np.full(int(we_receive.sum()), 10.0),
                timeouts[we_receive],
                timeouts[we_receive],
                is_home[we_receive],
                margin[we_receive],
                np.zeros(int(we_receive.sum())),
                spread_missing[we_receive],
            )
        other = ~we_receive
        if np.any(other):
            out[other] = self._defense_wp(
                new_score[other],
                gsr[other],
                kick_yl[other],
                timeouts[other],
                timeouts[other],
                is_home[other],
                margin[other],
                np.zeros(int(other.sum())),
                spread_missing[other],
            )
        return out

    def _offense_wp(
        self,
        score,
        gsr,
        yl,
        down,
        ytg,
        to_off,
        to_def,
        is_home,
        margin,
        receive,
        spread_missing,
    ) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        gsr = np.asarray(gsr, dtype=float)
        dead = gsr <= 0
        out = np.empty(len(score), dtype=float)
        if np.any(dead):
            out[dead] = terminal_win_prob(score[dead])
        live = ~dead
        if np.any(live):
            features = pack_wp_features(
                score[live],
                gsr[live],
                np.asarray(yl, dtype=float)[live],
                np.asarray(down, dtype=float)[live],
                np.asarray(ytg, dtype=float)[live],
                np.asarray(to_off, dtype=float)[live],
                np.asarray(to_def, dtype=float)[live],
                np.asarray(is_home, dtype=float)[live],
                np.asarray(margin, dtype=float)[live],
                np.asarray(receive, dtype=float)[live],
                np.asarray(spread_missing, dtype=float)[live],
            )
            out[live] = np.clip(self._predict(features), 1e-4, 1 - 1e-4)
        return out

    def _defense_wp(
        self,
        our_score,
        gsr,
        opp_yl,
        to_off,
        to_def,
        is_home,
        margin,
        receive,
        spread_missing,
    ) -> np.ndarray:
        """Our win probability when the opponent will snap the ball.

        The model speaks for the team about to snap. Their win probability is
        subtracted from 1. Ties are rare enough that this is a small bias,
        and it is called out in the limitations.
        """
        their_wp = self._offense_wp(
            -np.asarray(our_score, dtype=float),
            gsr,
            opp_yl,
            np.ones(len(np.asarray(our_score))),
            np.minimum(10.0, np.asarray(opp_yl, dtype=float)),
            to_def,
            to_off,
            1.0 - np.asarray(is_home, dtype=float),
            -np.asarray(margin, dtype=float),
            1.0 - np.asarray(receive, dtype=float),
            spread_missing,
        )
        return 1.0 - their_wp

    def _predict(self, features: np.ndarray) -> np.ndarray:
        """Win probability for a batch of feature rows, collapsing duplicates.

        Rounding to three decimals lets identical game states created by
        different options share one prediction. The cache on the engine
        reuses those states across go, punt, and kick.
        """
        if len(features) == 0:
            return np.array([], dtype=float)
        key = np.round(np.asarray(features, dtype=np.float64), 3)
        uniq, inverse = np.unique(key, axis=0, return_inverse=True)
        cached = np.empty(len(uniq), dtype=float)
        missing = np.ones(len(uniq), dtype=bool)
        for i, row in enumerate(uniq):
            hit = self._cache.get(row.tobytes())
            if hit is not None:
                cached[i] = hit
                missing[i] = False
        if np.any(missing):
            block = np.ascontiguousarray(uniq[missing], dtype=np.float32)
            predicted = np.asarray(self.wp.predict(block), dtype=float)
            cached[missing] = predicted
            for row, value in zip(uniq[missing], predicted, strict=True):
                self._cache[row.tobytes()] = float(value)
        return cached[inverse]


def _clock(gsr: np.ndarray, elapsed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split snaps into live play, end of half, and end of game."""
    gsr = np.asarray(gsr, dtype=float)
    half_left = np.where(gsr > 1800.0, gsr - 1800.0, gsr)
    game_over = (gsr <= 1800.0) & (elapsed >= half_left)
    halftime = (gsr > 1800.0) & (elapsed >= (gsr - 1800.0))
    live = ~game_over & ~halftime
    new_gsr = np.clip(gsr - elapsed, 0.0, 3600.0)
    return live, halftime, game_over, new_gsr


def pre_snap_win_prob(situations: pl.DataFrame, wp: WPModel) -> np.ndarray:
    """Win probability of the current snap, before any fourth-down choice.

    Used to drop garbage time from the coach tables. It is the same model
    that prices the options, evaluated at the state the coach is staring at.
    """
    missing = [col for col in SITUATION_COLUMNS if col not in situations.columns]
    if missing:
        raise ValueError(f"situations are missing columns: {missing}")
    features = pack_wp_features(
        situations["score_diff"].to_numpy(),
        situations["game_seconds_remaining"].to_numpy(),
        situations["yardline_100"].to_numpy(),
        np.full(situations.height, 4.0),
        situations["ydstogo"].to_numpy(),
        situations["timeouts_off"].to_numpy(),
        situations["timeouts_def"].to_numpy(),
        situations["is_home"].to_numpy(),
        situations["expected_margin"].to_numpy(),
        situations["receive_2h"].to_numpy(),
        situations["spread_missing"].to_numpy(),
    )
    return np.asarray(wp.predict(features), dtype=float)


def _ytg_bucket(ydstogo: np.ndarray) -> np.ndarray:
    ytg = np.asarray(ydstogo, dtype=int)
    bucket = np.full(ytg.shape, 11)
    bucket = np.where(ytg <= 10, 7, bucket)
    bucket = np.where(ytg <= 6, 4, bucket)
    bucket = np.where(ytg <= 3, 3, bucket)
    bucket = np.where(ytg == 2, 2, bucket)
    bucket = np.where(ytg <= 1, 1, bucket)
    return bucket
