"""Component models on synthetic data, and the decision rule with a fake win probability."""

import numpy as np
import polars as pl
import pytest

from fourth_down.conversion import SUCCESS_NAMES, _design, _fit_binomial
from fourth_down.decide import choice_win_probs
from fourth_down.field_goal import FieldGoalModel
from fourth_down.football import expit
from fourth_down.punt import PuntSpot
from fourth_down.situations import make_situations
from fourth_down.conversion import BUCKETS, TransitionBook


class _FakeWP:
    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        score = features[:, 0]
        yardline = features[:, 2]
        # Smooth and monotone: a lead helps, being farther from the end zone hurts.
        return expit(0.18 * score - 0.03 * (yardline - 50))


class _FakeFG:
    miss_residual = np.array([0.0])

    def predict(self, distance, indoor, wind, cold, altitude, season):
        return np.clip(0.98 - (np.asarray(distance, dtype=float) - 18.0) / 50.0, 0.02, 0.97)


class _FakeConv:
    def __init__(self):
        self.transitions = TransitionBook(
            elapsed_go=6.0,
            elapsed_score=5.0,
            elapsed_fg=5.0,
            td_point_weights=np.array([0.05, 0.90, 0.05]),
            safety_opp_yl=60.0,
            fd_extra={bucket: np.array([1.0, 4.0]) for bucket in BUCKETS},
            downs_shift=np.array([0.0, -1.0]),
            giveaway_shift=np.array([-4.0, 0.0]),
        )

    def predict(self, ydstogo, yardline_100, season):
        ytg = np.asarray(ydstogo, dtype=float)
        yl = np.asarray(yardline_100, dtype=float)
        p_success = 1.0 / (1.0 + 0.45 * (ytg - 1.0))
        p_td = np.where(yl <= ytg, 0.62, 0.04)
        p_td = np.minimum(p_td, p_success * 0.85)
        p_fd = p_success - p_td
        p_fail = 1.0 - p_success
        return {
            "p_td": p_td,
            "p_fd": p_fd,
            "p_downs": p_fail * 0.90,
            "p_giveaway": p_fail * 0.08,
            "p_def_td": p_fail * 0.015,
            "p_safety": p_fail * 0.005,
        }


class _FakePunt:
    elapsed = 8.0

    def spot(self, yardline_100: float) -> PuntSpot:
        # A deep punt leaves the opponent around their own 25 to 30.
        # A punt from the opponent's side is a bad idea and the spot says so.
        if yardline_100 >= 50:
            opp = 74.0
        else:
            opp = 88.0
        return PuntSpot(
            p_opp=0.96,
            p_keep=0.03,
            p_we_score=0.0,
            p_they_score=0.01,
            opp_yl=np.array([opp]),
            keep_yl=np.array([float(np.clip(yardline_100 - 8, 1, 99))]),
            opp_delta=np.array([0.0]),
            keep_delta=np.array([0.0]),
            they_score_points=7.0,
            we_score_points=7.0,
            n=100,
        )


def _neutral(yardline: float, ydstogo: float) -> pl.DataFrame:
    return make_situations(
        [
            {
                "score_diff": 0.0,
                "game_seconds_remaining": 3600.0,
                "yardline_100": yardline,
                "ydstogo": ydstogo,
                "season": 2024.0,
                "receive_2h": 1.0,
            }
        ]
    )


def test_backed_up_fourth_and_long_punts():
    graded = choice_win_probs(_neutral(90, 10), _FakeWP(), _FakeFG(), _FakeConv(), _FakePunt())
    row = graded.row(0, named=True)
    assert row["recommended"] == "punt"
    assert row["wp_punt"] > row["wp_go"]
    assert row["wp_edge"] > 0


def test_fourth_and_goal_from_the_two_goes_for_it():
    graded = choice_win_probs(_neutral(2, 1), _FakeWP(), _FakeFG(), _FakeConv(), _FakePunt())
    row = graded.row(0, named=True)
    assert row["recommended"] == "go"
    assert row["wp_go"] > row["wp_fg"]
    assert row["wp_go"] > row["wp_punt"]


def test_conversion_probability_falls_as_yards_to_go_grow():
    rng = np.random.default_rng(0)
    n = 2500
    ytg = rng.integers(1, 13, n).astype(float)
    yl = rng.integers(5, 95, n).astype(float)
    goal = (yl <= ytg).astype(float)
    years = np.zeros(n)
    # A logistic curve with a known negative yards-to-go slope, plus noise.
    eta = 1.4 - 1.6 * (ytg / 10) - 0.2 * (yl / 100)
    y = (rng.random(n) < expit(eta)).astype(float)
    coef, _, _, ok = _fit_binomial(y, _design(ytg, yl, goal, years), "success")
    assert ok
    short = expit(_design(np.array([1.0]), np.array([50.0]), np.array([0.0]), np.array([0.0])) @ coef)
    long = expit(_design(np.array([10.0]), np.array([50.0]), np.array([0.0]), np.array([0.0])) @ coef)
    assert short.item() > long.item()
    assert SUCCESS_NAMES[1] == "ydstogo_per_10"


def test_field_goal_curve_decreases_with_distance():
    rng = np.random.default_rng(1)
    distance = rng.integers(22, 62, 1800).astype(float)
    # Roughly the real shape: very likely inside 30, a coin flip around the mid-50s.
    p = np.clip(1.15 - distance / 70, 0.05, 0.98)
    made = rng.random(len(distance)) < p
    result = np.where(made, "made", "missed")
    kicks = pl.DataFrame(
        {
            "play_type": ["field_goal"] * len(distance),
            "field_goal_result": result,
            "kick_distance": distance,
            "qtr": np.full(len(distance), 2.0),
            "season": np.full(len(distance), 2022),
            "indoor": np.zeros(len(distance)),
            "wind_mph": np.zeros(len(distance)),
            "cold": np.zeros(len(distance)),
            "altitude": np.zeros(len(distance)),
            "transition_ok": np.zeros(len(distance), dtype=bool),
            "kept_ball": np.zeros(len(distance), dtype=bool),
            "yardline_100": distance - 17,
            "n_yardline_100": np.full(len(distance), 60.0),
        }
    )
    model = FieldGoalModel().fit(kicks)
    short = float(model.distance_probability(28))
    long = float(model.distance_probability(55))
    assert short > 0.7
    assert long < short
    assert long < 0.6
