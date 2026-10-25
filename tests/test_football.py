"""Rules and scoring functions that do not need the play-by-play file."""

import numpy as np
import polars as pl
import pytest

from fourth_down.data import attach_next_scrimmage
from fourth_down.decide import _clock
from fourth_down.features import with_model_columns
from fourth_down.football import (
    format_yardline,
    kickoff_yardline_100,
    missed_fg_opponent_yardline,
    pick_recommendation,
    terminal_win_prob,
    wp_left_on_table,
)
from fourth_down.metrics import binary_log_loss, brier_score, calibration_table, season_folds


def test_missed_field_goal_spot_from_the_opponents_30():
    # Line of scrimmage at the opponent's 30. Kick spot is 8 yards back,
    # the opponent's 38, which is 62 yards from the kicking team's end zone.
    assert missed_fg_opponent_yardline(30) == pytest.approx(62.0)


def test_missed_field_goal_inside_the_20_comes_out_to_the_20():
    # Line of scrimmage at our own 15 (yardline_100 = 85). The kick spot would
    # be inside our 20, so the rule places the ball at the 20 (yardline_100 80).
    assert missed_fg_opponent_yardline(85) == pytest.approx(80.0)


def test_kickoff_touchback_follows_the_rule_era():
    assert kickoff_yardline_100(2014) == 80
    assert kickoff_yardline_100(2018) == 75
    assert kickoff_yardline_100(2024) == 70
    assert kickoff_yardline_100(2025) == 70


def test_yardline_phrases():
    assert format_yardline(50) == "midfield"
    assert "opponent" in format_yardline(27)
    assert "own" in format_yardline(73)


def test_terminal_win_prob_and_ties():
    assert terminal_win_prob(3) == 1
    assert terminal_win_prob(-1) == 0
    assert terminal_win_prob(0) == 0.5


def test_recommendation_picks_the_highest_and_reports_the_edge():
    choice, edge = pick_recommendation(0.42, 0.40, 0.31)
    assert choice == "go"
    assert edge == pytest.approx(0.02)


def test_wp_left_is_zero_when_the_coach_matches_the_model():
    assert wp_left_on_table(0.50, 0.40, 0.30, "go") == pytest.approx(0.0)
    assert wp_left_on_table(0.50, 0.40, 0.30, "punt") == pytest.approx(0.10)


def test_log_loss_rewards_honest_probabilities():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    perfect = np.array([0.999999, 0.000001, 0.999999, 0.000001])
    coin = np.array([0.5, 0.5, 0.5, 0.5])
    assert binary_log_loss(y, perfect) < 0.01
    assert binary_log_loss(y, coin) == pytest.approx(np.log(2))
    assert brier_score(y, coin) == pytest.approx(0.25)


def test_calibration_bins_cover_every_row():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    table = calibration_table(y, p, n_bins=2)
    assert table["n"].sum() == 6
    assert table["actual_rate"].iloc[0] == 0
    assert table["actual_rate"].iloc[1] == 1


def test_season_folds_never_leak_the_test_season():
    folds = season_folds([2014, 2015, 2016, 2017, 2018, 2019], first_test=2018)
    assert [fold["test"] for fold in folds] == [2018, 2019]
    for fold in folds:
        assert fold["test"] not in fold["train"]
        assert fold["calibrate"] not in fold["train"]
        assert fold["test"] != fold["calibrate"]
        assert max(fold["train"]) < fold["calibrate"] < fold["test"]


def test_clock_split_ends_the_half_and_the_game():
    # 1820 is 20 seconds left in the first half, so a 30-second play ends the half.
    # 120 is still a live second-half snap. 5 seconds left ends the game.
    gsr = np.array([3600.0, 1820.0, 120.0, 5.0])
    live, half, over, _ = _clock(gsr, elapsed=30)
    assert live.tolist() == [True, False, True, False]
    assert half.tolist() == [False, True, False, False]
    assert over.tolist() == [False, False, False, True]


def test_spread_and_second_half_kickoff_flip_for_the_away_team():
    # home_opening_kickoff = 1 means the home team kicked off to start, so
    # the home team receives the second-half kickoff.
    frame = pl.DataFrame(
        {
            "score_differential": [0.0, 0.0],
            "posteam": ["HOME", "AWAY"],
            "defteam": ["AWAY", "HOME"],
            "home_team": ["HOME", "HOME"],
            "away_team": ["AWAY", "AWAY"],
            "spread_line": [3.5, 3.5],
            "roof": ["dome", "outdoors"],
            "wind": [None, 12.0],
            "temp": [70.0, 30.0],
            "home_opening_kickoff": [1.0, 1.0],
            "home_coach": ["A", "A"],
            "away_coach": ["B", "B"],
            "result": [7, 7],
            "posteam_timeouts_remaining": [3, 2],
            "defteam_timeouts_remaining": [3, 3],
            "play_type": ["pass", "punt"],
        }
    )
    out = with_model_columns(frame)
    home, away = out.row(0, named=True), out.row(1, named=True)
    assert home["expected_margin"] == pytest.approx(3.5)
    assert away["expected_margin"] == pytest.approx(-3.5)
    assert home["receive_2h"] == 1
    assert away["receive_2h"] == 0
    assert home["posteam_win"] == 1
    assert away["posteam_win"] == 0
    assert home["coach"] == "A"
    assert away["coach"] == "B"
    assert home["indoor"] == 1
    assert away["cold"] == 1
    assert home["choice"] == "go"
    assert away["choice"] == "punt"


def test_next_scrimmage_skips_the_kickoff():
    plays = pl.DataFrame(
        {
            "game_id": ["G", "G", "G"],
            "play_id": [10.0, 20.0, 30.0],
            "down": [4.0, None, 1.0],
            "play_type": ["field_goal", "kickoff", "pass"],
            "posteam": ["A", "B", "B"],
            "defteam": ["B", "A", "A"],
            "yardline_100": [30.0, None, 70.0],
            "score_differential": [0.0, -3.0, -3.0],
            "game_seconds_remaining": [1000.0, 996.0, 996.0],
            "qtr": [2.0, 2.0, 2.0],
            "ydstogo": [8.0, None, 10.0],
        }
    )
    out = attach_next_scrimmage(plays)
    first = out.row(0, named=True)
    assert first["n_play_id"] == 30
    assert first["n_posteam"] == "B"
    assert first["kept_ball"] is False
    # Next score differential is from B's view (-3). From A's view that is +3.
    assert first["n_score_for_us"] == pytest.approx(3.0)
    assert first["transition_ok"] is True
