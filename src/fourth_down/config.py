"""Paths, seasons, and knobs that the rest of the package shares.

Season choice: the nflverse "current" season can be a season still in progress.
This project stops at the last completed season so coach grades are not mixed
with a half-finished year. Update LAST_FULL_SEASON when a new season ends.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SLIM_PATH = RAW_DIR / "pbp_slim.parquet"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SQL_DIR = ROOT / "sql"

# 2014 is far enough back to train a win-probability model and still sits in
# the modern extra-point era's neighborhood (the PAT moved in 2015). 2025 is
# the last full season as of October 2026; the 2026 season was only in week 7.
FIRST_SEASON = 2014
LAST_FULL_SEASON = 2025
# First season graded out of sample. Earlier seasons are training history only,
# so every graded coach decision is scored by a model that never saw that season.
FIRST_TEST_SEASON = 2018

# A recommendation is "clear" when the best option beats the runner-up by this
# much win probability. Below this, the model is calling a toss-up.
CLEAR_EDGE = 0.02

# Leaderboard filters. See analysis.leaderboard_mask for the reasoning.
MIN_HALF_SECONDS = 30
MAX_YARDS_TO_GO = 20
WP_GARBAGE_LOW = 0.02
WP_GARBAGE_HIGH = 0.98

RANDOM_SEED = 42

# How many representative outcomes to keep per punt yard line and per
# conversion result. Quantile samples, not a Monte Carlo draw, so a rerun
# with the same data produces the same chart.
PUNT_SAMPLE_SIZE = 41
CONVERSION_SAMPLE_SIZE = 21

# XGBoost size. Overridden by --quick for a shorter debug pass.
# Depth 5, not 3 or 4: field position matters much more late than early, and a
# shallow tree spends its splits on the score. `field_time` carries that
# interaction explicitly so the tree does not have to discover it alone.
WP_TREES = 400
WP_LEARNING_RATE = 0.05
WP_MAX_DEPTH = 5
WP_MIN_CHILD_WEIGHT = 40

WP_FEATURES = [
    "score_diff",
    "game_seconds_remaining",
    "yardline_100",
    "down",
    "ydstogo",
    "timeouts_off",
    "timeouts_def",
    "is_home",
    "expected_margin",
    "receive_2h_if_relevant",
    "quarter",
    "score_time",
    "goal_to_go",
    "spread_missing",
    "field_time",
]

# +1 means "a larger value of this feature may not decrease predicted win
# probability." -1 means it may not increase it. 0 leaves the feature free.
# Score, pregame strength, home field, offensive timeouts, and (in the first
# half) receiving the second-half kickoff are constrained. Field position is
# constrained so moving away from the opponent's end zone cannot raise WP.
# `field_time` is positive when the offense is across midfield, and larger
# when the clock is short, so it is constrained upward too.
WP_MONOTONE = (1, 0, -1, 0, -1, 1, -1, 1, 1, 1, 0, 1, 0, 0, 1)

# Columns pulled from nflverse. Win probability, expected points, and fg_prob
# from nflfastR are kept only for the sanity check. They are never model inputs.
SLIM_COLUMNS = [
    "play_id",
    "game_id",
    "home_team",
    "away_team",
    "season_type",
    "week",
    "posteam",
    "posteam_type",
    "defteam",
    "side_of_field",
    "yardline_100",
    "game_date",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "qtr",
    "down",
    "goal_to_go",
    "time",
    "yrdln",
    "ydstogo",
    "desc",
    "play_type",
    "yards_gained",
    "field_goal_result",
    "kick_distance",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "posteam_score",
    "defteam_score",
    "score_differential",
    "score_differential_post",
    "wp",
    "vegas_wp",
    "punt_blocked",
    "first_down",
    "fourth_down_converted",
    "fourth_down_failed",
    "touchback",
    "interception",
    "touchdown",
    "fumble_lost",
    "safety",
    "td_team",
    "return_touchdown",
    "season",
    "roof",
    "temp",
    "wind",
    "home_coach",
    "away_coach",
    "spread_line",
    "result",
    "home_opening_kickoff",
    "qb_kneel",
    "qb_spike",
]
