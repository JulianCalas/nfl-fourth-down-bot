"""Pre-play features for the win-probability model.

Every column here is known before the snap. nflfastR's `wp`, `vegas_wp`,
`ep`, and `fg_prob` are deliberately absent: the point of the project is to
build the decision from our own components and only afterwards compare our
win probability to theirs.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fourth_down.config import WP_FEATURES
from fourth_down.football import quarter_from_seconds


def with_model_columns(plays: pl.DataFrame) -> pl.DataFrame:
    """Add the derived columns shared by the win-probability and decision code.

    `spread_line` in nflverse is the closing expected home margin: positive
    means the home team is favored. That was checked against 2024 results
    (the correlation between spread_line and final home margin is positive,
    and large favorites such as Baltimore at the Giants sit on the matching
    side of zero). `expected_margin` flips the sign for the away team so that
    positive always means "the team with the ball is favored."
    """
    return plays.with_columns(
        [
            pl.col("score_differential").alias("score_diff"),
            (pl.col("posteam") == pl.col("home_team")).cast(pl.Float64).alias("is_home"),
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("spread_line"))
            .otherwise(-pl.col("spread_line"))
            .alias("expected_margin"),
            pl.col("spread_line").is_null().cast(pl.Float64).alias("spread_missing"),
            # Opening kickoff decides who receives in the second half. If the
            # home team kicked off to start the game, home receives the second
            # half kickoff. The value is only allowed to affect win probability
            # while that kickoff is still in the future (first half).
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("home_opening_kickoff"))
            .otherwise(1 - pl.col("home_opening_kickoff"))
            .cast(pl.Float64)
            .alias("receive_2h"),
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("home_coach"))
            .otherwise(pl.col("away_coach"))
            .alias("coach"),
            pl.when(pl.col("result") == 0)
            .then(None)
            .when(
                ((pl.col("posteam") == pl.col("home_team")) & (pl.col("result") > 0))
                | ((pl.col("posteam") == pl.col("away_team")) & (pl.col("result") < 0))
            )
            .then(1.0)
            .otherwise(0.0)
            .alias("posteam_win"),
            (pl.col("roof").is_in(["dome", "closed"])).cast(pl.Float64).alias("indoor"),
            (pl.col("home_team") == "DEN").cast(pl.Float64).alias("altitude"),
            pl.col("posteam_timeouts_remaining").alias("timeouts_off"),
            pl.col("defteam_timeouts_remaining").alias("timeouts_def"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("spread_missing") == 1.0)
            .then(0.0)
            .otherwise(pl.col("expected_margin"))
            .alias("expected_margin"),
            pl.col("wind").is_null().cast(pl.Float64).alias("wind_missing"),
            pl.when((pl.col("indoor") == 1.0) | pl.col("wind").is_null())
            .then(0.0)
            .otherwise(pl.col("wind"))
            .alias("wind_mph"),
            pl.when((pl.col("indoor") == 0.0) & (pl.col("temp") <= 40))
            .then(1.0)
            .otherwise(0.0)
            .alias("cold"),
            # Visitors kick in Denver's air too. Altitude is a stadium fact.
            pl.when((pl.col("altitude") == 1.0) & (pl.col("indoor") == 0.0))
            .then(1.0)
            .otherwise(0.0)
            .alias("altitude"),
            pl.when(pl.col("play_type") == "field_goal")
            .then(pl.lit("fg"))
            .when(pl.col("play_type") == "punt")
            .then(pl.lit("punt"))
            .when(pl.col("play_type").is_in(["pass", "run"]))
            .then(pl.lit("go"))
            .otherwise(None)
            .alias("choice"),
        ]
    )


def pack_wp_features(
    score_diff: np.ndarray,
    game_seconds_remaining: np.ndarray,
    yardline_100: np.ndarray,
    down: np.ndarray,
    ydstogo: np.ndarray,
    timeouts_off: np.ndarray,
    timeouts_def: np.ndarray,
    is_home: np.ndarray,
    expected_margin: np.ndarray,
    receive_2h: np.ndarray,
    spread_missing: np.ndarray,
) -> np.ndarray:
    """Stack win-probability features in `WP_FEATURES` order.

    `score_time` is the lead divided by a slow function of time remaining.
    A 7-point lead in the first quarter and a 7-point lead with a minute left
    are different games; the tree can learn that from the raw inputs, and this
    column gives it the interaction directly. `field_time` does the same for
    field position: being on the opponent's side of the 50 is worth more with
    two minutes left than with a full half to play. `receive_2h` is zeroed
    once the second half has started so the model cannot keep paying a team
    for a kickoff that already happened.
    """
    gsr = np.clip(np.asarray(game_seconds_remaining, dtype=float), 0.0, 3600.0)
    score = np.asarray(score_diff, dtype=float)
    yl = np.clip(np.asarray(yardline_100, dtype=float), 1.0, 99.0)
    ytg = np.clip(np.asarray(ydstogo, dtype=float), 1.0, 40.0)
    receive = np.asarray(receive_2h, dtype=float)
    receive_feat = np.where(gsr > 1800.0, receive, 0.0)
    score_time = score / np.sqrt(gsr / 60.0 + 0.5)
    # Positive when the offense is in opponent territory. Dividing by a
    # function of time makes a short field more valuable late.
    field_time = (50.0 - yl) / np.sqrt(gsr / 60.0 + 0.5)
    goal = (yl <= ytg).astype(float)
    quarter = quarter_from_seconds(gsr)
    columns = [
        score,
        gsr,
        yl,
        np.asarray(down, dtype=float),
        ytg,
        np.clip(np.asarray(timeouts_off, dtype=float), 0.0, 3.0),
        np.clip(np.asarray(timeouts_def, dtype=float), 0.0, 3.0),
        np.asarray(is_home, dtype=float),
        np.asarray(expected_margin, dtype=float),
        receive_feat,
        quarter,
        score_time,
        goal,
        np.asarray(spread_missing, dtype=float),
        field_time,
    ]
    packed = np.column_stack(columns).astype(np.float32)
    if packed.shape[1] != len(WP_FEATURES):
        raise RuntimeError("feature pack does not match WP_FEATURES")
    return packed


def feature_frame_from_plays(plays: pl.DataFrame) -> np.ndarray:
    """Build the training matrix from a play-by-play frame that already has derived columns."""
    pdf = plays.select(
        [
            "score_diff",
            "game_seconds_remaining",
            "yardline_100",
            "down",
            "ydstogo",
            "timeouts_off",
            "timeouts_def",
            "is_home",
            "expected_margin",
            "receive_2h",
            "spread_missing",
        ]
    )
    cols = {name: pdf[name].to_numpy() for name in pdf.columns}
    return pack_wp_features(
        cols["score_diff"],
        cols["game_seconds_remaining"],
        cols["yardline_100"],
        cols["down"],
        cols["ydstogo"],
        cols["timeouts_off"],
        cols["timeouts_def"],
        cols["is_home"],
        cols["expected_margin"],
        cols["receive_2h"],
        cols["spread_missing"],
    )
