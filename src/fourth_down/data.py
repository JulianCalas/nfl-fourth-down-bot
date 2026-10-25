"""Load the cached play-by-play and attach the next scrimmage snap.

The next snap is how we see what a punt, a miss, or a conversion actually
produced: who has the ball, where, and what the score is. Penalty-only
`no_play` rows are not scrimmage snaps, so they are skipped and the replayed
down is the next state.
"""

from __future__ import annotations

import polars as pl

from fourth_down.config import SLIM_PATH
from fourth_down.features import with_model_columns

SCRIMMAGE_TYPES = ["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike"]


def load_slim() -> pl.DataFrame:
    """Read the slim parquet written by scripts/pull_data.py."""
    if not SLIM_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SLIM_PATH}. Run `python scripts/pull_data.py` first."
        )
    return pl.read_parquet(SLIM_PATH)


def attach_next_scrimmage(plays: pl.DataFrame) -> pl.DataFrame:
    """For every play, find the next snap that has a down.

    `join_asof(..., strategy='forward')` returns the first later scrimmage
    snap in the same game. Play ids are not consecutive (a field goal at id
    823 is followed by a kickoff at 847), so the lookup key is `play_id + 1`
    and the match is "first scrimmage at or after that key."
    """
    ordered = plays.sort(["game_id", "play_id"])
    scrim = (
        ordered.filter(
            pl.col("down").is_between(1, 4)
            & pl.col("play_type").is_in(SCRIMMAGE_TYPES)
            & pl.col("posteam").is_not_null()
            & pl.col("yardline_100").is_not_null()
        )
        .select(
            [
                "game_id",
                pl.col("play_id").alias("n_play_id"),
                pl.col("posteam").alias("n_posteam"),
                pl.col("yardline_100").alias("n_yardline_100"),
                pl.col("score_differential").alias("n_score_differential"),
                pl.col("game_seconds_remaining").alias("n_gsr"),
                pl.col("qtr").alias("n_qtr"),
                pl.col("down").alias("n_down"),
                pl.col("ydstogo").alias("n_ydstogo"),
            ]
        )
        .sort(["game_id", "n_play_id"])
    )
    keyed = ordered.with_columns((pl.col("play_id") + 1).alias("_pid_key")).sort(
        ["game_id", "_pid_key"]
    )
    joined = keyed.join_asof(
        scrim,
        left_on="_pid_key",
        right_on="n_play_id",
        by="game_id",
        strategy="forward",
    ).drop("_pid_key")
    # Score differential on the next snap is from the *next* offense's view.
    # Flip it when the ball changed hands so the column is always "our score
    # minus their score" for the team that just ran the play.
    return joined.with_columns(
        [
            pl.when(pl.col("n_posteam") == pl.col("posteam"))
            .then(pl.col("n_score_differential"))
            .when(pl.col("n_posteam") == pl.col("defteam"))
            .then(-pl.col("n_score_differential"))
            .otherwise(None)
            .alias("n_score_for_us"),
            (pl.col("n_posteam") == pl.col("posteam")).alias("kept_ball"),
            (pl.col("game_seconds_remaining") - pl.col("n_gsr")).alias("elapsed"),
        ]
    ).with_columns(
        (
            pl.col("n_yardline_100").is_not_null()
            & pl.col("n_posteam").is_not_null()
            & pl.col("n_score_for_us").is_not_null()
            & (pl.col("n_qtr") == pl.col("qtr"))
            & (pl.col("elapsed") >= 0)
            & (pl.col("elapsed") <= 40)
        ).alias("transition_ok")
    )


def prepare_plays(raw: pl.DataFrame) -> pl.DataFrame:
    """Regulation and playoff snaps with derived columns and the next snap attached."""
    plays = raw.filter(pl.col("season_type").is_in(["REG", "POST"]))
    # A missing timeout count is rare and usually a bad row. Filling with 3
    # (a full set) avoids dropping an otherwise usable snap. The pipeline logs
    # how often this happens.
    plays = plays.with_columns(
        [
            pl.col("posteam_timeouts_remaining").fill_null(3),
            pl.col("defteam_timeouts_remaining").fill_null(3),
            pl.col("home_opening_kickoff").fill_null(0),
        ]
    )
    plays = attach_next_scrimmage(plays)
    return with_model_columns(plays)


def scrimmage_mask() -> pl.Expr:
    """Plays a win-probability model can learn from: a real down, regulation only."""
    return (
        pl.col("play_type").is_in(SCRIMMAGE_TYPES)
        & pl.col("down").is_between(1, 4)
        & pl.col("qtr").is_between(1, 4)
        & pl.col("posteam").is_not_null()
        & pl.col("yardline_100").is_between(1, 99)
        & pl.col("ydstogo").is_not_null()
        & (pl.col("ydstogo") > 0)
        & pl.col("score_diff").is_not_null()
        & pl.col("game_seconds_remaining").is_between(0, 3600)
        & pl.col("posteam_win").is_not_null()
    )


def fourth_down_choices(plays: pl.DataFrame) -> pl.DataFrame:
    """Fourth downs where the coach actually punted, kicked, or went for it.

    `no_play` rows are excluded. Those are penalties, and the choice shows up
    again on the replayed snap, so keeping them would double-count.
    """
    return plays.filter(
        (pl.col("down") == 4)
        & pl.col("qtr").is_between(1, 4)
        & pl.col("choice").is_not_null()
        & pl.col("yardline_100").is_between(1, 99)
        & pl.col("ydstogo").is_between(1, 40)
        & pl.col("score_diff").is_not_null()
        & pl.col("half_seconds_remaining").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("coach").is_not_null()
    )
