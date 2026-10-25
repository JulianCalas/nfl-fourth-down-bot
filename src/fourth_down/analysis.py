"""Turn graded fourth downs into coach and team tables.

The aggregations are SQL on purpose. The modeling code decides what each
play was worth. DuckDB then does the grouping a hiring manager would expect
to see written as a query: rates, means, and a standard error.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from fourth_down.config import CLEAR_EDGE, MIN_HALF_SECONDS, SQL_DIR, WP_GARBAGE_HIGH, WP_GARBAGE_LOW
from fourth_down.football import format_clock, format_yardline


def annotate(decisions: pl.DataFrame) -> pl.DataFrame:
    """Flag the plays that belong in the coach study.

    Garbage time is dropped with the pre-snap win probability, not with a
    score cutoff. Down 14 with a minute left is not garbage. Up 24 in the
    first quarter is not an interesting fourth down either, and the win
    probability already says so.
    """
    return decisions.with_columns(
        [
            (
                (pl.col("half_seconds_remaining") >= MIN_HALF_SECONDS)
                & (pl.col("ydstogo") <= 20)
                & (pl.col("wp_pre") >= WP_GARBAGE_LOW)
                & (pl.col("wp_pre") <= WP_GARBAGE_HIGH)
            ).alias("in_leaderboard"),
            (pl.col("choice") == "go").alias("went_for_it"),
            (pl.col("wp_edge") >= CLEAR_EDGE).alias("clear_call"),
            (
                (pl.col("ydstogo") == 1)
                & pl.col("yardline_100").is_between(40, 80)
            ).alias("fourth_and_one_between_40s"),
        ]
    ).with_columns(
        (
            pl.col("in_leaderboard")
            & (pl.col("recommended") == "go")
            & (pl.col("wp_edge") >= CLEAR_EDGE)
        ).alias("should_go")
    )


def run_sql(decisions: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Execute the coach, team, and trend queries."""
    con = duckdb.connect()
    con.register("decisions", decisions.to_arrow())
    tables = {}
    for name in ("team_season", "coach_season", "coach_pooled", "team_pooled", "trend_by_year"):
        query = (SQL_DIR / f"{name}.sql").read_text()
        tables[name] = pl.from_pandas(con.execute(query).df())
    con.close()
    return tables


def pick_examples(decisions: pl.DataFrame, limit_each: int = 1) -> list[dict]:
    """A few real plays that show the decision rule at its loudest.

    These are not chosen because they are famous. They are the largest
    out-of-sample gaps in the graded file, plus one high-leverage call the
    coach got right. The play result is attached afterwards so a reader can
    see what happened. The model did not use that result.
    """
    pool = decisions.filter(pl.col("in_leaderboard") & pl.col("desc").is_not_null())
    specs = [
        ("costliest_miss", pool.filter(~pl.col("followed") & pl.col("clear_call")), "wp_left"),
        (
            "punted_instead_of_going",
            pool.filter((pl.col("choice") == "punt") & (pl.col("recommended") == "go") & pl.col("clear_call")),
            "wp_left",
        ),
        (
            "kicked_instead_of_going",
            pool.filter((pl.col("choice") == "fg") & (pl.col("recommended") == "go") & pl.col("clear_call")),
            "wp_left",
        ),
        (
            "went_instead_of_kicking_or_punting",
            pool.filter((pl.col("choice") == "go") & (pl.col("recommended") != "go") & pl.col("clear_call")),
            "wp_left",
        ),
        (
            "right_call_high_leverage",
            pool.filter(pl.col("followed") & (pl.col("recommended") == "go") & pl.col("clear_call")),
            "wp_edge",
        ),
    ]
    seen: set[tuple] = set()
    examples = []
    for kind, frame, rank_col in specs:
        if frame.height == 0:
            continue
        ranked = frame.sort(rank_col, descending=True)
        taken = 0
        for row in ranked.iter_rows(named=True):
            key = (row.get("game_id"), row.get("play_id"))
            if key in seen:
                continue
            seen.add(key)
            examples.append(_example_record(kind, row))
            taken += 1
            if taken >= limit_each:
                break
    return examples


def _example_record(kind: str, row: dict) -> dict:
    score = (
        f"{row['posteam']} { _num(row.get('posteam_score')) }, "
        f"{row['defteam']} { _num(row.get('defteam_score')) }"
    )
    choice = {"go": "went for it", "punt": "punted", "fg": "kicked a field goal"}[row["choice"]]
    recommended = {"go": "go for it", "punt": "punt", "fg": "kick the field goal"}[row["recommended"]]
    sentence = (
        f"{row['coach']} ({row['posteam']}) {choice} on 4th and {row['ydstogo']:.0f} "
        f"from {format_yardline(row['yardline_100'])}, {format_clock(row['game_seconds_remaining'])}. "
        f"Score at the snap: {score}. "
        f"The model valued go at {row['wp_go']:.1%}, punt at {row['wp_punt']:.1%}, "
        f"and a field goal at {row['wp_fg']:.1%}, so the recommendation was to {recommended}."
    )
    if row["choice"] != row["recommended"]:
        sentence += f" That choice left {row['wp_left']:.1%} win probability on the table."
    else:
        sentence += f" The edge over the next-best option was {row['wp_edge']:.1%}."
    return {
        "kind": kind,
        "sentence": sentence,
        "season": int(row["season"]),
        "week": _maybe_int(row.get("week")),
        "game_id": row.get("game_id"),
        "game_date": str(row.get("game_date")),
        "posteam": row.get("posteam"),
        "defteam": row.get("defteam"),
        "coach": row.get("coach"),
        "qtr": _maybe_int(row.get("qtr")),
        "time": row.get("time"),
        "ydstogo": float(row["ydstogo"]),
        "yardline_100": float(row["yardline_100"]),
        "yardline_phrase": format_yardline(row["yardline_100"]),
        "score_diff": float(row["score_diff"]),
        "posteam_score": _num(row.get("posteam_score")),
        "defteam_score": _num(row.get("defteam_score")),
        "choice": row["choice"],
        "recommended": row["recommended"],
        "wp_go": float(row["wp_go"]),
        "wp_punt": float(row["wp_punt"]),
        "wp_fg": float(row["wp_fg"]),
        "wp_edge": float(row["wp_edge"]),
        "wp_left": float(row["wp_left"]),
        "wp_pre": float(row["wp_pre"]),
        "desc": str(row.get("desc") or "")[:400],
        "what_happened": _what_happened(row),
    }


def _what_happened(row: dict) -> str:
    choice = row["choice"]
    if choice == "fg":
        return f"The kick was {row.get('field_goal_result') or 'unknown'}."
    if choice == "go":
        converted = row.get("fourth_down_converted")
        yards = row.get("yards_gained")
        if row.get("touchdown") == 1 and row.get("td_team") == row.get("posteam"):
            return f"They scored a touchdown (gained {_num(yards)} yards)."
        if converted == 1:
            return f"They converted (gained {_num(yards)} yards)."
        return f"They did not convert (gained {_num(yards)} yards)."
    if row.get("transition_ok") and row.get("n_yardline_100") is not None:
        return (
            "The next snap was at "
            f"yardline_100 {float(row['n_yardline_100']):.0f} "
            f"for {row.get('n_posteam')}."
        )
    return "The punt result is the next snap in the play-by-play."


def write_examples(path: Path, examples: list[dict]) -> None:
    path.write_text(json.dumps(examples, indent=2))


def _num(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def _maybe_int(value):
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return value
