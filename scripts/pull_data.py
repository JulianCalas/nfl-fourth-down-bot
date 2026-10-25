"""Download nflverse play-by-play and cache the columns this project uses.

The full nflverse file has hundreds of columns, including nflfastR's own win
probability and expected points. Those are not model inputs. A slim parquet
is what `fourth_down.data.load_slim` reads, so a later run does not have to
hit the network again.

Usage:
    python scripts/pull_data.py
    python scripts/pull_data.py --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nflreadpy as nfl  # noqa: E402
import polars as pl  # noqa: E402

from fourth_down.config import FIRST_SEASON, LAST_FULL_SEASON, RAW_DIR, SLIM_COLUMNS, SLIM_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache slim nflverse play-by-play.")
    parser.add_argument("--refresh", action="store_true", help="Download again even if the parquet exists.")
    parser.add_argument("--first", type=int, default=FIRST_SEASON)
    parser.add_argument("--last", type=int, default=LAST_FULL_SEASON)
    args = parser.parse_args()
    if SLIM_PATH.exists() and not args.refresh:
        existing = pl.read_parquet(SLIM_PATH)
        print(f"Already cached {existing.height:,} rows at {SLIM_PATH}. Pass --refresh to replace it.")
        return

    seasons = list(range(args.first, args.last + 1))
    print(f"Downloading play-by-play for {seasons[0]}–{seasons[-1]}...")
    print(f"nflverse current season/week: {nfl.get_current_season()} week {nfl.get_current_week()}")
    raw = nfl.load_pbp(seasons)
    missing = [col for col in SLIM_COLUMNS if col not in raw.columns]
    if missing:
        print(f"Columns not in this nflverse release (filled with null): {missing}")
        raw = raw.with_columns([pl.lit(None).alias(col) for col in missing])
    slim = raw.select(SLIM_COLUMNS)
    # Drop preseason if it ever shows up. REG and POST stay.
    if "season_type" in slim.columns:
        slim = slim.filter(pl.col("season_type").is_in(["REG", "POST"]))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slim.write_parquet(SLIM_PATH)
    print(f"Wrote {slim.height:,} rows, {slim.width} columns, to {SLIM_PATH}")
    by_season = slim.group_by("season").agg(pl.len().alias("plays"), pl.col("week").max().alias("max_week")).sort("season")
    print(by_season)


if __name__ == "__main__":
    main()
