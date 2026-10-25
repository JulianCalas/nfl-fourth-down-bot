"""Fit the models and write results/, figures/, and models/current.joblib.

Usage:
    python scripts/pull_data.py
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --quick   # debug only; do not quote these numbers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourth_down.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fourth-down study.")
    parser.add_argument("--quick", action="store_true", help="Short debug fit. Not the published model.")
    args = parser.parse_args()
    run_pipeline(quick=args.quick)


if __name__ == "__main__":
    main()
