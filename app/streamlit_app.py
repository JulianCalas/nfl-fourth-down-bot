"""A small desk for one question: go, punt, or kick?

The app reads the committed models/current.joblib bundle and nothing else.
It does not download data and it does not need data/raw/pbp_slim.parquet.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fourth_down.config import MODEL_DIR  # noqa: E402
from fourth_down.conversion import ConversionModel  # noqa: E402
from fourth_down.decide import choice_win_probs  # noqa: E402
from fourth_down.field_goal import FieldGoalModel  # noqa: E402
from fourth_down.football import format_clock, format_yardline, kickoff_yardline_100  # noqa: E402
from fourth_down.situations import make_situations  # noqa: E402

st.set_page_config(page_title="Fourth Down", page_icon="🏈", layout="wide")

PRESETS = {
    "4th and 1, own 40, tied, early": {
        "our_score": 0,
        "their_score": 0,
        "quarter": 1,
        "minutes": 15,
        "seconds": 0,
        "territory": "Own",
        "yard_marker": 40,
        "ydstogo": 1,
        "favored_by": 0.0,
    },
    "4th and 6, opponent's 28, tied, early": {
        "our_score": 0,
        "their_score": 0,
        "quarter": 1,
        "minutes": 15,
        "seconds": 0,
        "territory": "Opponent",
        "yard_marker": 28,
        "ydstogo": 6,
        "favored_by": 0.0,
    },
    "4th and 10, own 12, tied, early": {
        "our_score": 0,
        "their_score": 0,
        "quarter": 1,
        "minutes": 12,
        "seconds": 0,
        "territory": "Own",
        "yard_marker": 12,
        "ydstogo": 10,
        "favored_by": 0.0,
    },
    "Down 3, 2:00 left, opponent's 35, 4th and 4": {
        "our_score": 17,
        "their_score": 20,
        "quarter": 4,
        "minutes": 2,
        "seconds": 0,
        "territory": "Opponent",
        "yard_marker": 35,
        "ydstogo": 4,
        "favored_by": 0.0,
    },
}


@st.cache_resource
def load_bundle() -> dict:
    """Load the committed model bundle. Local file only; no network."""
    path = MODEL_DIR / "current.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "models/current.joblib is missing. Run `python scripts/run_pipeline.py` first."
        )
    return joblib.load(path)


@st.cache_data
def score_situation(
    our_score: int,
    their_score: int,
    quarter: int,
    minutes: int,
    seconds: int,
    territory: str,
    yard_marker: int,
    ydstogo: int,
    timeouts_off: int,
    timeouts_def: int,
    favored_by: float,
    is_home: bool,
    receive_2h: bool,
    indoor: bool,
    wind: int,
    cold: bool,
    altitude: bool,
) -> dict:
    """Win probability of go, punt, and kick for one fourth-down spot."""
    bundle = load_bundle()
    meta = bundle["meta"]
    yl = yardline_100(territory, int(yard_marker))
    gsr = game_seconds(int(quarter), int(minutes), int(seconds))
    situation = make_situations(
        [
            {
                "score_diff": float(our_score - their_score),
                "game_seconds_remaining": gsr,
                "yardline_100": yl,
                "ydstogo": float(ydstogo),
                "timeouts_off": float(timeouts_off),
                "timeouts_def": float(timeouts_def),
                "is_home": 1.0 if is_home else 0.0,
                "expected_margin": float(favored_by),
                "receive_2h": 1.0 if receive_2h else 0.0,
                "spread_missing": 0.0,
                "season": float(meta["calibrate_season"]),
                "indoor": 1.0 if indoor else 0.0,
                "wind_mph": 0.0 if indoor else float(wind),
                "cold": 0.0 if indoor else (1.0 if cold else 0.0),
                "altitude": 1.0 if altitude and not indoor else 0.0,
            }
        ]
    )
    graded = choice_win_probs(situation, bundle["wp"], bundle["fg"], bundle["conv"], bundle["punt"])
    row = graded.row(0, named=True)
    distance = float(yl + 17)
    conv: ConversionModel = bundle["conv"]
    fg: FieldGoalModel = bundle["fg"]
    probs = conv.predict(
        situation["ydstogo"].to_numpy(),
        situation["yardline_100"].to_numpy(),
        situation["season"].to_numpy(),
    )
    p_make = (
        float(
            fg.predict(
                np.array([distance]),
                situation["indoor"].to_numpy(),
                situation["wind_mph"].to_numpy(),
                situation["cold"].to_numpy(),
                situation["altitude"].to_numpy(),
                situation["season"].to_numpy(),
            )[0]
        )
        if distance <= 68
        else 0.0
    )
    return {
        "recommended": str(row["recommended"]),
        "wp_go": float(row["wp_go"]),
        "wp_punt": float(row["wp_punt"]),
        "wp_fg": float(row["wp_fg"]),
        "wp_edge": float(row["wp_edge"]),
        "p_success": float(probs["p_td"][0] + probs["p_fd"][0]),
        "p_td": float(probs["p_td"][0]),
        "p_make": p_make,
        "distance": distance,
        "yardline_100": float(yl),
        "game_seconds_remaining": float(gsr),
        "calibrate_season": int(meta["calibrate_season"]),
        "train_first": int(meta["train_seasons"][0]),
        "train_last": int(meta["train_seasons"][-1]),
    }


def yardline_100(territory: str, marker: int) -> float:
    """Convert a broadcast yard marker into nflverse yardline_100."""
    if territory == "Own":
        return float(100 - marker) if marker != 50 else 50.0
    return float(marker)


def game_seconds(quarter: int, minutes: int, seconds: int) -> float:
    remaining_in_quarter = minutes * 60 + seconds
    # Quarters left after this one, plus the time still on this quarter's clock.
    return float((4 - quarter) * 15 * 60 + remaining_in_quarter)


def _apply_preset() -> None:
    """Copy a named spot into the widgets. Custom leaves the current numbers alone."""
    name = st.session_state.get("preset_name", "Custom")
    if name not in PRESETS:
        return
    spot = PRESETS[name]
    st.session_state.our_score = int(spot["our_score"])
    st.session_state.their_score = int(spot["their_score"])
    st.session_state.quarter = int(spot["quarter"])
    st.session_state.minutes = int(spot["minutes"])
    st.session_state.seconds = int(spot["seconds"])
    st.session_state.territory = spot["territory"]
    st.session_state.yard_marker = int(spot["yard_marker"])
    st.session_state.ydstogo = int(spot["ydstogo"])
    st.session_state.favored_by = float(spot["favored_by"])


def _init_controls() -> None:
    """First visit opens on a real fourth-down spot, not an empty custom form."""
    if "preset_name" in st.session_state:
        return
    st.session_state.preset_name = "4th and 1, own 40, tied, early"
    st.session_state.timeouts_off = 3
    st.session_state.timeouts_def = 3
    st.session_state.is_home = True
    st.session_state.receive_2h = True
    st.session_state.indoor = False
    st.session_state.wind = 0
    st.session_state.cold = False
    st.session_state.altitude = False
    _apply_preset()


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; max-width: 1100px;}
        div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Fourth down")
    st.caption(
        "Win probability of going for it, punting, and kicking. "
        "The recommendation is whichever option leaves the offense with the higher chance of winning the game."
    )

    try:
        load_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    _init_controls()
    with st.sidebar:
        st.header("Situation")
        st.selectbox(
            "Start from a spot",
            ["Custom"] + list(PRESETS),
            key="preset_name",
            on_change=_apply_preset,
        )
        our_score = st.number_input("Our score", 0, 60, key="our_score")
        their_score = st.number_input("Their score", 0, 60, key="their_score")
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], key="quarter")
        clock_cols = st.columns(2)
        minutes = clock_cols[0].number_input("Minutes", 0, 15, key="minutes")
        seconds = clock_cols[1].number_input("Seconds", 0, 59, key="seconds")
        territory = st.radio("Ball is on", ["Own", "Opponent"], horizontal=True, key="territory")
        yard_marker = st.slider("Yard marker", 1, 50, key="yard_marker")
        ydstogo = st.slider("Yards to go", 1, 20, key="ydstogo")
        timeouts_off = st.slider("Our timeouts", 0, 3, key="timeouts_off")
        timeouts_def = st.slider("Their timeouts", 0, 3, key="timeouts_def")
        favored_by = st.slider(
            "We are favored by (points, negative if underdog)",
            -14.0,
            14.0,
            step=0.5,
            key="favored_by",
        )
        is_home = st.toggle("We are the home team", key="is_home")
        receive_2h = st.toggle("We receive the second-half kickoff", key="receive_2h")
        st.divider()
        st.caption("Kick conditions")
        indoor = st.toggle("Dome or closed roof", key="indoor")
        wind = st.slider("Wind (mph)", 0, 30, key="wind", disabled=indoor)
        cold = st.toggle("At or below 40°F, outdoors", key="cold", disabled=indoor)
        altitude = st.toggle("Denver", key="altitude")

    scored = score_situation(
        int(our_score),
        int(their_score),
        int(quarter),
        int(minutes),
        int(seconds),
        str(territory),
        int(yard_marker),
        int(ydstogo),
        int(timeouts_off),
        int(timeouts_def),
        float(favored_by),
        bool(is_home),
        bool(receive_2h),
        bool(indoor),
        int(wind),
        bool(cold),
        bool(altitude),
    )
    yl = scored["yardline_100"]
    gsr = scored["game_seconds_remaining"]
    distance = scored["distance"]
    label = {
        "go": "Go for it",
        "punt": "Punt",
        "fg": "Kick the field goal",
    }[scored["recommended"]]
    st.subheader(label)
    st.write(
        f"4th and {ydstogo:.0f} from {format_yardline(yl)}, {format_clock(gsr)}. "
        f"The best option is worth **{scored['wp_edge']:.1%}** more win probability than the next one."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Go for it", f"{scored['wp_go']:.1%}")
    c2.metric("Punt", f"{scored['wp_punt']:.1%}")
    c3.metric("Field goal", f"{scored['wp_fg']:.1%}" if distance <= 68 else "Out of range")

    chart = go.Figure(
        go.Bar(
            x=["Go for it", "Punt", "Field goal"],
            y=[scored["wp_go"], scored["wp_punt"], scored["wp_fg"] if distance <= 68 else 0],
            marker_color=["#2E7D4F", "#1F4E79", "#C48A1A"],
            text=[
                f"{scored['wp_go']:.1%}",
                f"{scored['wp_punt']:.1%}",
                f"{scored['wp_fg']:.1%}" if distance <= 68 else "—",
            ],
            textposition="outside",
        )
    )
    chart.update_layout(
        yaxis_title="Win probability",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#FBF8F3",
        paper_bgcolor="white",
    )
    st.plotly_chart(chart, width="stretch")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("**What the pieces say**")
        st.write(
            f"League-average conversion chance: **{scored['p_success']:.0%}**, "
            f"including a **{scored['p_td']:.0%}** chance the play is a touchdown."
        )
        if distance <= 68:
            st.write(
                f"A field goal from here is **{distance:.0f}** yards and goes in about **{scored['p_make']:.0%}** of the time."
            )
        else:
            st.write(f"A field goal would be **{distance:.0f}** yards, past the range this model will recommend.")
        touchback_yl = kickoff_yardline_100(int(scored["calibrate_season"]))
        st.write(
            f"After a score, the other team is spotted at their own {100 - touchback_yl:.0f} "
            "on a touchback, using that season's kickoff rule."
        )
    with detail_right:
        st.markdown("**How to read this**")
        st.write(
            "Each bar is the offense's chance of winning the game if they pick that option, "
            "before the play happens. It is not the chance the play itself works."
        )
        st.write(
            "Conversion and kicking rates are league averages, not this team's. "
            "A strong offense should go a little more often than the chart says; a shaky kicking game should kick a little less."
        )
        st.write(
            f"Trees trained on {scored['train_first']}–{scored['train_last']}, "
            f"probabilities scaled on {scored['calibrate_season']}. "
            "Coach grades in the README use a stricter out-of-season model."
        )


if __name__ == "__main__":
    main()
else:
    main()
