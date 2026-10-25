"""Football geometry and small decision rules.

These functions encode rules and definitions. They do not fit anything.
Keeping them separate makes the models easier to test: a missed field goal
should land on the right yard line even if every model file is deleted.
"""

from __future__ import annotations

import numpy as np

# The holder usually places the ball 7 or 8 yards behind the line of scrimmage.
# Eight yards matched next-snap field position on misses within about 1.5 yards
# in a check of the 2023-24 seasons (see the data notes in field_goal.py).
KICK_SPOT_BEHIND_LOS = 8.0

CHOICES = ("go", "punt", "fg")


def kickoff_yardline_100(season: int) -> float:
    """Yard line where a kickoff touchback starts the receiving team.

    Returned in the nflverse `yardline_100` convention: yards the receiving
    team still needs for a touchdown.

    The number is a rule, not a fitted parameter, so it is known before the
    season and is safe to use when grading that season.
    """
    return float(kickoff_yardline_array(season)[()])


def kickoff_yardline_array(season: np.ndarray | int) -> np.ndarray:
    """Vector version of `kickoff_yardline_100`."""
    seasons = np.asarray(season)
    yardline = np.full(seasons.shape, 70.0, dtype=float)
    yardline = np.where(seasons <= 2023, 75.0, yardline)
    yardline = np.where(seasons <= 2015, 80.0, yardline)
    return yardline


def missed_fg_opponent_yardline(yardline_100: float, behind: float = KICK_SPOT_BEHIND_LOS) -> float:
    """Opponent's yardline_100 after a missed (not returned) field goal.

    The defense takes over where the ball was kicked. If that spot is inside
    the kicking team's own 20, the ball is placed at the 20 instead.

    Example: line of scrimmage at the opponent's 30 (`yardline_100` = 30).
    The kick is from the opponent's 38. The opponent is 62 yards from a
    touchdown, so the function returns 62.
    """
    yl = np.asarray(yardline_100, dtype=float)
    spot_from_own_goal = 100.0 - yl - behind
    opp = 100.0 - (yl + behind)
    # Inside-the-20 rule. spot_from_own_goal < 20 means the kick spot is
    # inside our own 20, and the opponent is handed the ball at our 20,
    # which is yardline_100 = 80 from their point of view.
    opp = np.where(spot_from_own_goal < 20.0, 80.0, opp)
    opp = np.clip(opp, 1.0, 99.0)
    if np.isscalar(yardline_100):
        return float(opp)
    return opp


def format_yardline(yardline_100: float) -> str:
    """Turn nflverse yardline_100 into a phrase a coach would say."""
    yl = float(yardline_100)
    if abs(yl - 50) < 0.5:
        return "midfield"
    if yl < 50:
        return f"the opponent's {yl:.0f}"
    return f"their own {100 - yl:.0f}"


def format_clock(game_seconds_remaining: float) -> str:
    """Game clock and quarter from seconds left in the whole game."""
    gsr = int(max(0, round(float(game_seconds_remaining))))
    if gsr > 2700:
        quarter, sec = 1, gsr - 2700
    elif gsr > 1800:
        quarter, sec = 2, gsr - 1800
    elif gsr > 900:
        quarter, sec = 3, gsr - 900
    else:
        quarter, sec = 4, gsr
    return f"Q{quarter} {sec // 60}:{sec % 60:02d}"


def quarter_from_seconds(game_seconds_remaining: np.ndarray | float) -> np.ndarray:
    """Quarter implied by seconds remaining in the game.

    3600 is kickoff. 1800 is the start of the third quarter. The comparison
    is strict at the boundaries so 1800 maps to Q3 and 900 maps to Q4.
    """
    gsr = np.asarray(game_seconds_remaining, dtype=float)
    qtr = np.ones(gsr.shape, dtype=float)
    qtr = np.where(gsr > 2700, 1.0, qtr)
    qtr = np.where((gsr <= 2700) & (gsr > 1800), 2.0, qtr)
    qtr = np.where((gsr <= 1800) & (gsr > 900), 3.0, qtr)
    qtr = np.where(gsr <= 900, 4.0, qtr)
    return qtr


def terminal_win_prob(score_diff: np.ndarray | float) -> np.ndarray:
    """Win probability when the game is over. Ties are one half."""
    diff = np.asarray(score_diff, dtype=float)
    out = np.full(diff.shape, 0.5, dtype=float)
    out = np.where(diff > 0, 1.0, out)
    out = np.where(diff < 0, 0.0, out)
    if np.isscalar(score_diff):
        return float(out)
    return out


def pick_recommendation(wp_go: float, wp_punt: float, wp_fg: float) -> tuple[str, float]:
    """Return the best choice and how far it sits above the runner-up.

    The edge is in win-probability points, not in expected points. A value
    of 0.03 means the best option is worth 3 extra percentage points of win
    probability versus the next-best option.
    """
    scores = {"go": float(wp_go), "punt": float(wp_punt), "fg": float(wp_fg)}
    ordered = sorted(scores, key=scores.get, reverse=True)
    best = ordered[0]
    edge = scores[best] - scores[ordered[1]]
    return best, edge


def wp_left_on_table(wp_go: float, wp_punt: float, wp_fg: float, actual: str) -> float:
    """Win probability the actual choice gave up relative to the best choice.

    Zero when the coach picked the model's recommendation. Never negative:
    we do not credit a coach for beating the model on a single play, because
    the model is the decision rule being applied, not a forecast of luck.
    """
    scores = {"go": float(wp_go), "punt": float(wp_punt), "fg": float(wp_fg)}
    if actual not in scores:
        raise ValueError(f"actual choice must be one of {CHOICES}, got {actual!r}")
    return float(max(scores.values()) - scores[actual])


def logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    """Log-odds. Clipped so a 0 or 1 probability does not become infinity."""
    arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def expit(x: np.ndarray | float) -> np.ndarray:
    """Logistic function. Stable for large positive or negative inputs."""
    z = np.asarray(x, dtype=float)
    out = np.empty(z.shape, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    if np.isscalar(x):
        return float(out)
    return out
