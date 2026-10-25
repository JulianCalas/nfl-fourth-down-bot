"""Fit every component, grade every fourth down, and write the artifacts.

Historical grades for season T use a win-probability model trained on
seasons through T-2 and Platt-scaled on T-1. Field-goal, conversion, and
punt models for that grade are fit on every season before T. Season T is
not in any of those fits.

The chart and the Streamlit app use a separate "current" bundle: trees
trained through the penultimate season, calibrated on the last full season,
with the other components fit on every season. That bundle is for "what
should you do on Sunday." It is not the bundle that grades old coaches.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from fourth_down.analysis import annotate, pick_examples, run_sql, write_examples
from fourth_down.config import (
    FIGURES_DIR,
    FIRST_SEASON,
    FIRST_TEST_SEASON,
    LAST_FULL_SEASON,
    MODEL_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    WP_MIN_CHILD_WEIGHT,
    WP_TREES,
)
from fourth_down.conversion import ConversionModel
from fourth_down.data import fourth_down_choices, load_slim, prepare_plays, scrimmage_mask
from fourth_down.decide import choice_win_probs, pre_snap_win_prob
from fourth_down.features import feature_frame_from_plays
from fourth_down.field_goal import FieldGoalModel
from fourth_down.metrics import (
    binary_log_loss,
    brier_score,
    calibration_table,
    mean_absolute_calibration_error,
    season_folds,
)
from fourth_down.plots import (
    plot_bars,
    plot_calibration,
    plot_conversion,
    plot_decision_chart,
    plot_fg_curve,
    plot_logloss,
    plot_trend,
    plot_vs_nflfastr,
)
from fourth_down.punt import PuntModel
from fourth_down.situations import decision_grid, make_situations
from fourth_down.wp import WPModel

np.random.seed(RANDOM_SEED)


def run_pipeline(quick: bool = False) -> dict:
    """Run the full study. `--quick` is a debug pass and is not the published model."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading play-by-play...")
    raw = load_slim()
    print(f"  slim rows: {raw.height:,}  seasons: {raw['season'].min()}–{raw['season'].max()}")
    plays = prepare_plays(raw)
    if quick:
        plays = plays.filter(pl.col("season") >= 2021)
        first_test = 2024
        n_estimators = 40
        min_child = 10.0
        print("  QUICK MODE: seasons from 2021, 40 trees. Do not publish these numbers.")
    else:
        first_test = FIRST_TEST_SEASON
        n_estimators = WP_TREES
        min_child = WP_MIN_CHILD_WEIGHT

    seasons = sorted(plays["season"].unique().to_list())
    folds = season_folds(seasons, first_test)
    print(f"  prepared rows: {plays.height:,}  test folds: {[f['test'] for f in folds]}")

    metric_rows = []
    reliability_parts = []
    scatter_parts = []
    fg_parts = []
    conv_parts = []
    decision_parts = []

    for fold in folds:
        test = fold["test"]
        print(f"\n=== Grading {test}  (train {fold['train'][0]}–{fold['train'][-1]}, calibrate {fold['calibrate']}) ===")
        wp_model = WPModel().fit(
            plays,
            train_seasons=fold["train"],
            calibrate_season=fold["calibrate"],
            n_estimators=n_estimators,
            min_child_weight=min_child,
        )
        print(
            f"  WP trees used: {wp_model.best_iteration}  "
            f"cal log loss raw {wp_model.cal_log_loss_raw:.4f} → "
            f"calibrated {wp_model.cal_log_loss_calibrated:.4f}  "
            f"Platt slope {wp_model.platt.slope:.3f}"
        )
        history = plays.filter(pl.col("season") < test)
        fg_model = FieldGoalModel().fit(history)
        conv_model = ConversionModel().fit(history)
        punt_model = PuntModel().fit(history)
        print(
            f"  components: {fg_model.n_kicks} kicks, {conv_model.transitions.n_go} goes, "
            f"{punt_model.n_punts} punts"
        )

        test_plays = plays.filter(scrimmage_mask() & (pl.col("season") == test))
        metric_rows.append(_wp_metrics(test, test_plays, wp_model))
        reliability_parts.append(_reliability_frame(test_plays, wp_model))
        scatter_parts.append(_scatter_frame(test_plays, wp_model))

        graded = _grade_fourth_downs(plays.filter(pl.col("season") == test), wp_model, fg_model, conv_model, punt_model)
        decision_parts.append(graded)
        fg_parts.append(_fg_rows(plays.filter(pl.col("season") == test), fg_model))
        conv_parts.append(_conv_rows(plays.filter(pl.col("season") == test), conv_model))
        print(
            f"  held-out log loss {metric_rows[-1]['logloss_ours']:.4f} "
            f"(nflfastR {metric_rows[-1]['logloss_nflfastr']:.4f}, "
            f"vegas {metric_rows[-1]['logloss_vegas']:.4f})  "
            f"fourth downs {graded.height}"
        )

    decisions = annotate(pl.concat(decision_parts, how="diagonal_relaxed"))
    metrics = pl.DataFrame(metric_rows).sort("season")
    reliability = pl.concat(reliability_parts)
    scatter = pl.concat(scatter_parts)
    fg_oos = pl.concat(fg_parts) if fg_parts else pl.DataFrame()
    conv_oos = pl.concat(conv_parts) if conv_parts else pl.DataFrame()

    print("\n=== Current bundle for the chart and the app ===")
    # Trees train on every season except the last one. The last season is the
    # early-stopping and Platt season, so it never grows a tree. Component
    # models below use every season, including that last one: they are the
    # Sunday estimates, not the historical grades.
    cal_season = int(max(seasons))
    tree_seasons = [int(s) for s in seasons if int(s) < cal_season]
    wp_current = WPModel().fit(
        plays,
        train_seasons=tree_seasons,
        calibrate_season=cal_season,
        n_estimators=n_estimators,
        min_child_weight=min_child,
    )
    if wp_current.calibrate_season in set(wp_current.train_seasons):
        raise RuntimeError("current win-probability model saw its calibration season")
    fg_current = FieldGoalModel().fit(plays)
    conv_current = ConversionModel().fit(plays)
    punt_current = PuntModel().fit(plays)
    bundle = {
        "wp": wp_current,
        "fg": fg_current,
        "conv": conv_current,
        "punt": punt_current,
        "meta": {
            "train_seasons": wp_current.train_seasons,
            "calibrate_season": wp_current.calibrate_season,
            "component_seasons": seasons,
            "quick": quick,
            "first_season": FIRST_SEASON,
            "last_full_season": LAST_FULL_SEASON,
        },
    }
    joblib.dump(bundle, MODEL_DIR / "current.joblib")
    print(f"  saved {MODEL_DIR / 'current.joblib'}")

    sanity = _sanity(bundle)
    for row in sanity:
        flag = "OK" if row["ok"] else "CHECK"
        print(
            f"  [{flag}] {row['name']}: recommended {row['recommended']} "
            f"(go {row['wp_go']:.3f}, punt {row['wp_punt']:.3f}, fg {row['wp_fg']:.3f})"
        )

    tables = run_sql(decisions)
    examples = pick_examples(decisions)
    _write_tables(decisions, metrics, tables, examples, fg_current, conv_current, wp_current, fg_oos, conv_oos)
    _write_figures(bundle, metrics, reliability, scatter, tables, fg_oos, sanity)
    summary = _summary(
        plays, decisions, metrics, tables, examples, sanity, wp_current, fg_current, conv_current, punt_current, quick
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"\nWrote {RESULTS_DIR / 'summary.json'}")
    return summary


def _grade_fourth_downs(plays, wp_model, fg_model, conv_model, punt_model) -> pl.DataFrame:
    fourth = fourth_down_choices(plays)
    graded = choice_win_probs(fourth, wp_model, fg_model, conv_model, punt_model)
    graded = graded.with_columns(pl.Series("wp_pre", pre_snap_win_prob(graded, wp_model)))
    return graded


def _wp_metrics(season: int, test_plays: pl.DataFrame, wp_model: WPModel) -> dict:
    features = feature_frame_from_plays(test_plays)
    ours = wp_model.predict(features)
    raw = wp_model.predict_raw(features)
    y = test_plays["posteam_win"].to_numpy().astype(float)
    nfl = test_plays["wp"].to_numpy().astype(float)
    vegas = test_plays["vegas_wp"].to_numpy().astype(float)
    comparable = np.isfinite(nfl) & np.isfinite(vegas) & np.isfinite(ours)
    y_c, ours_c, raw_c, nfl_c, vegas_c = y[comparable], ours[comparable], raw[comparable], nfl[comparable], vegas[comparable]
    table = calibration_table(y_c, ours_c, n_bins=10)
    return {
        "season": int(season),
        "n": int(comparable.sum()),
        "logloss_ours": binary_log_loss(y_c, ours_c),
        "logloss_raw": binary_log_loss(y_c, raw_c),
        "logloss_nflfastr": binary_log_loss(y_c, nfl_c),
        "logloss_vegas": binary_log_loss(y_c, vegas_c),
        "brier_ours": brier_score(y_c, ours_c),
        "brier_nflfastr": brier_score(y_c, nfl_c),
        "brier_vegas": brier_score(y_c, vegas_c),
        "calibration_mae": mean_absolute_calibration_error(table),
        "corr_nflfastr": float(np.corrcoef(ours_c, nfl_c)[0, 1]),
        "corr_vegas": float(np.corrcoef(ours_c, vegas_c)[0, 1]),
    }


def _reliability_frame(test_plays: pl.DataFrame, wp_model: WPModel) -> pl.DataFrame:
    features = feature_frame_from_plays(test_plays)
    return pl.DataFrame(
        {
            "y": test_plays["posteam_win"].to_numpy().astype(float),
            "p": wp_model.predict(features),
        }
    )


def _scatter_frame(test_plays: pl.DataFrame, wp_model: WPModel) -> pl.DataFrame:
    features = feature_frame_from_plays(test_plays)
    frame = pl.DataFrame(
        {
            "our_wp": wp_model.predict(features),
            "nflfastr_wp": test_plays["wp"].to_numpy().astype(float),
        }
    ).drop_nulls()
    if frame.height > 1500:
        frame = frame.sample(n=1500, seed=RANDOM_SEED)
    return frame


def _fg_rows(plays: pl.DataFrame, fg_model: FieldGoalModel) -> pl.DataFrame:
    kicks = plays.filter(
        (pl.col("play_type") == "field_goal")
        & pl.col("field_goal_result").is_in(["made", "missed", "blocked"])
        & pl.col("kick_distance").is_between(18, 76)
        & pl.col("qtr").is_between(1, 4)
    )
    if kicks.height == 0:
        return pl.DataFrame({"distance": [], "made": [], "p_hat": []})
    distance = kicks["kick_distance"].to_numpy().astype(float)
    made = (kicks["field_goal_result"] == "made").to_numpy().astype(float)
    p_hat = fg_model.predict(
        distance,
        kicks["indoor"].to_numpy(),
        kicks["wind_mph"].to_numpy(),
        kicks["cold"].to_numpy(),
        kicks["altitude"].to_numpy(),
        kicks["season"].to_numpy(),
    )
    return pl.DataFrame({"distance": np.rint(distance), "made": made, "p_hat": p_hat})


def _conv_rows(plays: pl.DataFrame, conv_model: ConversionModel) -> pl.DataFrame:
    goes = plays.filter(
        (pl.col("down") == 4)
        & (pl.col("choice") == "go")
        & pl.col("qtr").is_between(1, 4)
        & pl.col("ydstogo").is_between(1, 20)
    )
    if goes.height == 0:
        return pl.DataFrame({"ydstogo": [], "success": [], "p_hat": []})
    pred = conv_model.predict(
        goes["ydstogo"].to_numpy(),
        goes["yardline_100"].to_numpy(),
        goes["season"].to_numpy(),
    )
    success = (
        (goes["fourth_down_converted"].fill_null(0).to_numpy() == 1)
        | (
            (goes["touchdown"].fill_null(0).to_numpy() == 1)
            & (goes["td_team"].to_numpy() == goes["posteam"].to_numpy())
        )
    ).astype(float)
    return pl.DataFrame(
        {
            "ydstogo": goes["ydstogo"].to_numpy(),
            "success": success,
            "p_hat": pred["p_td"] + pred["p_fd"],
        }
    )


def _sanity(bundle: dict) -> list[dict]:
    """A few situations where a sign error would recommend the wrong thing."""
    # Own-40, 4th-and-1 is left unconstrained. A conversion near 70% and a
    # punt that pins the opponent inside their 20 are close in win probability
    # early in a tied game, and forcing "go" would hide that.
    specs = [
        ("own 10, 4th and 10, tied, opening kickoff", {"yardline_100": 90, "ydstogo": 10}, "punt"),
        ("own 40, 4th and 1, tied, opening kickoff", {"yardline_100": 60, "ydstogo": 1}, None),
        ("opponent 2, 4th and 1, tied, opening kickoff", {"yardline_100": 2, "ydstogo": 1}, "go"),
        ("own 35, 4th and 12, tied, opening kickoff", {"yardline_100": 65, "ydstogo": 12}, "punt"),
        ("opponent 28, 4th and 6, tied, opening kickoff", {"yardline_100": 28, "ydstogo": 6}, None),
    ]
    base = {
        "score_diff": 0.0,
        "game_seconds_remaining": 3600.0,
        "season": float(LAST_FULL_SEASON),
        "receive_2h": 1.0,
        "expected_margin": 0.0,
        "is_home": 1.0,
    }
    rows = []
    for name, override, expect in specs:
        situation = make_situations([{**base, **override}])
        graded = choice_win_probs(situation, bundle["wp"], bundle["fg"], bundle["conv"], bundle["punt"])
        rec = graded.row(0, named=True)
        rows.append(
            {
                "name": name,
                "expected": expect,
                "recommended": rec["recommended"],
                "ok": True if expect is None else rec["recommended"] == expect,
                "wp_go": rec["wp_go"],
                "wp_punt": rec["wp_punt"],
                "wp_fg": rec["wp_fg"],
                "wp_edge": rec["wp_edge"],
            }
        )
    return rows


def _write_tables(decisions, metrics, tables, examples, fg_model, conv_model, wp_model, fg_oos, conv_oos) -> None:
    # The full decision file is the audit trail. Descriptions make it larger
    # than the summary tables and are worth keeping for the example plays.
    decisions.write_parquet(RESULTS_DIR / "decisions.parquet")
    metrics.write_csv(RESULTS_DIR / "wp_metrics_by_season.csv")
    for name, frame in tables.items():
        frame.write_csv(RESULTS_DIR / f"{name}.csv")
    write_examples(RESULTS_DIR / "examples.json", examples)
    pl.DataFrame(fg_model.curve_table(LAST_FULL_SEASON)).write_csv(RESULTS_DIR / "fg_curve.csv")
    pl.DataFrame(conv_model.probability_table(LAST_FULL_SEASON)).write_csv(RESULTS_DIR / "conversion_grid.csv")
    if fg_oos.height:
        fg_oos.write_parquet(RESULTS_DIR / "fg_oos.parquet")
    if conv_oos.height:
        conv_oos.write_parquet(RESULTS_DIR / "conversion_oos.parquet")
    notes = [
        "Win-probability feature importance (current bundle)",
        *[f"{name}: {score:.4f}" for name, score in wp_model.feature_importance()],
        "",
        "Platt scaling on the calibration season",
        f"intercept {wp_model.platt.intercept:.4f}",
        f"slope {wp_model.platt.slope:.4f}",
        f"calibration-season log loss raw {wp_model.cal_log_loss_raw:.4f}",
        f"calibration-season log loss calibrated {wp_model.cal_log_loss_calibrated:.4f}",
        "",
        "Field-goal weather GLM (offset = distance-curve log-odds)",
        fg_model.weather_summary,
        "",
        "Conversion GLM: P(success)",
        conv_model.success_summary,
        "",
        "Conversion GLM: P(touchdown)",
        conv_model.td_summary,
        "",
        "Failure split (share of failed attempts)",
        f"downs {conv_model.transitions.fail_downs:.3f}",
        f"giveaway {conv_model.transitions.fail_giveaway:.3f}",
        f"defensive TD {conv_model.transitions.fail_def_td:.3f}",
        f"safety {conv_model.transitions.fail_safety:.3f}",
        f"TD point weights 6/7/8 {conv_model.transitions.td_point_weights.tolist()}",
    ]
    (RESULTS_DIR / "model_notes.txt").write_text("\n".join(notes))


def _write_figures(bundle, metrics, reliability, scatter, tables, fg_oos, sanity) -> None:
    # Overall reliability, pooled across held-out seasons.
    table = calibration_table(
        reliability["y"].to_numpy(),
        reliability["p"].to_numpy(),
        n_bins=10,
    )
    table.to_csv(RESULTS_DIR / "wp_calibration_bins.csv", index=False)
    plot_calibration(
        pl.from_pandas(table),
        FIGURES_DIR / "wp_calibration.png",
        "Held-out win probability is close to the diagonal",
    )
    plot_logloss(metrics, FIGURES_DIR / "wp_logloss_by_season.png")
    plot_vs_nflfastr(scatter, FIGURES_DIR / "wp_vs_nflfastr.png")

    scenarios = [
        (
            "decision_chart_opening.png",
            "Opening kickoff, tied game",
            "Neutral spread, home team, three timeouts. Early on, a field goal is worth only a few points of win probability, so the chart is go or punt.",
            {"score_diff": 0.0, "game_seconds_remaining": 3600.0, "receive_2h": 1.0},
        ),
        (
            "decision_chart_q4_tied.png",
            "Tied, 5:00 left in the fourth",
            "Same neutral spot as the opening chart, with the clock now mattering.",
            {"score_diff": 0.0, "game_seconds_remaining": 300.0, "receive_2h": 0.0},
        ),
        (
            "decision_chart_q4_down3.png",
            "Down 3, 2:00 left in the fourth",
            "A field goal ties the game. Going for it can win it.",
            {"score_diff": -3.0, "game_seconds_remaining": 120.0, "receive_2h": 0.0},
        ),
    ]
    for filename, title, subtitle, scenario in scenarios:
        grid = decision_grid({**scenario, "season": float(bundle["meta"]["calibrate_season"])})
        graded = choice_win_probs(grid, bundle["wp"], bundle["fg"], bundle["conv"], bundle["punt"])
        plot_decision_chart(graded, FIGURES_DIR / filename, title, subtitle)

    _plot_fg(fg_oos)
    conv_table = pl.DataFrame(bundle["conv"].probability_table(int(bundle["meta"]["calibrate_season"])))
    plot_conversion(conv_table, FIGURES_DIR / "conversion_by_ydstogo.png")
    plot_trend(tables["trend_by_year"], FIGURES_DIR / "go_rate_trend.png")
    _plot_leaderboards(tables)
    pl.DataFrame(sanity).write_csv(RESULTS_DIR / "sanity.csv")


def _plot_fg(fg_oos: pl.DataFrame) -> None:
    if fg_oos.height == 0:
        return
    binned = (
        fg_oos.group_by("distance")
        .agg(
            pl.len().alias("n"),
            pl.col("made").mean().alias("actual_rate"),
            pl.col("p_hat").mean().alias("league_mix"),
        )
        .sort("distance")
    )
    # The shared FG figure expects a `distance` curve. Here `league_mix` is the
    # mean out-of-sample prediction, plotted against the actual make rate.
    plot_fg_curve(binned, FIGURES_DIR / "fg_by_distance.png")


def _plot_leaderboards(tables: dict[str, pl.DataFrame]) -> None:
    latest = int(tables["team_season"]["season"].max())
    teams = (
        tables["team_season"]
        .filter(pl.col("season") == latest)
        .sort("wp_left_per_play", descending=True)
    )
    teams = teams.with_columns(
        (pl.col("team") + "  n=" + pl.col("decisions").cast(pl.Utf8)).alias("label")
    )
    plot_bars(
        teams,
        FIGURES_DIR / "team_wp_left_latest.png",
        "label",
        "wp_left_per_play",
        f"Win probability left per fourth down, {latest} (longer bar = more left on the table)",
        "Average win probability left on the table (percentage points)",
    )
    coaches = tables["coach_pooled"].sort("wp_left_per_play")
    if coaches.height > 16:
        show = pl.concat([coaches.head(8), coaches.tail(8)])
    else:
        show = coaches
    show = show.with_columns(
        (pl.col("coach") + " (" + pl.col("teams") + ")").alias("label")
    ).sort("wp_left_per_play", descending=True)
    plot_bars(
        show,
        FIGURES_DIR / "coach_wp_left_pooled.png",
        "label",
        "wp_left_per_play",
        "Head coaches, pooled graded seasons (best and worst tails)",
        "Average win probability left per fourth down (percentage points)",
    )


def _summary(plays, decisions, metrics, tables, examples, sanity, wp, fg, conv, punt, quick) -> dict:
    board = decisions.filter(pl.col("in_leaderboard"))
    clear = board.filter(pl.col("clear_call"))
    weights = metrics["n"].to_numpy()

    def wavg(col: str) -> float:
        return float(np.average(metrics[col].to_numpy(), weights=weights))

    trend = tables["trend_by_year"]
    return {
        "quick": quick,
        "seasons_in_file": [int(s) for s in sorted(plays["season"].unique().to_list())],
        "graded_seasons": [int(s) for s in metrics["season"].to_list()],
        "n_plays_prepared": int(plays.height),
        "n_fourth_downs": int(decisions.height),
        "n_leaderboard": int(board.height),
        "n_clear": int(clear.height),
        "follow_rate_leaderboard": _mean(board["followed"]),
        "follow_rate_clear": _mean(clear["followed"]),
        "wp_left_per_play": _mean(board["wp_left"]),
        "wp_left_per_clear_play": _mean(clear["wp_left"]),
        "wp_edge_mean": _mean(board["wp_edge"]),
        "logloss_ours": wavg("logloss_ours"),
        "logloss_raw": wavg("logloss_raw"),
        "logloss_nflfastr": wavg("logloss_nflfastr"),
        "logloss_vegas": wavg("logloss_vegas"),
        "brier_ours": wavg("brier_ours"),
        "brier_nflfastr": wavg("brier_nflfastr"),
        "calibration_mae": wavg("calibration_mae"),
        "corr_nflfastr": wavg("corr_nflfastr"),
        "metrics_by_season": metrics.to_dicts(),
        "trend": trend.to_dicts(),
        "team_pooled_best": tables["team_pooled"].head(5).to_dicts(),
        "team_pooled_worst": tables["team_pooled"].tail(5).to_dicts(),
        "coach_pooled_best": tables["coach_pooled"].head(5).to_dicts(),
        "coach_pooled_worst": tables["coach_pooled"].tail(5).to_dicts(),
        "latest_season_teams": tables["team_season"]
        .filter(pl.col("season") == tables["team_season"]["season"].max())
        .sort("wp_left_per_play")
        .head(5)
        .to_dicts(),
        "examples": examples,
        "sanity": sanity,
        "wp_importance": wp.feature_importance(),
        "platt_slope": wp.platt.slope,
        "platt_intercept": wp.platt.intercept,
        "wp_train_seasons": wp.train_seasons,
        "wp_calibrate_season": wp.calibrate_season,
        "wp_best_iteration": wp.best_iteration,
        "fg_n": fg.n_kicks,
        "fg_make_rate": fg.n_made / fg.n_kicks if fg.n_kicks else None,
        "fg_weather_coef": fg.weather_coef,
        "fg_weather_pvalue": fg.weather_pvalue,
        "fg_curve_marks": {
            str(dist): float(fg.distance_probability(dist))
            for dist in (30, 40, 45, 50, 55, 60)
        },
        "conversion_n": conv.transitions.n_go,
        "conversion_success_rate": conv.transitions.n_success / conv.transitions.n_go,
        "conversion_td_rate": conv.transitions.n_td / conv.transitions.n_go,
        "td_point_weights": conv.transitions.td_point_weights.tolist(),
        "fail_rates": {
            "downs": conv.transitions.fail_downs,
            "giveaway": conv.transitions.fail_giveaway,
            "def_td": conv.transitions.fail_def_td,
            "safety": conv.transitions.fail_safety,
        },
        "punt_n": punt.n_punts,
        "punt_elapsed": punt.elapsed,
        "elapsed_go": conv.transitions.elapsed_go,
        "elapsed_score": conv.transitions.elapsed_score,
        "conversion_at_midfield": {
            str(ytg): float(
                conv.predict(np.array([ytg]), np.array([50.0]), np.array([float(LAST_FULL_SEASON)]))["p_td"][0]
                + conv.predict(np.array([ytg]), np.array([50.0]), np.array([float(LAST_FULL_SEASON)]))["p_fd"][0]
            )
            for ytg in (1, 2, 3, 5, 10)
        },
    }


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _mean(series: pl.Series) -> float | None:
    if series.len() == 0:
        return None
    value = series.drop_nulls().mean()
    if value is None:
        return None
    return float(value)
