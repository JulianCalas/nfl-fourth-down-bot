"""Win probability, fit from the game state and nothing else.

The model is an XGBoost classifier with a handful of monotone constraints:
a larger lead cannot lower predicted win probability, and moving away from
the opponent's end zone cannot raise it. Constraints are not a substitute
for data. They stop the tree from inventing a silly answer in a sparse
corner of the state space (a 3-point lead with 20 seconds left, a yard line
the training sample barely saw).

Probabilities are then Platt-scaled on the previous season. The decision
rule subtracts punt win probability from go win probability, so those
numbers have to mean what they say. A raw boosting score is not a promise
of that. Platt scaling is a two-parameter logistic regression on the
model's log-odds, fit only on the calibration season.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from fourth_down.config import (
    RANDOM_SEED,
    WP_FEATURES,
    WP_LEARNING_RATE,
    WP_MAX_DEPTH,
    WP_MIN_CHILD_WEIGHT,
    WP_MONOTONE,
    WP_TREES,
)
from fourth_down.data import scrimmage_mask
from fourth_down.features import feature_frame_from_plays
from fourth_down.football import logit
from fourth_down.metrics import binary_log_loss


@dataclass
class PlattCalibrator:
    """Maps a raw probability through a one-variable logistic regression.

    Fit on a season the trees did not train on. `C` is huge on purpose:
    this is maximum likelihood for two coefficients, not a second model
    with its own shrinkage story.
    """

    model: LogisticRegression | None = None

    def fit(self, raw_probability: np.ndarray, won: np.ndarray) -> "PlattCalibrator":
        x = logit(raw_probability).reshape(-1, 1)
        y = np.asarray(won, dtype=int)
        # Essentially unpenalized. A finite C avoids a solver edge case with
        # infinite regularization in some scikit-learn versions.
        self.model = LogisticRegression(C=1e6, solver="lbfgs")
        self.model.fit(x, y)
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Platt calibrator is not fit")
        x = logit(raw_probability).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]

    @property
    def intercept(self) -> float:
        return float(self.model.intercept_[0])

    @property
    def slope(self) -> float:
        return float(self.model.coef_[0, 0])


@dataclass
class WPModel:
    """Possession-team win probability at the snap."""

    booster: xgb.XGBClassifier | None = None
    platt: PlattCalibrator | None = None
    train_seasons: list[int] = field(default_factory=list)
    calibrate_season: int | None = None
    n_train: int = 0
    n_calibrate: int = 0
    best_iteration: int | None = None
    cal_log_loss_raw: float | None = None
    cal_log_loss_calibrated: float | None = None

    def fit(
        self,
        plays,
        train_seasons: list[int],
        calibrate_season: int,
        n_estimators: int = WP_TREES,
        learning_rate: float = WP_LEARNING_RATE,
        max_depth: int = WP_MAX_DEPTH,
        min_child_weight: float = WP_MIN_CHILD_WEIGHT,
    ) -> "WPModel":
        train = plays.filter(scrimmage_mask() & pl_season_in(train_seasons))
        cal = plays.filter(scrimmage_mask() & pl_season_is(calibrate_season))
        if train.height < 5000 or cal.height < 1000:
            raise RuntimeError(
                f"WP split is too small (train={train.height}, cal={cal.height}). "
                "Check that the slim parquet covers these seasons."
            )
        x_train = feature_frame_from_plays(train)
        y_train = train["posteam_win"].to_numpy().astype(np.int8)
        x_cal = feature_frame_from_plays(cal)
        y_cal = cal["posteam_win"].to_numpy().astype(np.int8)

        booster = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            early_stopping_rounds=40,
            monotone_constraints=WP_MONOTONE,
            random_state=RANDOM_SEED,
            n_jobs=0,
        )
        booster.fit(
            x_train,
            y_train,
            eval_set=[(x_cal, y_cal)],
            verbose=False,
        )
        raw = booster.predict_proba(x_cal)[:, 1]
        platt = PlattCalibrator().fit(raw, y_cal)
        calibrated = platt.predict(raw)

        self.booster = booster
        self.platt = platt
        self.train_seasons = list(train_seasons)
        self.calibrate_season = int(calibrate_season)
        self.n_train = int(train.height)
        self.n_calibrate = int(cal.height)
        self.best_iteration = int(booster.best_iteration or 0)
        self.cal_log_loss_raw = binary_log_loss(y_cal, raw)
        self.cal_log_loss_calibrated = binary_log_loss(y_cal, calibrated)
        return self

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("WP model is not fit")
        return self.booster.predict_proba(features)[:, 1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Calibrated possession-team win probability."""
        raw = self.predict_raw(features)
        if self.platt is None:
            return raw
        return self.platt.predict(raw)

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.booster is None:
            return []
        scores = self.booster.feature_importances_
        pairs = list(zip(WP_FEATURES, [float(s) for s in scores]))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs


def pl_season_in(seasons: list[int]):
    import polars as pl

    return pl.col("season").is_in(list(seasons))


def pl_season_is(season: int):
    import polars as pl

    return pl.col("season") == season
