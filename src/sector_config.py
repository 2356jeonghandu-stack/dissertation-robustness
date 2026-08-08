"""Immutable settings for the sector robustness analysis."""

from types import MappingProxyType

REQUESTED_START_DATE = "2000-01-03"
REQUESTED_END_DATE_EXCLUSIVE = "2026-07-01"
TRAINING_PROPORTION = 0.80
PURGE_LENGTH = 5
TARGET_HORIZON = 5
TARGET_QUANTILE = 0.75
ANNUALISATION_FACTOR = 252
CLASSIFICATION_THRESHOLD = 0.5
RANDOM_SEEDS = tuple(range(30))
MODERATE_NOISE_INTENSITIES = (0.05, 0.10, 0.20)
STRESS_TEST_INTENSITIES = (0.50, 1.00)
NOISE_INTENSITIES = MODERATE_NOISE_INTENSITIES + STRESS_TEST_INTENSITIES

PREDICTORS = (
    "log_return_1d",
    "return_5d",
    "return_20d",
    "volatility_5d",
    "volatility_20d",
    "downside_volatility_20d",
    "price_to_ma_10",
    "price_to_ma_50",
    "rsi_14",
    "volume_ratio_20d",
    "intraday_range",
)

FEATURE_GROUPS = MappingProxyType(
    {
        "returns": PREDICTORS[0:3],
        "volatility": PREDICTORS[3:6],
        "momentum": PREDICTORS[6:9],
        "volume_and_range": PREDICTORS[9:11],
        "all_predictors": PREDICTORS,
    }
)

LOGISTIC_REGRESSION_SPECIFICATION = MappingProxyType(
    {
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
    }
)

RANDOM_FOREST_SPECIFICATION = MappingProxyType(
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    }
)

