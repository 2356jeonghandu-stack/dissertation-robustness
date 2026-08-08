"""Reusable, leakage-safe sector robustness pipeline.

The moderate-noise range (0.05–0.20) is the primary experiment. Intensities
0.50 and 1.00 are exploratory bounded-input stress tests; clipping is always
reported explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sector_config import (
    ANNUALISATION_FACTOR,
    CLASSIFICATION_THRESHOLD,
    FEATURE_GROUPS,
    LOGISTIC_REGRESSION_SPECIFICATION,
    NOISE_INTENSITIES,
    PREDICTORS,
    PURGE_LENGTH,
    RANDOM_FOREST_SPECIFICATION,
    RANDOM_SEEDS,
    TARGET_HORIZON,
    TARGET_QUANTILE,
    TRAINING_PROPORTION,
)

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision",
)
DETAIL_KEYS = ("model", "feature_group", "noise_intensity", "seed")
SUMMARY_KEYS = ("model", "feature_group", "noise_intensity", "metric")
CLIPPING_KEYS = ("feature_group", "noise_intensity", "seed")


@dataclass(frozen=True)
class SectorDataset:
    ticker: str
    raw: pd.DataFrame
    modelling: pd.DataFrame
    purge_dates: pd.DatetimeIndex
    first_test_date: pd.Timestamp
    training_threshold: float
    raw_sample_size: int
    valid_sample_size: int


@dataclass(frozen=True)
class ExperimentResults:
    ticker: str
    models: Mapping[str, object]
    clean_metrics: pd.DataFrame
    detailed: pd.DataFrame
    summary: pd.DataFrame
    clipping: pd.DataFrame
    model_fit_count: int


def _require_ticker(ticker: str) -> str:
    value = str(ticker).strip().upper()
    if not value:
        raise ValueError("A ticker or sector identifier is required.")
    return value


def normalise_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return one ticker's OHLCV columns from single-level or MultiIndex data."""
    ticker = _require_ticker(ticker)
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        matching_levels = [
            level
            for level in range(data.columns.nlevels)
            if ticker in set(map(str, data.columns.get_level_values(level)))
        ]
        if len(matching_levels) != 1:
            raise ValueError(
                f"{ticker}: expected one unambiguous ticker level; found {matching_levels}."
            )
        data = data.xs(ticker, axis=1, level=matching_levels[0], drop_level=True)
        if isinstance(data.columns, pd.MultiIndex):
            singleton_levels = [
                level
                for level in range(data.columns.nlevels)
                if len(set(data.columns.get_level_values(level))) == 1
            ]
            for level in reversed(singleton_levels):
                data = data.droplevel(level, axis=1)
    data.columns = pd.Index(map(str, data.columns))
    return data


def load_local_ohlcv_csv(path: str | Path, ticker: str) -> pd.DataFrame:
    """Load and validate a local yfinance OHLCV CSV without network access."""
    ticker = _require_ticker(ticker)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{ticker}: local raw file is unavailable: {path}")
    first_two = pd.read_csv(path, nrows=2, header=None)
    multiindex = first_two.shape[0] >= 2 and ticker in set(first_two.iloc[1].astype(str))
    if multiindex:
        loaded = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        loaded = pd.read_csv(path, index_col=0, parse_dates=True)
    data = normalise_yfinance_columns(loaded, ticker)
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"
    required = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{ticker}: missing required OHLCV columns: {missing}")
    if not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError(f"{ticker}: dates must be strictly increasing and unique.")
    if not (data[list(required[:-1])] > 0).all().all():
        raise ValueError(f"{ticker}: OHLC and adjusted prices must be positive.")
    if not (data["Volume"] >= 0).all():
        raise ValueError(f"{ticker}: volume must be non-negative.")
    if data[list(required)].isna().any().any():
        raise ValueError(f"{ticker}: raw OHLCV values must be complete.")
    return data


def calculate_future_realised_volatility(
    raw: pd.DataFrame, ticker: str, horizon: int = TARGET_HORIZON
) -> pd.Series:
    """Calculate future realised volatility from returns t+1 through t+horizon."""
    ticker = _require_ticker(ticker)
    if "Adj Close" not in raw:
        raise ValueError(f"{ticker}: Adjusted Close is required.")
    log_return = np.log(raw["Adj Close"] / raw["Adj Close"].shift(1))
    terms = pd.concat(
        [log_return.shift(-offset).pow(2) for offset in range(1, horizon + 1)],
        axis=1,
    )
    result = np.sqrt(terms.sum(axis=1, min_count=horizon))
    result.name = "future_rv_5d"
    if not result.notna().equals(terms.notna().all(axis=1)):
        raise AssertionError(f"{ticker}: an incomplete future target was constructed.")
    if not result.iloc[-horizon:].isna().all():
        raise AssertionError(f"{ticker}: final target observations must be missing.")
    return result


def calculate_predictors(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Calculate the exact eleven predictors using information through date t."""
    ticker = _require_ticker(ticker)
    adjusted = raw["Adj Close"]
    log_return = np.log(adjusted / adjusted.shift(1))
    annualiser = np.sqrt(ANNUALISATION_FACTOR)
    features = pd.DataFrame(index=raw.index)
    features["log_return_1d"] = log_return
    features["return_5d"] = log_return.rolling(5).sum()
    features["return_20d"] = log_return.rolling(20).sum()
    features["volatility_5d"] = log_return.rolling(5).std(ddof=1) * annualiser
    features["volatility_20d"] = log_return.rolling(20).std(ddof=1) * annualiser
    features["downside_volatility_20d"] = (
        np.sqrt(log_return.clip(upper=0).pow(2).rolling(20).mean()) * annualiser
    )
    features["price_to_ma_10"] = adjusted / adjusted.rolling(10).mean() - 1
    features["price_to_ma_50"] = adjusted / adjusted.rolling(50).mean() - 1
    change = adjusted.diff()
    average_gain = change.clip(lower=0).rolling(14).mean()
    average_loss = (-change.clip(upper=0)).rolling(14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    features["rsi_14"] = rsi
    trailing_volume = raw["Volume"].rolling(20).mean()
    if not (trailing_volume.dropna() > 0).all() or not (raw["Close"] > 0).all():
        raise ValueError(f"{ticker}: feature denominators must be positive.")
    features["volume_ratio_20d"] = raw["Volume"] / trailing_volume
    features["intraday_range"] = (raw["High"] - raw["Low"]) / raw["Close"]
    return features.loc[:, list(PREDICTORS)]


def construct_chronological_split(
    valid_target_frame: pd.DataFrame,
    ticker: str,
    training_proportion: float = TRAINING_PROPORTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the provisional chronological training/test split."""
    ticker = _require_ticker(ticker)
    if not valid_target_frame.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: target dates are not chronological.")
    split_index = int(np.floor(training_proportion * len(valid_target_frame)))
    if not 0 < split_index < len(valid_target_frame):
        raise ValueError(f"{ticker}: chronological split would be empty.")
    return (
        valid_target_frame.iloc[:split_index].copy(),
        valid_target_frame.iloc[split_index:].copy(),
    )


def purge_provisional_training(
    provisional_training: pd.DataFrame,
    test: pd.DataFrame,
    raw_dates: pd.DatetimeIndex,
    ticker: str,
    purge_length: int = PURGE_LENGTH,
    target_horizon: int = TARGET_HORIZON,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Purge final provisional observations whose targets overlap the test period."""
    ticker = _require_ticker(ticker)
    if len(provisional_training) <= purge_length or test.empty:
        raise ValueError(f"{ticker}: insufficient observations for the purge.")
    purge_dates = provisional_training.index[-purge_length:].copy()
    training = provisional_training.iloc[:-purge_length].copy()
    first_test_date = test.index[0]
    if len(purge_dates) != purge_length or not (purge_dates < first_test_date).all():
        raise AssertionError(f"{ticker}: invalid purge dates.")
    for date in purge_dates:
        position = raw_dates.get_loc(date)
        target_dates = raw_dates[position + 1 : position + 1 + target_horizon]
        if len(target_dates) != target_horizon or not (target_dates >= first_test_date).any():
            raise AssertionError(f"{ticker}: purged target does not overlap the test boundary.")
    for date in training.index:
        position = raw_dates.get_loc(date)
        if raw_dates[position + target_horizon] >= first_test_date:
            raise AssertionError(f"{ticker}: retained training target overlaps the test period.")
    return training, purge_dates


def derive_training_target(
    training: pd.DataFrame,
    test: pd.DataFrame,
    ticker: str,
    quantile: float = TARGET_QUANTILE,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Derive one threshold from final training observations and label both samples."""
    ticker = _require_ticker(ticker)
    threshold = float(training["future_rv_5d"].quantile(quantile))
    labelled_training = training.copy()
    labelled_test = test.copy()
    labelled_training["high_volatility"] = (
        labelled_training["future_rv_5d"] > threshold
    ).astype(int)
    labelled_test["high_volatility"] = (
        labelled_test["future_rv_5d"] > threshold
    ).astype(int)
    labelled_training["sample_period"] = "train"
    labelled_test["sample_period"] = "test"
    if not np.isclose(
        threshold, training["future_rv_5d"].quantile(quantile), rtol=0, atol=0
    ):
        raise AssertionError(f"{ticker}: threshold was not training-derived.")
    return labelled_training, labelled_test, threshold


def build_sector_dataset(raw: pd.DataFrame, ticker: str) -> SectorDataset:
    """Build the verified target, purge, feature set and final modelling data."""
    ticker = _require_ticker(ticker)
    predictors = calculate_predictors(raw, ticker)
    future = calculate_future_realised_volatility(raw, ticker)
    log_return = predictors["log_return_1d"]
    valid = pd.DataFrame(
        {
            "Adjusted Close": raw["Adj Close"],
            "log_return": log_return,
            "volatility_20d": predictors["volatility_20d"],
            "future_rv_5d": future,
        }
    ).dropna()
    provisional, test = construct_chronological_split(valid, ticker)
    training, purge_dates = purge_provisional_training(
        provisional, test, raw.index, ticker
    )
    training, test, threshold = derive_training_target(training, test, ticker)
    targets = pd.concat([training, test])
    modelling = targets[["future_rv_5d", "high_volatility", "sample_period"]].join(
        predictors, how="left"
    )
    modelling = modelling.dropna(subset=list(PREDICTORS)).reset_index()
    modelling = modelling[
        ["Date", *PREDICTORS, "future_rv_5d", "high_volatility", "sample_period"]
    ]
    if modelling["Date"].isin(purge_dates).any():
        raise AssertionError(f"{ticker}: purged observations remain in modelling data.")
    if not modelling["Date"].is_monotonic_increasing or not modelling["Date"].is_unique:
        raise AssertionError(f"{ticker}: modelling dates are invalid.")
    if not np.isfinite(modelling[list(PREDICTORS)].to_numpy()).all():
        raise AssertionError(f"{ticker}: predictors are not finite.")
    return SectorDataset(
        ticker=ticker,
        raw=raw,
        modelling=modelling,
        purge_dates=purge_dates,
        first_test_date=test.index[0],
        training_threshold=threshold,
        raw_sample_size=len(raw),
        valid_sample_size=len(valid),
    )


def fit_clean_models(
    modelling: pd.DataFrame, ticker: str
) -> tuple[dict[str, object], int]:
    """Fit exactly two pre-specified models using clean training data only."""
    ticker = _require_ticker(ticker)
    training = modelling.loc[modelling["sample_period"].eq("train")]
    test_dates = set(modelling.loc[modelling["sample_period"].eq("test"), "Date"])
    if set(training["Date"]) & test_dates:
        raise AssertionError(f"{ticker}: test observations entered fitting.")
    X_training = training.loc[:, list(PREDICTORS)]
    y_training = training["high_volatility"].astype(int)
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(**dict(LOGISTIC_REGRESSION_SPECIFICATION))),
        ]
    )
    forest = RandomForestClassifier(**dict(RANDOM_FOREST_SPECIFICATION))
    models = {"Logistic Regression": logistic, "Random Forest": forest}
    fit_count = 0
    for model in models.values():
        model.fit(X_training, y_training)
        fit_count += 1
    if not np.allclose(logistic["scaler"].mean_, X_training.mean().to_numpy()):
        raise AssertionError(f"{ticker}: scaler was not fitted to training data.")
    if len(forest.estimators_) != RANDOM_FOREST_SPECIFICATION["n_estimators"]:
        raise AssertionError(f"{ticker}: Random Forest tree count is incorrect.")
    return models, fit_count


def _metric_values(target: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(target, prediction),
        "balanced_accuracy": balanced_accuracy_score(target, prediction),
        "precision": precision_score(target, prediction, zero_division=0),
        "recall": recall_score(target, prediction, zero_division=0),
        "f1_score": f1_score(target, prediction, zero_division=0),
        "roc_auc": roc_auc_score(target, probability),
        "average_precision": average_precision_score(target, probability),
    }


def calculate_clean_metrics(
    models: Mapping[str, object], modelling: pd.DataFrame, ticker: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Evaluate both fitted models once on the fixed clean test sample."""
    ticker = _require_ticker(ticker)
    test = modelling.loc[modelling["sample_period"].eq("test")]
    X_test = test.loc[:, list(PREDICTORS)]
    target = test["high_volatility"].astype(int)
    by_model: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        probability = model.predict_proba(X_test)[:, 1]
        if not np.logical_and(probability >= 0, probability <= 1).all():
            raise AssertionError(f"{ticker}: invalid {name} probabilities.")
        by_model[name] = _metric_values(target, probability)
    table = pd.DataFrame([{"model": name, **values} for name, values in by_model.items()])
    return table, by_model


def apply_logical_clipping(
    noisy: pd.DataFrame,
    selected_features: Sequence[str],
    ticker: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply only logical domain bounds and count changed values."""
    _require_ticker(ticker)
    bounded = noisy.copy()
    counts = {feature: 0 for feature in PREDICTORS}
    lower_zero = {
        "volatility_5d",
        "volatility_20d",
        "downside_volatility_20d",
        "volume_ratio_20d",
        "intraday_range",
    }
    for feature in selected_features:
        before = bounded[feature].copy()
        if feature in lower_zero:
            bounded[feature] = bounded[feature].clip(lower=0)
        elif feature == "rsi_14":
            bounded[feature] = bounded[feature].clip(lower=0, upper=100)
        elif feature in {"price_to_ma_10", "price_to_ma_50"}:
            bounded[feature] = bounded[feature].clip(lower=-1)
        counts[feature] = int(before.ne(bounded[feature]).sum())
    return bounded, counts


def produce_noisy_test_dataset(
    X_test: pd.DataFrame,
    training_standard_deviation: pd.Series,
    feature_group: str,
    intensity: float,
    seed: int,
    ticker: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Produce one deterministic noisy test frame for paired model evaluation."""
    ticker = _require_ticker(ticker)
    if feature_group not in FEATURE_GROUPS:
        raise KeyError(f"{ticker}: unknown feature group {feature_group}.")
    selected = list(FEATURE_GROUPS[feature_group])
    untouched = [feature for feature in PREDICTORS if feature not in selected]
    generator = np.random.default_rng(seed)
    standard_noise = generator.normal(size=(len(X_test), len(selected)))
    noisy = X_test.copy(deep=True)
    scale = intensity * training_standard_deviation[selected].to_numpy()
    noisy.loc[:, selected] = X_test[selected].to_numpy() + standard_noise * scale
    noisy, counts = apply_logical_clipping(noisy, selected, ticker)
    if untouched:
        pd.testing.assert_frame_equal(noisy[untouched], X_test[untouched], check_exact=True)
    if not np.isfinite(noisy.to_numpy()).all():
        raise AssertionError(f"{ticker}: noise created missing or infinite values.")
    return noisy, counts


def evaluate_noise_experiment(
    models: Mapping[str, object],
    modelling: pd.DataFrame,
    clean_by_model: Mapping[str, Mapping[str, float]],
    ticker: str,
    intensities: Sequence[float] = NOISE_INTENSITIES,
    seeds: Sequence[int] = RANDOM_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all paired model/group/intensity/seed test-time conditions."""
    ticker = _require_ticker(ticker)
    training = modelling.loc[modelling["sample_period"].eq("train")]
    test = modelling.loc[modelling["sample_period"].eq("test")]
    X_training = training.loc[:, list(PREDICTORS)]
    X_test = test.loc[:, list(PREDICTORS)]
    target = test["high_volatility"].astype(int)
    target_before = target.copy(deep=True)
    scales = X_training.std(ddof=1)
    if not (scales > 0).all():
        raise AssertionError(f"{ticker}: every training noise scale must be positive.")
    detailed_rows: list[dict[str, object]] = []
    clipping_rows: list[dict[str, object]] = []
    for group, selected in FEATURE_GROUPS.items():
        for intensity in intensities:
            for seed in seeds:
                noisy, counts = produce_noisy_test_dataset(
                    X_test, scales, group, intensity, seed, ticker
                )
                signature = int(pd.util.hash_pandas_object(noisy, index=True).sum())
                clipping_rows.append(
                    {
                        "feature_group": group,
                        "noise_intensity": intensity,
                        "seed": seed,
                        "total_clipped_values": int(sum(counts.values())),
                        **{
                            f"clipped_{feature}": counts[feature]
                            for feature in PREDICTORS
                        },
                    }
                )
                paired_probabilities = {}
                for model_name, model in models.items():
                    probability = model.predict_proba(noisy)[:, 1]
                    paired_probabilities[model_name] = probability
                    values = _metric_values(target, probability)
                    row: dict[str, object] = {
                        "model": model_name,
                        "feature_group": group,
                        "noise_intensity": intensity,
                        "seed": seed,
                        "dataset_signature": signature,
                    }
                    for metric in METRICS:
                        clean_value = clean_by_model[model_name][metric]
                        row[f"clean_{metric}"] = clean_value
                        row[f"noisy_{metric}"] = values[metric]
                        row[f"{metric}_degradation"] = clean_value - values[metric]
                    detailed_rows.append(row)
                if len(paired_probabilities) != len(models):
                    raise AssertionError(f"{ticker}: paired model evaluation is incomplete.")
    pd.testing.assert_series_equal(target, target_before, check_exact=True)
    detailed = pd.DataFrame(detailed_rows)
    clipping = pd.DataFrame(clipping_rows)
    expected = len(FEATURE_GROUPS) * len(intensities) * len(seeds)
    if len(detailed) != expected * len(models) or len(clipping) != expected:
        raise AssertionError(f"{ticker}: noise result row counts are incorrect.")
    return detailed, clipping


def create_summary_table(detailed: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Summarise seed-level performance and degradation with 95% intervals."""
    _require_ticker(ticker)
    rows = []
    for keys, condition in detailed.groupby(list(DETAIL_KEYS[:-1]), sort=False):
        model, group, intensity = keys
        for metric in METRICS:
            performance = condition[f"noisy_{metric}"]
            degradation = condition[f"{metric}_degradation"]
            count = len(condition)
            performance_sd = performance.std(ddof=1)
            degradation_sd = degradation.std(ddof=1)
            performance_half = 1.96 * performance_sd / np.sqrt(count)
            degradation_half = 1.96 * degradation_sd / np.sqrt(count)
            rows.append(
                {
                    "model": model,
                    "feature_group": group,
                    "noise_intensity": intensity,
                    "metric": metric,
                    "seed_count": count,
                    "clean_performance": condition[f"clean_{metric}"].iloc[0],
                    "mean_noisy_performance": performance.mean(),
                    "std_noisy_performance": performance_sd,
                    "noisy_performance_ci95_lower": performance.mean() - performance_half,
                    "noisy_performance_ci95_upper": performance.mean() + performance_half,
                    "mean_performance_degradation": degradation.mean(),
                    "std_performance_degradation": degradation_sd,
                    "degradation_ci95_lower": degradation.mean() - degradation_half,
                    "degradation_ci95_upper": degradation.mean() + degradation_half,
                }
            )
    return pd.DataFrame(rows)


def run_sector_pipeline(raw: pd.DataFrame, ticker: str) -> tuple[SectorDataset, ExperimentResults]:
    """Run the complete reusable in-memory pipeline for one sector."""
    dataset = build_sector_dataset(raw, ticker)
    models, fit_count = fit_clean_models(dataset.modelling, ticker)
    clean_table, clean_by_model = calculate_clean_metrics(models, dataset.modelling, ticker)
    detailed, clipping = evaluate_noise_experiment(
        models, dataset.modelling, clean_by_model, ticker
    )
    summary = create_summary_table(detailed, ticker)
    return dataset, ExperimentResults(
        ticker=_require_ticker(ticker),
        models=models,
        clean_metrics=clean_table,
        detailed=detailed,
        summary=summary,
        clipping=clipping,
        model_fit_count=fit_count,
    )


def generate_degradation_figures(
    summary: pd.DataFrame, output_directory: str | Path, ticker: str
) -> list[Path]:
    """Generate the four degradation figures and the intensity-1.00 heatmap."""
    ticker = _require_ticker(ticker)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    titles = {
        "balanced_accuracy": "Balanced accuracy",
        "f1_score": "F1-score",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average precision",
    }
    paths = []
    for metric, title in titles.items():
        subset = summary.loc[summary["metric"].eq(metric)]
        low = subset["degradation_ci95_lower"].min()
        high = subset["degradation_ci95_upper"].max()
        margin = max((high - low) * 0.08, 0.002)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        for axis, (model, model_rows) in zip(
            axes, subset.groupby("model", sort=False), strict=True
        ):
            for group, rows in model_rows.groupby("feature_group", sort=False):
                rows = rows.sort_values("noise_intensity")
                x = rows["noise_intensity"].to_numpy(float)
                y = rows["mean_performance_degradation"].to_numpy(float)
                axis.plot(x, y, marker="o", label=group.replace("_", " ").title())
                axis.fill_between(
                    x,
                    rows["degradation_ci95_lower"].to_numpy(float),
                    rows["degradation_ci95_upper"].to_numpy(float),
                    alpha=0.16,
                )
            axis.axhline(0, color="black", linestyle="--", linewidth=1)
            axis.set_title(model)
            axis.set_xlabel("Noise intensity")
            axis.set_ylim(low - margin, high + margin)
        axes[0].set_ylabel(f"{title} degradation (clean − noisy)")
        axes[1].legend(fontsize=8)
        figure.suptitle(f"{ticker} {title} degradation under test-time feature noise")
        figure.tight_layout()
        path = output_directory / f"{ticker.lower()}_noise_{metric}_degradation.png"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    heat = summary.loc[
        summary["noise_intensity"].eq(1.0) & summary["metric"].isin(titles)
    ].copy()
    heat["row"] = heat["model"] + " — " + heat["metric"].map(titles)
    matrix = heat.pivot(
        index="row", columns="feature_group", values="mean_performance_degradation"
    ).reindex(columns=list(FEATURE_GROUPS))
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        center=0,
        ax=axis,
        cbar_kws={"label": "Mean degradation (clean − noisy)"},
    )
    axis.set_title(f"{ticker} mean degradation at noise intensity 1.00")
    axis.set_xlabel("Feature group")
    axis.set_ylabel("Model and metric")
    figure.tight_layout()
    path = output_directory / f"{ticker.lower()}_noise_degradation_heatmap_intensity_1_00.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def create_cross_sector_comparison(
    summaries: Mapping[str, pd.DataFrame],
    ticker: str,
    feature_groups: Sequence[str] = ("all_predictors", "volatility"),
    intensities: Sequence[float] = (0.20, 0.50, 1.00),
    metrics: Sequence[str] = (
        "balanced_accuracy",
        "f1_score",
        "roc_auc",
        "average_precision",
    ),
) -> pd.DataFrame:
    """Create a stable comparison table for explicitly supplied sectors."""
    _require_ticker(ticker)
    parts = []
    for sector, frame in summaries.items():
        part = frame.loc[
            frame["feature_group"].isin(feature_groups)
            & frame["noise_intensity"].isin(intensities)
            & frame["metric"].isin(metrics)
        ].copy()
        part.insert(0, "sector", _require_ticker(sector))
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["sector", *SUMMARY_KEYS], kind="stable"
    ).reset_index(drop=True)


def hash_files(paths: Sequence[str | Path], ticker: str) -> dict[str, str]:
    """Return SHA-256 hashes for structured validation records."""
    _require_ticker(ticker)
    return {
        str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in paths
    }


def create_validation_output(
    dataset: SectorDataset,
    results: ExperimentResults,
    protected_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Create the requested structured validation output."""
    modelling = dataset.modelling
    training = modelling.loc[modelling["sample_period"].eq("train")]
    test = modelling.loc[modelling["sample_period"].eq("test")]
    principal_models = {"Logistic Regression", "Random Forest"}
    return {
        "ticker": dataset.ticker,
        "raw_sample_size": dataset.raw_sample_size,
        "modelling_sample_size": len(modelling),
        "training_sample_size": len(training),
        "test_sample_size": len(test),
        "first_date": dataset.raw.index.min().strftime("%Y-%m-%d"),
        "final_date": dataset.raw.index.max().strftime("%Y-%m-%d"),
        "purge_dates": [date.strftime("%Y-%m-%d") for date in dataset.purge_dates],
        "first_test_date": dataset.first_test_date.strftime("%Y-%m-%d"),
        "training_threshold": dataset.training_threshold,
        "training_class_prevalence": training["high_volatility"].mean(),
        "test_class_prevalence": test["high_volatility"].mean(),
        "model_fit_count": results.model_fit_count,
        "clean_metric_row_count": int(
            results.clean_metrics["model"].isin(principal_models).sum()
        ),
        "detailed_row_count": len(results.detailed),
        "summary_row_count": len(results.summary),
        "clipping_row_count": len(results.clipping),
        "clipping_total": int(results.clipping["total_clipped_values"].sum()),
        "protected_file_hashes": dict(protected_hashes),
        "output_file_hashes": dict(output_hashes),
        "noise_range_labels": {
            "moderate": "Primary moderate-noise experiment (0.05–0.20)",
            "stress": "Exploratory bounded-input stress tests (0.50–1.00)",
        },
    }
