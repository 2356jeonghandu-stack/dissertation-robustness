"""Tests for the reusable sector robustness pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sector_config import FEATURE_GROUPS, PREDICTORS  # noqa: E402
from sector_robustness_pipeline import (  # noqa: E402
    apply_logical_clipping,
    build_sector_dataset,
    create_summary_table,
    create_validation_output,
    derive_training_target,
    evaluate_noise_experiment,
    fit_clean_models,
    produce_noisy_test_dataset,
)


def synthetic_raw(rows: int = 140) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=rows)
    trend = 100 * np.exp(np.linspace(0, 0.25, rows))
    cycle = 1 + 0.015 * np.sin(np.arange(rows) / 3)
    adjusted = trend * cycle
    return pd.DataFrame(
        {
            "Open": adjusted * 0.998,
            "High": adjusted * 1.012,
            "Low": adjusted * 0.988,
            "Close": adjusted,
            "Adj Close": adjusted,
            "Volume": 1_000_000 + (np.arange(rows) % 17) * 10_000,
        },
        index=pd.Index(dates, name="Date"),
    )


def test_target_leakage_prevention_and_five_observation_purge():
    raw = synthetic_raw()
    original = raw.copy(deep=True)
    dataset = build_sector_dataset(raw, "TST")
    assert len(dataset.purge_dates) == 5
    assert not dataset.modelling["Date"].isin(dataset.purge_dates).any()
    assert (dataset.purge_dates < dataset.first_test_date).all()
    pd.testing.assert_frame_equal(raw, original)
    shifted_future = raw["Adj Close"].copy()
    shifted_future.iloc[-1] *= 10
    changed = raw.copy()
    changed["Adj Close"] = shifted_future
    # A future change cannot affect predictors dated before it.
    earlier = dataset.modelling["Date"] < raw.index[-1]
    rebuilt = build_sector_dataset(changed, "TST").modelling
    common = dataset.modelling.loc[earlier, ["Date", *PREDICTORS]].merge(
        rebuilt[["Date", *PREDICTORS]], on="Date", suffixes=("_a", "_b")
    )
    for feature in PREDICTORS:
        np.testing.assert_allclose(common[f"{feature}_a"], common[f"{feature}_b"])


def test_training_only_threshold_and_unchanged_test_targets():
    index = pd.bdate_range("2020-01-01", periods=20)
    training = pd.DataFrame({"future_rv_5d": np.arange(15.0)}, index=index[:15])
    test = pd.DataFrame({"future_rv_5d": np.arange(100.0, 105.0)}, index=index[15:])
    before = test.copy(deep=True)
    labelled_training, labelled_test, threshold = derive_training_target(
        training, test, "TST"
    )
    assert threshold == training["future_rv_5d"].quantile(0.75)
    pd.testing.assert_series_equal(
        labelled_test["future_rv_5d"], before["future_rv_5d"]
    )
    assert labelled_training["high_volatility"].sum() > 0
    assert labelled_test["high_volatility"].eq(1).all()


def test_deterministic_noise_group_isolation_and_clipping():
    columns = list(PREDICTORS)
    X = pd.DataFrame(np.ones((8, len(columns))), columns=columns)
    scales = pd.Series(0.5, index=columns)
    first, first_counts = produce_noisy_test_dataset(
        X, scales, "returns", 0.2, 7, "TST"
    )
    second, second_counts = produce_noisy_test_dataset(
        X, scales, "returns", 0.2, 7, "TST"
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_counts == second_counts
    untouched = [column for column in columns if column not in FEATURE_GROUPS["returns"]]
    pd.testing.assert_frame_equal(first[untouched], X[untouched])
    bounded = X.copy()
    bounded.loc[0, "rsi_14"] = 120
    bounded.loc[1, "volatility_5d"] = -1
    bounded.loc[2, "price_to_ma_10"] = -2
    clipped, counts = apply_logical_clipping(
        bounded, ["rsi_14", "volatility_5d", "price_to_ma_10"], "TST"
    )
    assert clipped.loc[0, "rsi_14"] == 100
    assert clipped.loc[1, "volatility_5d"] == 0
    assert clipped.loc[2, "price_to_ma_10"] == -1
    assert sum(counts.values()) == 3


def test_paired_noise_no_refitting_and_expected_row_counts():
    dataset = build_sector_dataset(synthetic_raw(180), "TST")
    models, fit_count = fit_clean_models(dataset.modelling, "TST")
    assert fit_count == 2
    test = dataset.modelling.query("sample_period == 'test'")
    clean = {}
    from sector_robustness_pipeline import calculate_clean_metrics

    _, clean = calculate_clean_metrics(models, dataset.modelling, "TST")
    model_ids = {name: id(model) for name, model in models.items()}
    detailed, clipping = evaluate_noise_experiment(
        models, dataset.modelling, clean, "TST", intensities=(0.1,), seeds=(0, 1)
    )
    assert len(detailed) == 2 * 5 * 1 * 2
    assert len(clipping) == 5 * 1 * 2
    assert detailed.groupby(
        ["feature_group", "noise_intensity", "seed"]
    )["dataset_signature"].nunique().eq(1).all()
    assert {name: id(model) for name, model in models.items()} == model_ids
    assert len(test) > 0
    summary = create_summary_table(detailed, "TST")
    assert len(summary) == 2 * 5 * 1 * 7


def test_british_english_validation_labels():
    dataset = build_sector_dataset(synthetic_raw(), "TST")
    models, fit_count = fit_clean_models(dataset.modelling, "TST")
    from sector_robustness_pipeline import calculate_clean_metrics

    clean_table, clean = calculate_clean_metrics(models, dataset.modelling, "TST")
    detailed, clipping = evaluate_noise_experiment(
        models, dataset.modelling, clean, "TST", intensities=(0.1,), seeds=(0,)
    )
    from sector_robustness_pipeline import ExperimentResults

    results = ExperimentResults(
        "TST",
        models,
        clean_table,
        detailed,
        create_summary_table(detailed, "TST"),
        clipping,
        fit_count,
    )
    validation = create_validation_output(dataset, results, {}, {})
    labels = " ".join(validation["noise_range_labels"].values())
    assert "moderate-noise" in labels
    assert "bounded-input stress tests" in labels
    assert "normalization" not in labels


if __name__ == "__main__":
    tests = [
        test_target_leakage_prevention_and_five_observation_purge,
        test_training_only_threshold_and_unchanged_test_targets,
        test_deterministic_noise_group_isolation_and_clipping,
        test_paired_noise_no_refitting_and_expected_row_counts,
        test_british_english_validation_labels,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"{len(tests)} automated tests passed.")
