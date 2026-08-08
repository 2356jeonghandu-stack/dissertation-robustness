# Robustness to Feature Noise in Financial Machine Learning

This repository contains the reproducible research code for a dissertation examining how feature-level noise affects machine-learning predictions of next-week high-volatility regimes in US sector exchange-traded funds (ETFs).

## Research question

How does feature-level noise affect the robustness of machine-learning models for predicting next-week high-volatility regimes in sector portfolios?

## Study design

- Nine US sector ETFs with long price histories.
- Daily OHLCV observations obtained through Yahoo Finance using `yfinance`.
- Eleven predictors covering returns, volatility, momentum, volume and trading range.
- A forward five-trading-day realised-volatility target.
- A sector-specific high-volatility threshold estimated from the training sample only.
- Chronological 80/20 train-test split with a five-observation purge at the boundary.
- Logistic Regression and Random Forest models fitted to clean training data.
- Controlled feature noise applied only to test predictors.
- Thirty paired simulation seeds per experimental condition.
- Noise intensities 0.05–0.20 treated as the primary moderate-noise experiment; 0.50–1.00 treated as exploratory bounded-input stress tests.

## Main finding

Neither model is uniformly more robust within the primary moderate-noise range. Robustness varies materially across sectors, feature groups and evaluation metrics. Random Forest shows a clearer relative advantage in the exploratory high-intensity stress tests, although extensive logical-bound clipping limits their interpretation.

The repository supports this empirical result; it should not be interpreted as investment advice.

## Repository structure

```text
data/
├── raw/          # Local downloads; excluded from version control
└── processed/    # Generated feature datasets; excluded from version control
notebooks/        # Numbered analysis workflow
outputs/
├── figures/      # Generated figures
└── tables/       # Generated tables and validation reports
src/              # Reusable analysis pipeline and configuration
tests/            # Automated tests for the reusable pipeline
writing/          # Research-design and reproducibility notes
```

## Installation

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Market data are not committed to this repository. Run the first notebook to download the required Yahoo Finance observations, or provide equivalent OHLCV CSV files under `data/raw/`. The analysis uses data through 30 June 2026, with an exclusive requested end date of 1 July 2026.

Users are responsible for complying with the data provider's terms. Generated feature datasets under `data/processed/` are also excluded from version control.

## Reproduction workflow

Run the notebooks in numerical order:

1. `01_xlk_data_feasibility.ipynb`
2. `02_xlk_feature_engineering.ipynb`
3. `03_xlk_logistic_baseline.ipynb`
4. `04_xlk_random_forest_baseline.ipynb`
5. `05_xlk_feature_noise_robustness.ipynb`
6. `06_xle_cross_sector_replication.ipynb`
7. `07_pipeline_parity_validation.ipynb`
8. `08_multi_sector_batch_analysis.ipynb`

The notebooks discover the repository root dynamically and should be launched from anywhere inside the cloned repository.

Run the automated tests with:

```bash
python -m pytest -q
```

## Reproducibility notes

- All target thresholds, scaling parameters and noise scales are estimated from training data only.
- The same noisy test dataset is supplied to both models for each paired condition.
- Seed-based intervals quantify Monte Carlo uncertainty within experimental conditions; they do not provide population-level inference across sectors.
- A small historical XLF Random Forest discrepancy is documented in `writing/reproducibility_audit_note.md`. Software-version drift is a plausible but unconfirmed explanation.
- Stored tables and figures are research artefacts. Regenerated values may depend on the recorded package versions and upstream market-data revisions.

## AI assistance

AI tools assisted with code development, debugging, documentation and research-workflow organisation. The researcher remains responsible for the research design, verification, interpretation and final dissertation. Any formal disclosure should follow the applicable university policy.

## Licence

The current snapshot is provided for review under an all-rights-reserved notice. Replace `LICENSE` with an open-source licence only after confirming university, data-provider and dissertation-publication requirements.
