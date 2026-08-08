# Research Design v0.1

## 1. Provisional Title

Model Robustness under Feature Noise in Financial Machine Learning: Predicting High-Volatility Regimes Using Sector ETFs

## 2. Main Research Question

How does feature-level noise affect the robustness of machine learning models for predicting next-week high-volatility regimes in sector portfolios?

## 3. Research Motivation

Financial machine-learning models are normally evaluated using clean historical data. In practice, however, financial data may contain measurement error, reporting noise, missingness, revisions or unstable signals. A model that performs strongly on clean data may not remain reliable when its inputs are perturbed. This study therefore considers both predictive performance and the degradation of performance under controlled feature noise.

## 4. Prediction Task

The provisional prediction task is to:

- use US sector exchange-traded funds (ETFs) as proxies for sector portfolios;
- use daily Open, High, Low, Close and Volume (OHLCV) data;
- predict whether an ETF will experience a high-volatility regime during the following trading week;
- formulate the task as binary classification;
- begin the feasibility study using XLK only; and
- expand to other sector ETFs only after the pipeline has been validated.

## 5. Candidate Data

The candidate dataset will contain daily Open, High, Low, Close, Adjusted Close and Volume observations. The data source and sample period have not yet been finalised. All predictors must be constructed exclusively from information available at the prediction date. Future information must not be used in feature construction.

## 6. Candidate Features

Candidate predictors will be organised into the following provisional groups:

- return features;
- volatility and risk features;
- momentum features;
- volume features; and
- technical indicators.

The specific lookback windows have not yet been finalised. The initial analysis should use a controlled number of features to avoid unnecessary complexity and to make the behaviour of the baseline pipeline easier to analyse.

## 7. Target Definition

The following target design will be treated as a proposal to be validated:

- calculate realised volatility over the following five trading days;
- define a high-volatility regime using a threshold derived from the distribution of future realised volatility;
- use an upper-quantile threshold as the initial approach; and
- estimate any thresholds used during model development from the training data only.

The final realised-volatility formula, threshold and resulting class proportions have not yet been determined. These choices will be decided through feasibility testing.

## 8. Candidate Models

The candidate models are:

- Logistic Regression as an interpretable baseline;
- Random Forest as a nonlinear, tree-based comparison; and
- XGBoost as a provisional third model.

The third model may be adjusted in response to supervisor feedback and time constraints.

## 9. Robustness Experiment

The provisional robustness experiment will:

- train models using clean training data;
- evaluate baseline performance on clean test data;
- add controlled noise to selected groups of test-set features;
- compare several noise intensities;
- repeat random perturbations using multiple random seeds;
- measure average performance degradation and its variability; and
- avoid perturbing the target variable.

Gaussian noise is an initial candidate rather than a finalised perturbation method. The noise scale should be determined from feature distributions in the training data so that differences in measurement units do not produce inconsistent perturbation magnitudes.

## 10. Evaluation Metrics

Candidate evaluation metrics are:

- balanced accuracy;
- F1-score;
- ROC-AUC;
- precision; and
- recall.

Ordinary accuracy may also be reported, but it must not be used as the sole metric when the classes are imbalanced.

A provisional definition of performance degradation is:

> Performance degradation = clean-test performance − noisy-test performance

## 11. Validation Strategy and Leakage Prevention

The analysis will:

- use chronological, time-based splitting rather than random splitting;
- fit scalers, thresholds and all preprocessing steps using training data only;
- preserve the temporal order of observations;
- exclude future values when constructing predictors; and
- consider walk-forward or time-series cross-validation after the baseline pipeline is functioning correctly.

## 12. Initial Feasibility Test

The first stage will use XLK only and will examine:

- OHLCV data availability;
- missing observations;
- feature construction;
- the future five-day realised-volatility target;
- class distribution;
- the sample size remaining after rolling calculations; and
- one simple Logistic Regression baseline.

The initial outputs will be:

- a price chart;
- a rolling-volatility chart;
- a target-class distribution;
- a small modelling dataframe; and
- a baseline evaluation table.

## 13. Interesting Aspect and Potential Contribution

The study moves beyond the comparison of predictive scores obtained from clean data by investigating how different model classes respond to controlled perturbations. It may identify the feature groups that cause the greatest deterioration and may reveal a trade-off between clean-data performance and robustness. The nature and strength of any contribution will depend on the empirical findings.

## 14. Decisions Not Yet Finalised

| Item | Status |
|---|---|
| Data source | To be confirmed after feasibility testing or supervisor feedback |
| Sample period | To be confirmed after feasibility testing or supervisor feedback |
| Final ETF universe | To be confirmed after feasibility testing or supervisor feedback |
| Exact volatility formula | To be confirmed after feasibility testing or supervisor feedback |
| High-volatility threshold | To be confirmed after feasibility testing or supervisor feedback |
| Feature lookback windows | To be confirmed after feasibility testing or supervisor feedback |
| Noise distribution | To be confirmed after feasibility testing or supervisor feedback |
| Noise intensity levels | To be confirmed after feasibility testing or supervisor feedback |
| Number of random repetitions | To be confirmed after feasibility testing or supervisor feedback |
| Final third model | To be confirmed after feasibility testing or supervisor feedback |
| Final validation method | To be confirmed after feasibility testing or supervisor feedback |

## 15. Immediate Next Step

Construct and validate a minimal XLK dataset containing dates, selected features and the next-week high-volatility target before expanding the empirical scope.
