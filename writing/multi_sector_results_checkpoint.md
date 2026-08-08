# Multi-sector results checkpoint

## Clean predictive performance

The clean baselines are stable across sectors, but the winning model is not uniform. Logistic Regression retains an edge in some ETFs, while Random Forest is stronger in others. The verified XLK and XLE outputs remain unchanged.

## Moderate-noise robustness

Across the primary moderate-noise range (0.05–0.20), the paired simulation-interval classification records 307 Random Forest-favoured conditions, 294 Logistic Regression-favoured conditions, and 344 conditions with no clear paired simulation difference. These intervals quantify within-condition Monte Carlo uncertainty across the 30 noise seeds; they do not provide population-level inference across sectors. The pattern is mixed and sector-dependent rather than universal.

## Stress-test robustness

Within the exploratory bounded-input stress tests (0.50–1.00), Random Forest is favoured in 406 conditions, Logistic Regression in 105 conditions, and 119 conditions show no clear paired simulation difference. Clipping is materially larger here, so these cases should be read as bounded-input stress scenarios.

## Feature-group sensitivity

Volatility perturbations are often damaging, but they are not uniformly the worst feature group in every sector and metric. Returns are frequently less disruptive, but that is not consistent enough to state as a universal rule.

## Sector heterogeneity

Sector-level differences remain substantial. The same noise regime can favour different models depending on the ETF, and many condition-level intervals include zero because the estimated paired differences are small relative to within-condition simulation variability. The nine sector ETFs, rather than the 30 seeds, are the units for cross-sector generalisation.

## Clipping-related limitations

Clipping becomes more frequent at higher noise intensities, especially in the stress-test range. Those results are useful as bounded-input stress tests, but they are not unconstrained perturbations of the underlying feature distributions.

## Apparent improvements under noise

The negative-degradation cases examined in XLF and XLU are consistent with threshold crossing around the fixed 0.5 decision rule and seed-to-seed variability. The diagnostics do not indicate a data-leakage problem or a modelling implementation error, but the apparent gains are not strong enough to be treated as substantive beneficial-noise effects.

## Validation summary

Validation status is PASS for all nine ETFs. Each sector has two principal clean-model metric rows, one for Logistic Regression and one for Random Forest. The purge length remains five observations in every sector, and the reconstructed feature datasets match the saved modelling files within tolerance. XLK's separate Dummy Classifier result is an intentional auxiliary baseline and is excluded from the principal-model count.

## Clipping totals

High-intensity clipping remains the main practical limitation of the exploratory stress tests. The audit therefore distinguishes moderate-noise evidence from bounded-input stress evidence rather than collapsing them into one headline average.

## Clean refit drift

A fresh refit of XLF Random Forest shows a small numerical drift relative to the stored clean baseline, while Logistic Regression matches exactly. The cause cannot be established conclusively because the stored probability vector and complete historical software environment are unavailable; software or environment differences remain a plausible explanation. The drift is limited to the refit comparison and does not change the broader audit classification, but it is recorded explicitly as a reproducibility caveat.
