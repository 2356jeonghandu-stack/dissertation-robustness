# Reproducibility audit note

## XLK clean-metric row count

The XLK clean-metric table contains three rows because `outputs/tables/xlk_clean_model_comparison.csv` includes the intentional Dummy Classifier baseline alongside the two principal models. The validation count is now defined to include only Logistic Regression and Random Forest, so `clean_metric_row_count` is 2 for XLK as well as for the other sectors.

## XLF Random Forest reproducibility

A fresh XLF Random Forest refit in the current environment is deterministic across repeat fits, with a maximum absolute probability difference of 3.331e-16 between two immediate refits. The fitted configuration, predictor ordering, training ordering, class weights, tree count, and random state all match the stored specification.

However, the stored XLF Random Forest baseline metrics correspond to an older fitted artefact: the fresh refit produces accuracy 0.853162650602 versus the stored 0.852409638554, and the inferred confusion matrix changes from TN=1075, FP=86, FN=110, TP=57 to TN=1075, FP=86, FN=109, TP=58.

The raw stored probability vector and the complete historical software environment are not persisted in the project, so an exact stored-versus-fresh probability-vector comparison cannot be made and the cause cannot be established conclusively. The fresh fit is internally stable and the available checks do not identify a predictor-ordering, observation-ordering or stated-parameter mismatch. Software-version or wider environment differences are therefore a plausible explanation for the small numerical drift, rather than a confirmed cause. The current environment reports scikit-learn 1.5.1, NumPy 1.26.4 and pandas 2.2.2.

## Impact assessment

No downstream sector analysis needed to be rerun for the audit itself. The established XLF clean-baseline artefacts remain internally coherent with the stored XLF clean baseline and the downstream noise tables, and the validation-summary correction does not change any empirical conclusion. The only correction applied was to the validation definition so that clean-metric row counts reflect the two principal models only.

## Readiness for Results drafting

The project is ready for formal Results drafting, with two caveats recorded explicitly: the XLK validation count now excludes the Dummy Classifier baseline, and the XLF Random Forest clean-baseline drift remains a small unresolved reproducibility limitation for which software or environment differences are a plausible explanation.
