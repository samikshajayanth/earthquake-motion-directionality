# Earthquake Ground-Motion Directionality: Final Results

## Objective
Investigate whether spectral characteristics at periods other than 1.0 second
contain information useful for classifying and estimating directionality at 1.0 second.

## Target definition
The directionality ratio is RotD100 / RotD50 at T = 1.0 second.
Records with a ratio >= 1.3 are labelled as strong-directionality records.
The target-defining 1.0-second ordinates were excluded from predictor features.

## Dataset
The cached response-spectrum dataset contains 196 records.
The cache does not currently contain earthquake event identifiers or all
metadata required for event-independent validation.

## Nested cross-validation results

### Classification — Extra Trees
- ROC-AUC: 0.6479
- Accuracy: 0.6480
- Precision: 0.4737
- Recall: 0.5538
- F1-score: 0.5106

### Regression — Extra Trees
- R-squared: 0.0634
- MAE in log-ratio units: 0.0733
- RMSE in log-ratio units: 0.0854

## Interpretation
Classification performance is modest. Regression explains only a small
fraction of target variation, so continuous prediction remains limited.
Performance also varies across the outer folds.

## Limitations
1. Only 196 cached records are currently used.
2. Earthquake event identifiers are unavailable in the cache, preventing
   validation that holds out entire earthquakes.
3. Record-level cross-validation may overestimate performance on unseen events
   when records from related events occur in both training and test folds.
4. Regression R-squared is low, indicating limited explanatory and predictive power.
5. Results are exploratory and should not be presented as a validated operational
   earthquake prediction system.

## Future work
1. Recover event identifiers and relevant metadata from the original flatfiles.
2. Use grouped cross-validation that holds out entire earthquake events.
3. Evaluate class balance, calibration, uncertainty, and performance on independent data.
4. Compare against simple baselines and report fold-wise variability.
5. Investigate whether additional physically justified features improve generalization.

## Reproducibility
The classification and regression fold results, selected parameters, final metrics,
and PNG figures are provided alongside this report.
