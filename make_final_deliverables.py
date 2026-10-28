from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

root = Path(".")
out = root / "outputs_final_deliverables"
out.mkdir(exist_ok=True)

tuned = root / "outputs_tuned"
summary = json.loads((tuned / "nested_cv_summary.json").read_text())
cls_folds = pd.read_csv(tuned / "classification_outer_folds.csv")
reg_folds = pd.read_csv(tuned / "regression_outer_folds.csv")
params = pd.read_csv(tuned / "selected_parameters_by_fold.csv")

def get_metrics(task):
    for key in (task, f"{task}_metrics"):
        if key in summary and isinstance(summary[key], dict):
            return summary[key]
    raise ValueError(f"Could not find {task} metrics in nested_cv_summary.json")

cls = get_metrics("classification")
reg = get_metrics("regression")

metrics = pd.DataFrame([
    ["Classification", "Extra Trees", "ROC-AUC", cls["roc_auc"], "Higher is better"],
    ["Classification", "Extra Trees", "Accuracy", cls["accuracy"], "Higher is better"],
    ["Classification", "Extra Trees", "Precision", cls["precision"], "Higher is better"],
    ["Classification", "Extra Trees", "Recall", cls["recall"], "Higher is better"],
    ["Classification", "Extra Trees", "F1-score", cls["f1"], "Higher is better"],
    ["Regression", "Extra Trees", "R-squared", reg["r2"], "Higher is better"],
    ["Regression", "Extra Trees", "MAE (log ratio)", reg["mae_log_ratio"], "Lower is better"],
    ["Regression", "Extra Trees", "RMSE (log ratio)", reg["rmse_log_ratio"], "Lower is better"],
], columns=["Task", "Model", "Metric", "Value", "Interpretation"])

metrics.to_csv(out / "final_metrics.csv", index=False)
cls_folds.to_csv(out / "classification_fold_results.csv", index=False)
reg_folds.to_csv(out / "regression_fold_results.csv", index=False)
params.to_csv(out / "selected_parameters_by_fold.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(cls_folds["fold"].astype(str), cls_folds["roc_auc"])
plt.axhline(0.5, linestyle="--", label="Chance-level ROC-AUC")
plt.ylim(0, 1)
plt.xlabel("Outer cross-validation fold")
plt.ylabel("ROC-AUC")
plt.title("Classification Performance Across Folds")
plt.legend()
plt.tight_layout()
plt.savefig(out / "classification_roc_auc_by_fold.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(reg_folds["fold"].astype(str), reg_folds["r2"])
plt.axhline(0, linestyle="--", label="Zero R-squared")
plt.xlabel("Outer cross-validation fold")
plt.ylabel("R-squared")
plt.title("Regression Performance Across Folds")
plt.legend()
plt.tight_layout()
plt.savefig(out / "regression_r2_by_fold.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(reg_folds["fold"], reg_folds["mae_log_ratio"], marker="o", label="MAE")
plt.plot(reg_folds["fold"], reg_folds["rmse_log_ratio"], marker="s", label="RMSE")
plt.xlabel("Outer cross-validation fold")
plt.ylabel("Error (log-ratio units)")
plt.title("Regression Errors Across Folds")
plt.legend()
plt.tight_layout()
plt.savefig(out / "regression_errors_by_fold.png", dpi=200)
plt.close()

report = f"""# Earthquake Ground-Motion Directionality: Final Results

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
- ROC-AUC: {cls['roc_auc']:.4f}
- Accuracy: {cls['accuracy']:.4f}
- Precision: {cls['precision']:.4f}
- Recall: {cls['recall']:.4f}
- F1-score: {cls['f1']:.4f}

### Regression — Extra Trees
- R-squared: {reg['r2']:.4f}
- MAE in log-ratio units: {reg['mae_log_ratio']:.4f}
- RMSE in log-ratio units: {reg['rmse_log_ratio']:.4f}

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
"""

(out / "final_results_and_limitations.md").write_text(report)
print("Created deliverables in:", out)
for file in sorted(out.iterdir()):
    print(f"{file.name} ({file.stat().st_size:,} bytes)")
