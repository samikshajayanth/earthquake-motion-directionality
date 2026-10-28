from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

out = Path("synthetic_experiment/repeated_validation")
cls = pd.read_csv(out / "classification_all_splits.csv")
reg = pd.read_csv(out / "regression_all_splits.csv")

# Classification: mean ROC-AUC and standard deviation across splits
summary = (
    cls.groupby(["training_mode", "model"])["roc_auc"]
    .agg(["mean", "std"]).reset_index()
)
modes = [
    "real_only",
    "real_plus_synthetic_25pct",
    "real_plus_synthetic_50pct",
    "real_plus_synthetic_100pct",
]
labels = ["Real only", "25% synthetic", "50% synthetic", "100% synthetic"]

fig, ax = plt.subplots(figsize=(10, 6))
for model in ["Logistic regression", "Random forest", "Gradient boosting"]:
    part = summary[summary["model"] == model].set_index("training_mode")
    means = [part.loc[m, "mean"] if m in part.index else np.nan for m in modes]
    stds = [part.loc[m, "std"] if m in part.index else 0 for m in modes]
    ax.errorbar(labels, means, yerr=stds, marker="o", capsize=4, label=model)

ax.axhline(0.5, linestyle="--", label="Chance-level AUC")
ax.set_ylabel("ROC-AUC")
ax.set_title("Classification Performance by Synthetic Augmentation")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig(out / "final_classification_auc.png", dpi=200)
plt.close(fig)

# Classification: threshold-dependent metrics for logistic regression
lr = cls[cls["model"] == "Logistic regression"]
metrics = ["precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(labels))
width = 0.24

for i, metric in enumerate(metrics):
    stats = lr.groupby("training_mode")[metric].agg(["mean", "std"])
    means = [stats.loc[m, "mean"] if m in stats.index else np.nan for m in modes]
    stds = [stats.loc[m, "std"] if m in stats.index else 0 for m in modes]
    ax.bar(x + (i - 1) * width, means, width, yerr=stds,
           capsize=3, label=metric.capitalize())

ax.set_xticks(x, labels)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Logistic Regression: Precision, Recall and F1")
ax.legend()
fig.tight_layout()
fig.savefig(out / "final_classification_metrics.png", dpi=200)
plt.close(fig)

# Regression: MAE comparison
r = reg[reg["model"] == "Ridge regression"]
stats = r.groupby("training_mode")["mae_log_ratio"].agg(["mean", "std"])
means = [stats.loc[m, "mean"] if m in stats.index else np.nan for m in modes]
stds = [stats.loc[m, "std"] if m in stats.index else 0 for m in modes]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labels, means, yerr=stds, capsize=4)
ax.set_ylabel("MAE (log directionality ratio)")
ax.set_title("Ridge Regression Error by Training Approach")
fig.tight_layout()
fig.savefig(out / "final_regression_mae.png", dpi=200)
plt.close(fig)

print("Created:")
for name in [
    "final_classification_auc.png",
    "final_classification_metrics.png",
    "final_regression_mae.png",
]:
    print(out / name)
