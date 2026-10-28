
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_synthetic_repeated import load_real
from features import build_features, groups_for_cv


OUT = Path("synthetic_experiment/residual_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.1,
            max_iter=20000,
        ),
    )


def main():
    data = load_real().reset_index(drop=True)
    X, _ = build_features(data)
    X = X.reset_index(drop=True)
    y = data["log_ratio"].astype(float).reset_index(drop=True)

    groups, _ = groups_for_cv(data)
    groups = pd.Series(groups).reset_index(drop=True)

    # Leading feature set from the previous comparison.
    columns = ["log_rrup", "log_vs30"]
    X = X[columns]

    n_splits = min(5, groups.nunique())
    if n_splits < 2:
        raise ValueError("At least two independent earthquake groups are required.")

    cv = GroupKFold(n_splits=n_splits)
    predictions = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1)

    # Produce one held-out prediction for every record.
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        model = clone(make_model())
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
        fold_ids[test_idx] = fold

    valid = np.isfinite(predictions)
    if not valid.all():
        raise RuntimeError("Some records did not receive an out-of-fold prediction.")

    residuals = y.to_numpy() - predictions

    results = data[["rsn", "event_id", "ratio", "log_ratio"]].copy()
    results["actual_log_ratio"] = y
    results["predicted_log_ratio"] = predictions
    results["residual"] = residuals
    results["absolute_error"] = np.abs(residuals)
    results["outer_fold"] = fold_ids
    results.to_csv(OUT / "out_of_fold_predictions.csv", index=False)

    # Aggregate metrics over all out-of-fold predictions.
    overall = pd.DataFrame([{
        "n_records": len(y),
        "n_events": groups.nunique(),
        "oof_mae": mean_absolute_error(y, predictions),
        "oof_rmse": np.sqrt(mean_squared_error(y, predictions)),
        "pooled_oof_r2": r2_score(y, predictions),
    }])
    overall.to_csv(OUT / "overall_metrics.csv", index=False)

    # 1. Actual versus predicted.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, predictions, alpha=0.75)
    low = min(float(y.min()), float(predictions.min()))
    high = max(float(y.max()), float(predictions.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual log(RotD100 / RotD50)")
    ax.set_ylabel("Predicted log(RotD100 / RotD50)")
    ax.set_title("Out-of-fold predictions: actual vs predicted")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 2. Residuals versus predicted values.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predictions, residuals, alpha=0.75)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted log-ratio")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Out-of-fold residuals versus predictions")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "residuals_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 3. Residual distribution.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(residuals, bins=15, edgecolor="black")
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Number of records")
    ax.set_title("Distribution of out-of-fold residuals")
    fig.tight_layout()
    fig.savefig(OUT / "residual_distribution.png", dpi=200)
    plt.close(fig)

    # 4. Performance by earthquake event.
    event_rows = []
    for event, idx in data.groupby("event_id").groups.items():
        idx = np.asarray(list(idx), dtype=int)
        actual = y.iloc[idx].to_numpy()
        pred = predictions[idx]

        event_rows.append({
            "event_id": event,
            "n_records": len(idx),
            "mae": mean_absolute_error(actual, pred),
            "rmse": np.sqrt(mean_squared_error(actual, pred)),
            "mean_residual": float(np.mean(actual - pred)),
        })

    event_metrics = pd.DataFrame(event_rows).sort_values(
        "mae", ascending=False
    )
    event_metrics.to_csv(OUT / "event_level_metrics.csv", index=False)

    # Plot events with the highest MAE.
    top = event_metrics.head(15).sort_values("mae")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["event_id"].astype(str), top["mae"])
    ax.set_xlabel("MAE on held-out predictions")
    ax.set_ylabel("Earthquake event")
    ax.set_title("Events with the highest out-of-fold MAE")
    fig.tight_layout()
    fig.savefig(OUT / "event_level_mae.png", dpi=200)
    plt.close(fig)

    print("\nOVERALL OUT-OF-FOLD METRICS")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nEVENT-LEVEL RESULTS — HIGHEST MAE FIRST")
    print(event_metrics.head(15).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    ))

    print(f"\nAll files saved in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
