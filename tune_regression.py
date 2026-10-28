
"""Tune Ridge and Elastic Net with earthquake-grouped validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_synthetic_repeated import load_real
from features import build_features, groups_for_cv


OUT = Path("synthetic_experiment/regression_tuning")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    data = load_real()

    X, feature_names = build_features(data)
    y = data["log_ratio"].astype(float).reset_index(drop=True)
    X = X.reset_index(drop=True)

    groups, _ = groups_for_cv(data)
    groups = pd.Series(groups).reset_index(drop=True)

    n_groups = groups.nunique()
    if n_groups < 3:
        raise ValueError(
            f"Need at least 3 independent groups; found {n_groups}."
        )

    cv = GroupKFold(n_splits=min(5, n_groups))

    candidates = []

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((
            f"Ridge_alpha_{alpha}",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=alpha),
            ),
        ))

    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            candidates.append((
                f"ElasticNet_alpha_{alpha}_l1_{l1_ratio}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=20000,
                    ),
                ),
            ))

    results = []

    for name, template in candidates:
        oof_pred = np.full(len(y), np.nan)
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            model = clone(template)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx]

            oof_pred[test_idx] = pred

            fold_metrics.append({
                "model": name,
                "fold": fold,
                "test_events": groups.iloc[test_idx].nunique(),
                "mae": mean_absolute_error(actual, pred),
                "rmse": np.sqrt(mean_squared_error(actual, pred)),
                "r2": r2_score(actual, pred),
            })

        valid = np.isfinite(oof_pred)
        results.append({
            "model": name,
            "oof_mae": mean_absolute_error(y[valid], oof_pred[valid]),
            "oof_rmse": np.sqrt(
                mean_squared_error(y[valid], oof_pred[valid])
            ),
            "oof_r2": r2_score(y[valid], oof_pred[valid]),
            "mean_fold_mae": np.mean([m["mae"] for m in fold_metrics]),
            "std_fold_mae": np.std([m["mae"] for m in fold_metrics]),
            "mean_fold_r2": np.mean([m["r2"] for m in fold_metrics]),
            "std_fold_r2": np.std([m["r2"] for m in fold_metrics]),
        })

    summary = pd.DataFrame(results).sort_values(
        ["oof_mae", "oof_rmse"]
    )
    summary.to_csv(OUT / "tuning_summary.csv", index=False)

    print("\nREGRESSION TUNING RESULTS — LOWER MAE IS BETTER")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {OUT / 'tuning_summary.csv'}")
    print(f"Features used: {feature_names}")


if __name__ == "__main__":
    main()
