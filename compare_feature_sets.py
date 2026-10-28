
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_synthetic_repeated import load_real
from features import build_features, groups_for_cv

OUT = Path("synthetic_experiment/feature_comparison")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    data = load_real()
    X, _ = build_features(data)
    X = X.reset_index(drop=True)
    y = data["log_ratio"].astype(float).reset_index(drop=True)

    groups, _ = groups_for_cv(data)
    groups = pd.Series(groups).reset_index(drop=True)

    feature_sets = {
        "full": list(X.columns),
        "distance_focused": [
            c for c in X.columns
            if c in ["magnitude", "log_rrup", "log_vs30"]
            or c.startswith("mech_")
        ],
        "minimal": ["log_rrup", "log_vs30"],
        "minimal_plus_magnitude": [
            "log_rrup",
            "log_vs30",
            "magnitude",
        ],
    }

    models = {
        "Ridge_alpha_10": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "ElasticNet_alpha_0.01_l1_0.1": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.1,
                max_iter=20000,
            ),
        ),
    }

    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    results = []

    for feature_name, columns in feature_sets.items():
        X_subset = X[columns]

        for model_name, template in models.items():
            predictions = np.full(len(y), np.nan)
            fold_rows = []

            for fold, (tr, te) in enumerate(
                cv.split(X_subset, y, groups), start=1
            ):
                model = clone(template)
                model.fit(X_subset.iloc[tr], y.iloc[tr])
                pred = model.predict(X_subset.iloc[te])
                predictions[te] = pred

                fold_rows.append({
                    "feature_set": feature_name,
                    "model": model_name,
                    "fold": fold,
                    "mae": mean_absolute_error(y.iloc[te], pred),
                    "rmse": np.sqrt(
                        mean_squared_error(y.iloc[te], pred)
                    ),
                    "r2": r2_score(y.iloc[te], pred),
                })

            valid = np.isfinite(predictions)
            results.append({
                "feature_set": feature_name,
                "model": model_name,
                "features": ", ".join(columns),
                "oof_mae": mean_absolute_error(
                    y[valid], predictions[valid]
                ),
                "oof_rmse": np.sqrt(
                    mean_squared_error(y[valid], predictions[valid])
                ),
                "oof_r2": r2_score(y[valid], predictions[valid]),
                "mean_fold_mae": np.mean(
                    [r["mae"] for r in fold_rows]
                ),
                "std_fold_mae": np.std(
                    [r["mae"] for r in fold_rows]
                ),
                "mean_fold_r2": np.mean(
                    [r["r2"] for r in fold_rows]
                ),
            })

    summary = pd.DataFrame(results).sort_values("oof_mae")
    summary.to_csv(OUT / "feature_set_results.csv", index=False)

    print("\nFEATURE SET COMPARISON")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results to {OUT / 'feature_set_results.csv'}")


if __name__ == "__main__":
    main()
