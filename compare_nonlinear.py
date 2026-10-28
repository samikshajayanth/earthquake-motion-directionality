
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from compare_synthetic_repeated import load_real
from features import build_features, groups_for_cv

OUT = Path("synthetic_experiment/nonlinear_comparison")
OUT.mkdir(parents=True, exist_ok=True)


def make_linear():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=20000),
    )


def make_polynomial():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=20000),
    )


def main():
    data = load_real().reset_index(drop=True)
    features, _ = build_features(data)

    X = features[["log_rrup", "log_vs30"]].reset_index(drop=True)
    y = data["log_ratio"].astype(float).reset_index(drop=True)

    groups, _ = groups_for_cv(data)
    groups = pd.Series(groups).reset_index(drop=True)

    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    models = {
        "linear_elastic_net": make_linear(),
        "polynomial_elastic_net": make_polynomial(),
    }

    rows = []

    for name, template in models.items():
        predictions = np.full(len(y), np.nan)
        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):
            model = clone(template)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            predictions[test_idx] = pred

            fold_rows.append({
                "model": name,
                "fold": fold,
                "mae": mean_absolute_error(y.iloc[test_idx], pred),
                "rmse": np.sqrt(
                    mean_squared_error(y.iloc[test_idx], pred)
                ),
                "r2": r2_score(y.iloc[test_idx], pred),
            })

        valid = np.isfinite(predictions)

        rows.append({
            "model": name,
            "oof_mae": mean_absolute_error(y[valid], predictions[valid]),
            "oof_rmse": np.sqrt(
                mean_squared_error(y[valid], predictions[valid])
            ),
            "oof_r2": r2_score(y[valid], predictions[valid]),
            "mean_fold_mae": np.mean([r["mae"] for r in fold_rows]),
            "std_fold_mae": np.std([r["mae"] for r in fold_rows]),
            "mean_fold_r2": np.mean([r["r2"] for r in fold_rows]),
            "std_fold_r2": np.std([r["r2"] for r in fold_rows]),
        })

    summary = pd.DataFrame(rows).sort_values("oof_mae")
    summary.to_csv(OUT / "results.csv", index=False)

    print("\nLINEAR VS NONLINEAR REGRESSION")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
