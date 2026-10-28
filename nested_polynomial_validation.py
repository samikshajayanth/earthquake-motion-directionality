
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

OUT = Path("synthetic_experiment/nested_polynomial_validation")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("linear", False, 0.001, 0.1),
    ("linear", False, 0.01, 0.1),
    ("linear", False, 0.01, 0.5),
    ("polynomial", True, 0.001, 0.1),
    ("polynomial", True, 0.01, 0.1),
    ("polynomial", True, 0.01, 0.5),
    ("polynomial", True, 0.1, 0.1),
]


def make_model(polynomial, alpha, l1_ratio):
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if polynomial:
        steps.append((
            "poly",
            PolynomialFeatures(degree=2, include_bias=False)
        ))

    steps.extend([
        ("scale", StandardScaler()),
        ("model", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20000,
        )),
    ])

    from sklearn.pipeline import Pipeline
    return Pipeline(steps)


def main():
    data = load_real().reset_index(drop=True)
    features, _ = build_features(data)

    X = features[["log_rrup", "log_vs30"]].reset_index(drop=True)
    y = data["log_ratio"].astype(float).reset_index(drop=True)

    group_values, _ = groups_for_cv(data)
    groups = pd.Series(group_values).reset_index(drop=True)

    n_groups = groups.nunique()
    if n_groups < 6:
        raise ValueError(f"Need at least 6 groups; found {n_groups}.")

    outer = GroupKFold(n_splits=min(5, n_groups))
    rows = []

    for outer_fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        gtr = groups.iloc[tr]

        inner = GroupKFold(n_splits=min(4, gtr.nunique()))
        candidate_scores = []

        # Select model type and parameters using outer-training events only.
        for name, polynomial, alpha, l1_ratio in CANDIDATES:
            fold_maes = []

            for itr, iva in inner.split(Xtr, ytr, gtr):
                model = make_model(polynomial, alpha, l1_ratio)
                model.fit(Xtr.iloc[itr], ytr.iloc[itr])
                pred = model.predict(Xtr.iloc[iva])
                fold_maes.append(
                    mean_absolute_error(ytr.iloc[iva], pred)
                )

            candidate_scores.append((
                float(np.mean(fold_maes)),
                name, polynomial, alpha, l1_ratio
            ))

        inner_mae, name, polynomial, alpha, l1_ratio = min(
            candidate_scores, key=lambda item: item[0]
        )

        model = make_model(polynomial, alpha, l1_ratio)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        rows.append({
            "outer_fold": outer_fold,
            "selected_model": name,
            "polynomial": polynomial,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "inner_mae": inner_mae,
            "n_test_records": len(te),
            "n_test_events": groups.iloc[te].nunique(),
            "mae": mean_absolute_error(yte, pred),
            "rmse": np.sqrt(mean_squared_error(yte, pred)),
            "r2": r2_score(yte, pred),
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "outer_fold_results.csv", index=False)

    summary = results[["mae", "rmse", "r2"]].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")

    print("\nNESTED VALIDATION — LINEAR/POLYNOMIAL SELECTION")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nOUTER-FOLD METRICS (mean and standard deviation)")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nSelected model counts:")
    print(results["selected_model"].value_counts().to_string())
    print(f"\nSaved results in: {OUT.resolve()}")


if __name__ == "__main__":
    main()
