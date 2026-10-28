
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_synthetic_repeated import load_real
from features import build_features, groups_for_cv

OUT = Path("synthetic_experiment/nested_validation")
OUT.mkdir(parents=True, exist_ok=True)


def make_model(alpha=0.01, l1_ratio=0.1, baseline=False):
    if baseline:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            DummyRegressor(strategy="mean"),
        )

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
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

    n_events = groups.nunique()
    if n_events < 6:
        raise ValueError(
            f"Nested validation needs at least 6 groups; found {n_events}."
        )

    outer = GroupKFold(n_splits=min(5, n_events))
    inner_candidates = [
        (0.001, 0.1),
        (0.001, 0.5),
        (0.001, 0.9),
        (0.01, 0.1),
        (0.01, 0.5),
        (0.01, 0.9),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.1, 0.9),
        (1.0, 0.1),
        (1.0, 0.5),
        (1.0, 0.9),
    ]

    feature_sets = {
        "full": list(X.columns),
        "minimal": ["log_rrup", "log_vs30"],
    }

    fold_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer.split(X, y, groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        # Baseline: predict the mean of the outer training fold.
        baseline = make_model(baseline=True)
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)

        fold_rows.append({
            "outer_fold": outer_fold,
            "feature_set": "baseline",
            "model": "training_mean",
            "test_events": g_train.nunique() * 0
            + groups.iloc[test_idx].nunique(),
            "mae": mean_absolute_error(y_test, baseline_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, baseline_pred)),
            "r2": r2_score(y_test, baseline_pred),
        })

        inner = GroupKFold(n_splits=min(4, g_train.nunique()))

        for feature_name, columns in feature_sets.items():
            best_params = None
            best_inner_mae = np.inf

            # All parameter selection stays inside this outer training fold.
            for alpha, l1_ratio in inner_candidates:
                inner_maes = []

                for itrain, ival in inner.split(
                    X_train[columns], y_train, g_train
                ):
                    model = make_model(alpha, l1_ratio)
                    model.fit(
                        X_train[columns].iloc[itrain],
                        y_train.iloc[itrain],
                    )
                    pred = model.predict(X_train[columns].iloc[ival])
                    inner_maes.append(
                        mean_absolute_error(y_train.iloc[ival], pred)
                    )

                candidate_mae = float(np.mean(inner_maes))
                if candidate_mae < best_inner_mae:
                    best_inner_mae = candidate_mae
                    best_params = (alpha, l1_ratio)

            alpha, l1_ratio = best_params
            final_model = make_model(alpha, l1_ratio)
            final_model.fit(X_train[columns], y_train)
            pred = final_model.predict(X_test[columns])

            fold_rows.append({
                "outer_fold": outer_fold,
                "feature_set": feature_name,
                "model": f"ElasticNet(alpha={alpha}, l1={l1_ratio})",
                "selected_alpha": alpha,
                "selected_l1_ratio": l1_ratio,
                "inner_mae": best_inner_mae,
                "test_events": groups.iloc[test_idx].nunique(),
                "mae": mean_absolute_error(y_test, pred),
                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                "r2": r2_score(y_test, pred),
            })

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "outer_fold_results.csv", index=False)

    summary = (
        fold_df.groupby(["feature_set", "model"])[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "summary.csv")

    print("\nNESTED GROUPED VALIDATION — OUTER FOLDS")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved fold results: {OUT / 'outer_fold_results.csv'}")
    print(f"Saved summary: {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
