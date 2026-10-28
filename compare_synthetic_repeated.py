"""Compare real-only vs training-only synthetic augmentation.

The held-out real earthquake events are never used to generate synthetic
training records. Existing project files and baseline outputs are preserved.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, r2_score
)

from features import build_features, make_labels, groups_for_cv
from models import classifiers, regressors, SEED

REAL_CACHE = Path("outputs_foldcheck/rotd_cache.csv")
METADATA = Path("combined_input/_SearchResults.csv")
OUT = Path("synthetic_experiment")
THRESHOLD = 1.3
N_SYNTHETIC = 500


def encode_mechanism(series):
    s = series.astype(str).str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.str.contains("strike", na=False)] = 0
    out[s.str.contains("reverse oblique|reverse-oblique", regex=True, na=False)] = 3
    out[s.str.contains("normal oblique|normal-oblique", regex=True, na=False)] = 4
    out[s.str.contains("reverse|thrust", regex=True, na=False)] = 2
    out[s.str.contains("normal", na=False) & out.isna()] = 1
    numeric = pd.to_numeric(series, errors="coerce")
    out[numeric.notna()] = numeric[numeric.notna()]
    return out


def load_real():
    cache = pd.read_csv(REAL_CACHE)
    meta = pd.read_csv(METADATA)

    meta = meta.rename(columns={
        "Record Sequence Number": "rsn",
        "Earthquake Name": "event_id",
        "Magnitude": "magnitude",
        "Rrup (km)": "rrup",
        "Rjb (km)": "rjb",
        "Vs30 (m/sec)": "vs30",
        "Mechanism": "mechanism",
    })
    cols = [
        "rsn", "event_id", "magnitude", "rrup",
        "rjb", "vs30", "mechanism"
    ]
    meta = meta[cols].copy()
    meta["rsn"] = pd.to_numeric(meta["rsn"], errors="coerce")
    cache["rsn"] = pd.to_numeric(cache["rsn"], errors="coerce")
    meta = meta.dropna(subset=["rsn"]).drop_duplicates("rsn")

    data = cache.merge(meta, on="rsn", how="inner", validate="one_to_one")
    data["mechanism"] = encode_mechanism(data["mechanism"])
    data["ratio"] = data["RotD100_T1.000"] / data["RotD50_T1.000"]
    data["log_ratio"] = np.log(data["ratio"])
    for c in ["magnitude", "rrup", "rjb", "vs30"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=["event_id", "ratio", "log_ratio", "magnitude", "vs30"]
    )
    return data.reset_index(drop=True)


def make_training_synthetic(train, n, seed):
    """Bootstrap and perturb training records only."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(train), size=n)
    syn = train.iloc[idx].reset_index(drop=True).copy()

    # Make it clear these are generated records, not real RSNs/events.
    syn["rsn"] = [f"SYN_TRAIN_{i:06d}" for i in range(1, n + 1)]
    syn["data_source"] = "synthetic_training_only"
    syn["source_rsn"] = train.iloc[idx]["rsn"].to_numpy()

    # Perturb input features, not target labels independently.
    for col, sigma in [
        ("magnitude", 0.01),
        ("rrup", 0.02),
        ("rjb", 0.02),
        ("vs30", 0.02),
    ]:
        values = pd.to_numeric(syn[col], errors="coerce").to_numpy(float)
        values = values * np.exp(rng.normal(0, sigma, n))
        if col in ("rrup", "rjb"):
            values = np.maximum(values, 0.1)
        elif col == "vs30":
            values = np.maximum(values, 50)
        syn[col] = values

    # Slightly perturb positive response spectra, maintaining RotD100 >= RotD50.
    for period in [
        "0.050", "0.100", "0.200", "0.300", "0.500", "0.750",
        "1.000", "1.500", "2.000", "3.000", "4.000", "5.000",
        "7.500", "10.000"
    ]:
        c50, c100 = f"RotD50_T{period}", f"RotD100_T{period}"
        if c50 not in syn or c100 not in syn:
            continue
        old50 = np.maximum(syn[c50].to_numpy(float), 1e-10)
        old100 = np.maximum(syn[c100].to_numpy(float), old50)
        ratio = old100 / old50
        new50 = old50 * np.exp(rng.normal(0, 0.02, n))
        new_ratio = np.maximum(
            ratio * np.exp(rng.normal(0, 0.01, n)),
            1.0
        )
        syn[c50] = new50
        syn[c100] = new50 * new_ratio

    syn["ratio"] = syn["RotD100_T1.000"] / syn["RotD50_T1.000"]
    syn["log_ratio"] = np.log(syn["ratio"])
    return syn


def choose_oof_threshold(X, y, groups, template):
    """Select an F1 threshold from grouped out-of-fold training predictions."""
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    if y.nunique() < 2 or groups.nunique() < 3:
        return 0.5, np.nan

    n_splits = min(4, groups.nunique())
    oof_prob = np.full(len(y), np.nan)

    cv = GroupKFold(n_splits=n_splits)

    for fit_idx, val_idx in cv.split(X, y, groups):
        y_fit = y.iloc[fit_idx]
        if y_fit.nunique() < 2:
            continue

        model = clone(template)
        model.fit(X.iloc[fit_idx], y_fit)
        oof_prob[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    valid = np.isfinite(oof_prob)
    if valid.sum() == 0 or y.iloc[np.where(valid)[0]].nunique() < 2:
        return 0.5, np.nan

    y_valid = y.iloc[np.where(valid)[0]].to_numpy()
    p_valid = oof_prob[valid]

    candidates = np.arange(0.20, 0.81, 0.01)
    scores = [
        f1_score(y_valid, (p_valid >= t).astype(int), zero_division=0)
        for t in candidates
    ]
    best_idx = int(np.argmax(scores))
    return float(candidates[best_idx]), float(scores[best_idx])


def evaluate(train, test, threshold, mode):
    X_train, feature_names = build_features(train)
    X_test, _ = build_features(test)
    X_test = X_test.reindex(columns=feature_names)

    ytr_cls = (train["ratio"] >= threshold).astype(int)
    yte_cls = (test["ratio"] >= threshold).astype(int)
    ytr_reg = train["log_ratio"].astype(float)
    yte_reg = test["log_ratio"].astype(float)

    # Keep each earthquake together during inner threshold selection.
    train_groups, _ = groups_for_cv(train)
    train_groups = pd.Series(train_groups).reset_index(drop=True)

    cls_rows, reg_rows = [], []
    for name, template in classifiers().items():
        best_threshold, oof_f1 = choose_oof_threshold(
            X_train.reset_index(drop=True),
            ytr_cls.reset_index(drop=True),
            train_groups,
            template,
        )

        model = clone(template)
        model.fit(X_train, ytr_cls)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= best_threshold).astype(int)

        cls_rows.append({
            "training_mode": mode,
            "model": name,
            "decision_threshold": best_threshold,
            "oof_threshold_f1": oof_f1,
            "roc_auc": roc_auc_score(yte_cls, prob) if yte_cls.nunique() > 1 else np.nan,
            "accuracy": accuracy_score(yte_cls, pred),
            "precision": precision_score(yte_cls, pred, zero_division=0),
            "recall": recall_score(yte_cls, pred, zero_division=0),
            "f1": f1_score(yte_cls, pred, zero_division=0),
        })

    for name, template in regressors().items():
        model = clone(template)
        model.fit(X_train, ytr_reg)
        pred = model.predict(X_test)
        reg_rows.append({
            "training_mode": mode,
            "model": name,
            "r2": r2_score(yte_reg, pred),
            "rmse_log_ratio": float(np.sqrt(np.mean((yte_reg - pred) ** 2))),
            "mae_log_ratio": mean_absolute_error(yte_reg, pred),
        })

    return cls_rows, reg_rows, feature_names


def main():
    out = OUT / "repeated_validation"
    out.mkdir(parents=True, exist_ok=True)

    data = load_real()
    _, threshold = make_labels(data, threshold=THRESHOLD)
    groups, group_description = groups_for_cv(data)

    all_cls, all_reg, split_summaries = [], [], []
    n_splits = 5

    splitter = GroupShuffleSplit(
        n_splits=n_splits,
        test_size=0.25,
        random_state=SEED
    )

    for split_id, (tr_idx, te_idx) in enumerate(
        splitter.split(data, groups=groups), start=1
    ):
        train = data.iloc[tr_idx].reset_index(drop=True)
        test = data.iloc[te_idx].reset_index(drop=True)

        train_events = set(train["event_id"].astype(str))
        test_events = set(test["event_id"].astype(str))

        if train_events & test_events:
            raise RuntimeError(
                f"Event leakage detected in split {split_id}."
            )

        # Evaluate real-only and three training-only augmentation levels.
        base_cls, base_reg, features = evaluate(
            train, test, threshold, "real_only"
        )

        for row in base_cls + base_reg:
            row["split"] = split_id
            row["test_events"] = len(test_events)

        all_cls.extend(base_cls)
        all_reg.extend(base_reg)

        synthetic_counts = {}

        for percentage in (25, 50, 100):
            n_synthetic = max(1, round(len(train) * percentage / 100))
            synthetic_seed = SEED + split_id * 100 + percentage

            syn = make_training_synthetic(
                train, n_synthetic, synthetic_seed
            )

            if not set(syn["source_rsn"]).issubset(set(train["rsn"])):
                raise RuntimeError(
                    f"Synthetic source leakage in split {split_id}."
                )

            mode = f"real_plus_synthetic_{percentage}pct"
            augmented = pd.concat([train, syn], ignore_index=True)

            aug_cls, aug_reg, _ = evaluate(
                augmented, test, threshold, mode
            )

            for row in aug_cls + aug_reg:
                row["split"] = split_id
                row["test_events"] = len(test_events)

            all_cls.extend(aug_cls)
            all_reg.extend(aug_reg)
            synthetic_counts[mode] = len(syn)

        split_summaries.append({
            "split": split_id,
            "real_training_records": len(train),
            "real_test_records": len(test),
            "training_events": len(train_events),
            "held_out_test_events": len(test_events),
            "synthetic_records_by_mode": synthetic_counts,
            "train_test_events_disjoint": True,
            "synthetic_sources_from_training_only": True,
        })

        print(
            f"Finished split {split_id}/{n_splits}: "
            f"{len(train)} training records, "
            f"{len(test)} test records."
        )

    cls = pd.DataFrame(all_cls)
    reg = pd.DataFrame(all_reg)

    cls.to_csv(out / "classification_all_splits.csv", index=False)
    reg.to_csv(out / "regression_all_splits.csv", index=False)

    # Summarize metrics across splits; report variability as well as mean.
    cls_metrics = [
        "roc_auc", "accuracy", "precision", "recall", "f1"
    ]
    reg_metrics = ["r2", "rmse_log_ratio", "mae_log_ratio"]

    cls_summary = (
        cls.groupby(["training_mode", "model"])[cls_metrics]
        .agg(["mean", "std"])
        .round(3)
    )
    reg_summary = (
        reg.groupby(["training_mode", "model"])[reg_metrics]
        .agg(["mean", "std"])
        .round(3)
    )

    cls_summary.to_csv(out / "classification_mean_std.csv")
    reg_summary.to_csv(out / "regression_mean_std.csv")

    summary = {
        "real_records": len(data),
        "n_splits": n_splits,
        "test_size": 0.25,
        "synthetic_records_per_split": N_SYNTHETIC,
        "event_grouping": group_description,
        "threshold": float(threshold),
        "seed": SEED,
        "split_details": split_summaries,
        "limitations": [
            "Repeated random splits can have overlapping test events across splits.",
            "The test records within each split are real records only.",
            "Synthetic records are perturbed bootstrap samples, not new earthquakes.",
            "Repeated splits are not independent experiments.",
            "Performance differences should be interpreted cautiously."
        ]
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nCLASSIFICATION MEAN AND STANDARD DEVIATION")
    print(cls_summary.to_string())

    print("\nREGRESSION MEAN AND STANDARD DEVIATION")
    print(reg_summary.to_string())

    print(f"\nSaved repeated-validation results in {out.resolve()}")


if __name__ == "__main__":
    main()
