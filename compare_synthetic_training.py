"""Compare real-only vs training-only synthetic augmentation.

The held-out real earthquake events are never used to generate synthetic
training records. Existing project files and baseline outputs are preserved.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, r2_score
)

from features import build_features, make_labels, groups_for_cv
from models import classifiers, regressors, SEED

REAL_CACHE = Path("outputs_196/rotd_cache.csv")
METADATA = Path("combined_input/_SearchResults.csv")
OUT = Path("synthetic_experiment")
THRESHOLD = 1.3367660281644023
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


def evaluate(train, test, threshold, mode):
    X_train, feature_names = build_features(train)
    X_test, _ = build_features(test)
    X_test = X_test.reindex(columns=feature_names)

    ytr_cls = (train["ratio"] >= threshold).astype(int)
    yte_cls = (test["ratio"] >= threshold).astype(int)
    ytr_reg = train["log_ratio"].astype(float)
    yte_reg = test["log_ratio"].astype(float)

    cls_rows, reg_rows = [], []
    for name, template in classifiers().items():
        model = clone(template)
        model.fit(X_train, ytr_cls)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cls_rows.append({
            "training_mode": mode, "model": name,
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
            "training_mode": mode, "model": name,
            "r2": r2_score(yte_reg, pred),
            "rmse_log_ratio": float(np.sqrt(np.mean((yte_reg - pred) ** 2))),
            "mae_log_ratio": mean_absolute_error(yte_reg, pred),
        })
    return cls_rows, reg_rows, feature_names


def main():
    OUT.mkdir(exist_ok=True)
    data = load_real()
    _, threshold = make_labels(data, threshold=THRESHOLD)
    groups, group_description = groups_for_cv(data)

    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, te_idx = next(split.split(data, groups=groups))
    train = data.iloc[tr_idx].reset_index(drop=True)
    test = data.iloc[te_idx].reset_index(drop=True)

    # Guard: no earthquake event can occur in both sets.
    train_events = set(train["event_id"].astype(str))
    test_events = set(test["event_id"].astype(str))
    if train_events & test_events:
        raise RuntimeError("Event leakage detected; stopping.")

    # Baseline, then synthetic augmentation sourced only from training events.
    base_cls, base_reg, features = evaluate(train, test, threshold, "real_only")
    syn = make_training_synthetic(train, N_SYNTHETIC, SEED)
    aug_train = pd.concat([train, syn], ignore_index=True)
    aug_cls, aug_reg, _ = evaluate(
        aug_train, test, threshold, "real_plus_training_only_synthetic"
    )

    cls = pd.DataFrame(base_cls + aug_cls)
    reg = pd.DataFrame(base_reg + aug_reg)
    cls.to_csv(OUT / "real_vs_augmented_classification.csv", index=False)
    reg.to_csv(OUT / "real_vs_augmented_regression.csv", index=False)
    syn.to_csv(OUT / "training_only_synthetic_records.csv", index=False)

    # Verify every synthetic source record comes from the real training subset.
    allowed = set(train["rsn"])
    source_ids = set(syn["source_rsn"])
    if not source_ids <= allowed:
        raise RuntimeError("Synthetic source leakage detected.")

    summary = {
        "real_records": int(len(data)),
        "real_training_records": int(len(train)),
        "real_test_records": int(len(test)),
        "training_events": len(train_events),
        "held_out_test_events": len(test_events),
        "training_synthetic_records": int(len(syn)),
        "event_grouping": group_description,
        "threshold": float(threshold),
        "features": features,
        "seed": SEED,
        "synthetic_sources_subset_of_training_RSNs": True,
        "train_and_test_events_disjoint": True,
        "limitations": [
            "Synthetic records are perturbed bootstrap samples, not new earthquakes.",
            "The synthetic target is derived from perturbed RotD spectra.",
            "A single small event-held-out split has substantial uncertainty.",
            "Synthetic augmentation may not improve real-event generalization."
        ],
    }
    (OUT / "real_vs_augmented_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    print("\nCLASSIFICATION COMPARISON")
    print(cls.round(3).to_string(index=False))
    print("\nREGRESSION COMPARISON")
    print(reg.round(3).to_string(index=False))
    print(f"\nSaved results in {OUT.resolve()}")


if __name__ == "__main__":
    main()
