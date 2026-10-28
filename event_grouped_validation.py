from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, r2_score,
    mean_absolute_error, mean_squared_error
)

CACHE = Path("outputs_196/rotd_cache.csv")
FLATFILE = Path(
    "/Users/samikshajayanth/Desktop/NGA_West2_flatfiles/"
    "NGA_West2_Flatfile_RotD50_d050_public_version.xlsx"
)
OUT = Path("outputs_event_grouped")
OUT.mkdir(exist_ok=True)

cache = pd.read_csv(CACHE)
meta = pd.read_excel(
    FLATFILE,
    sheet_name=0,
    usecols=[
        "Record Sequence Number", "EQID", "Earthquake Name",
        "Earthquake Magnitude", "Vs30 (m/s) selected for analysis",
        "Joyner-Boore Dist. (km)"
    ]
)

cache["RSN"] = pd.to_numeric(
    cache["h1_file"].str.extract(r"RSN(\d+)")[0],
    errors="coerce"
)
meta["Record Sequence Number"] = pd.to_numeric(
    meta["Record Sequence Number"], errors="coerce"
)

df = cache.merge(
    meta,
    left_on="RSN",
    right_on="Record Sequence Number",
    how="left",
    validate="one_to_one"
)

if df["EQID"].isna().any() or len(df) != 196:
    raise ValueError("Metadata matching failed; stopping without evaluating.")

spectral_cols = [
    c for c in df.columns
    if c.startswith(("RotD50_T", "RotD100_T"))
    and c != "RotD50_T1.000"
    and c != "RotD100_T1.000"
]

if not spectral_cols:
    raise ValueError("No spectral predictor columns found.")

for c in spectral_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["ratio"] = (
    pd.to_numeric(df["RotD100_T1.000"], errors="coerce") /
    pd.to_numeric(df["RotD50_T1.000"], errors="coerce")
)
df = df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["ratio", "EQID"]
).reset_index(drop=True)

df["log_ratio"] = np.log(df["ratio"])
df["strong_directionality"] = (df["ratio"] >= 1.3).astype(int)

ratio_cols = []
for c in spectral_cols:
    if c.startswith("RotD100_T"):
        period = c.removeprefix("RotD100_T")
        counterpart = "RotD50_T" + period
        if counterpart in df.columns:
            feature = "ratio_" + period
            df[feature] = df[c] / df[counterpart].replace(0, np.nan)
            ratio_cols.append(feature)

feature_cols = spectral_cols + ratio_cols
X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y_cls = df["strong_directionality"].to_numpy()
y_reg = df["log_ratio"].to_numpy()
groups = df["EQID"].to_numpy()

if len(np.unique(groups)) < 5:
    raise ValueError("Too few unique events for grouped cross-validation.")

cls_splits = list(
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    .split(X, y_cls, groups)
)
reg_splits = list(
    GroupKFold(n_splits=5).split(X, y_reg, groups)
)

def make_classifier():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=10,
            max_features=0.7,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )

def make_regressor():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            max_features=0.7,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
    )

cls_rows = []
cls_true, cls_prob, cls_pred = [], [], []
for fold, (tr, te) in enumerate(cls_splits, 1):
    model = make_classifier()
    model.fit(X.iloc[tr], y_cls[tr])
    prob = model.predict_proba(X.iloc[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_cls[te], prob) if len(np.unique(y_cls[te])) == 2 else np.nan
    cls_rows.append({
        "fold": fold,
        "train_records": len(tr),
        "test_records": len(te),
        "train_events": len(np.unique(groups[tr])),
        "test_events": len(np.unique(groups[te])),
        "roc_auc": auc,
        "accuracy": accuracy_score(y_cls[te], pred),
        "precision": precision_score(y_cls[te], pred, zero_division=0),
        "recall": recall_score(y_cls[te], pred, zero_division=0),
        "f1": f1_score(y_cls[te], pred, zero_division=0)
    })
    cls_true.extend(y_cls[te])
    cls_prob.extend(prob)
    cls_pred.extend(pred)

reg_rows = []
reg_true, reg_pred = [], []
for fold, (tr, te) in enumerate(reg_splits, 1):
    model = make_regressor()
    model.fit(X.iloc[tr], y_reg[tr])
    pred = model.predict(X.iloc[te])
    reg_rows.append({
        "fold": fold,
        "train_records": len(tr),
        "test_records": len(te),
        "train_events": len(np.unique(groups[tr])),
        "test_events": len(np.unique(groups[te])),
        "r2": r2_score(y_reg[te], pred),
        "mae_log_ratio": mean_absolute_error(y_reg[te], pred),
        "rmse_log_ratio": mean_squared_error(y_reg[te], pred) ** 0.5
    })
    reg_true.extend(y_reg[te])
    reg_pred.extend(pred)

cls_df = pd.DataFrame(cls_rows)
reg_df = pd.DataFrame(reg_rows)
cls_df.to_csv(OUT / "classification_event_grouped_folds.csv", index=False)
reg_df.to_csv(OUT / "regression_event_grouped_folds.csv", index=False)

cls_auc = (
    roc_auc_score(cls_true, cls_prob)
    if len(np.unique(cls_true)) == 2 else None
)
summary = {
    "records_evaluated": int(len(df)),
    "distinct_events": int(df["EQID"].nunique()),
    "positive_class_threshold_ratio": 1.3,
    "predictors": "RotD50/RotD100 spectra excluding T=1.0s, plus ratios at other periods",
    "validation": "5-fold event-grouped CV; fixed Extra Trees settings",
    "classification": {
        "roc_auc": cls_auc,
        "accuracy": accuracy_score(cls_true, cls_pred),
        "precision": precision_score(cls_true, cls_pred, zero_division=0),
        "recall": recall_score(cls_true, cls_pred, zero_division=0),
        "f1": f1_score(cls_true, cls_pred, zero_division=0)
    },
    "regression": {
        "r2": r2_score(reg_true, reg_pred),
        "mae_log_ratio": mean_absolute_error(reg_true, reg_pred),
        "rmse_log_ratio": mean_squared_error(reg_true, reg_pred) ** 0.5
    },
    "note": "Event-grouped validation is stricter than record-level validation. These are fixed-model grouped CV results, not nested grouped CV."
}
(OUT / "event_grouped_summary.json").write_text(json.dumps(summary, indent=2))

print("Records:", len(df))
print("Distinct EQIDs:", df["EQID"].nunique())
print("\nEVENT-GROUPED CLASSIFICATION")
print(json.dumps(summary["classification"], indent=2))
print("\nEVENT-GROUPED REGRESSION")
print(json.dumps(summary["regression"], indent=2))
print("\nCLASSIFICATION FOLDS")
print(cls_df.to_string(index=False))
print("\nREGRESSION FOLDS")
print(reg_df.to_string(index=False))
print("\nSaved results in", OUT)
