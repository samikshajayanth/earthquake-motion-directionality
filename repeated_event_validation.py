import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error

out = Path("outputs_repeated_event_cv")
out.mkdir(exist_ok=True)

cache = pd.read_csv("outputs_196/rotd_cache.csv")
flatfile = "/Users/samikshajayanth/Desktop/NGA_West2_flatfiles/NGA_West2_Flatfile_RotD50_d050_public_version.xlsx"
meta = pd.read_excel(flatfile, usecols=["Record Sequence Number", "EQID"])
cache["RSN"] = pd.to_numeric(cache["h1_file"].str.extract(r"RSN(\d+)")[0])
meta["Record Sequence Number"] = pd.to_numeric(meta["Record Sequence Number"])
df = cache.merge(meta, left_on="RSN", right_on="Record Sequence Number", validate="one_to_one")
if df["EQID"].isna().any():
    raise ValueError("Missing EQID matches; stopping.")

spectral = [c for c in df if c.startswith(("RotD50_T", "RotD100_T")) and c not in ["RotD50_T1.000", "RotD100_T1.000"]]
for c in spectral:
    df[c] = pd.to_numeric(df[c], errors="coerce")
ratio_cols = []
for c in spectral:
    if c.startswith("RotD100_T"):
        other = "RotD50_T" + c.removeprefix("RotD100_T")
        if other in df:
            name = "ratio_" + c.removeprefix("RotD100_T")
            df[name] = df[c] / df[other].replace(0, np.nan)
            ratio_cols.append(name)

df["ratio"] = pd.to_numeric(df["RotD100_T1.000"], errors="coerce") / pd.to_numeric(df["RotD50_T1.000"], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio", "EQID"]).reset_index(drop=True)
X = df[spectral + ratio_cols].replace([np.inf, -np.inf], np.nan)
y = (df["ratio"].to_numpy() >= 1.3).astype(int)
yr = np.log(df["ratio"].to_numpy())
groups = df["EQID"].to_numpy()

def classifier():
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, max_features=0.7, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1))

def regressor():
    return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=400, min_samples_leaf=4, max_features=0.7, max_depth=8, random_state=42, n_jobs=-1))

repeat_rows, fold_rows = [], []
for seed in [42, 43, 44, 45, 46]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cls_true, cls_prob, cls_pred = [], [], []
    reg_true, reg_pred = [], []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups), 1):
        cm = classifier()
        cm.fit(X.iloc[tr], y[tr])
        prob = cm.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        cls_true.extend(y[te])
        cls_prob.extend(prob)
        cls_pred.extend(pred)

        rm = regressor()
        rm.fit(X.iloc[tr], yr[tr])
        rp = rm.predict(X.iloc[te])
        reg_true.extend(yr[te])
        reg_pred.extend(rp)

        fold_rows.append({
            "seed": seed, "fold": fold,
            "train_events": len(np.unique(groups[tr])),
            "test_events": len(np.unique(groups[te])),
            "train_records": len(tr), "test_records": len(te),
            "classification_accuracy": accuracy_score(y[te], pred),
            "classification_f1": f1_score(y[te], pred, zero_division=0),
            "regression_r2": r2_score(yr[te], rp),
            "regression_mae": mean_absolute_error(yr[te], rp)
        })
    repeat_rows.append({
        "seed": seed,
        "classification_roc_auc": roc_auc_score(cls_true, cls_prob),
        "classification_accuracy": accuracy_score(cls_true, cls_pred),
        "classification_precision": precision_score(cls_true, cls_pred, zero_division=0),
        "classification_recall": recall_score(cls_true, cls_pred, zero_division=0),
        "classification_f1": f1_score(cls_true, cls_pred, zero_division=0),
        "regression_r2": r2_score(reg_true, reg_pred),
        "regression_mae_log_ratio": mean_absolute_error(reg_true, reg_pred),
        "regression_rmse_log_ratio": mean_squared_error(reg_true, reg_pred) ** 0.5
    })

repeats = pd.DataFrame(repeat_rows)
folds = pd.DataFrame(fold_rows)
repeats.to_csv(out / "repeat_summary.csv", index=False)
folds.to_csv(out / "fold_details.csv", index=False)
metrics = [c for c in repeats.columns if c != "seed"]
summary = {c: {"mean": float(repeats[c].mean()), "std": float(repeats[c].std(ddof=1))} for c in metrics}
summary["records"] = int(len(df))
summary["events"] = int(df["EQID"].nunique())
summary["repeats"] = 5
summary["validation"] = "Five repetitions of five-fold stratified event-grouped CV; fixed Extra Trees settings"
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("REPEATED EVENT-GROUPED CV RESULTS")
print(repeats.to_string(index=False))
print("\nMEAN AND STANDARD DEVIATION")
print(json.dumps(summary, indent=2))
print("\nSaved in", out)
