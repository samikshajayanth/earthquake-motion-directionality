import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

OUT = Path("outputs_improved")
OUT.mkdir(exist_ok=True)

d = pd.read_csv("outputs_196/rotd_cache.csv")
periods = ["0.050", "0.100", "0.200", "0.300", "0.500",
           "0.750", "1.000", "1.500", "2.000", "3.000",
           "4.000", "5.000", "7.500", "10.000"]

d["ratio"] = d["RotD100_T1.000"] / d["RotD50_T1.000"]
d["log_ratio"] = np.log(d["ratio"])
d = d.replace([np.inf, -np.inf], np.nan)
d = d.dropna(subset=["ratio", "log_ratio"]).reset_index(drop=True)
y = d["strong_directionality"] = (d["ratio"] >= 1.3).astype(int)
yr = d["log_ratio"]

# Feature set A: raw spectra, excluding the target period.
raw_cols = [c for c in d.columns
            if c.startswith(("RotD50_T", "RotD100_T"))
            and not c.endswith("_T1.000")]
X_raw = d[raw_cols].copy()

# Feature set B: directionality ratios at other periods.
ratio_cols = []
for p in periods:
    if p == "1.000":
        continue
    name = f"directionality_ratio_T{p}"
    d[name] = d[f"RotD100_T{p}"] / d[f"RotD50_T{p}"]
    d[name] = d[name].replace([np.inf, -np.inf], np.nan)
    ratio_cols.append(name)
X_ratio = d[ratio_cols].copy()

# Feature set C: ratios plus raw spectra at other periods.
X_combined = pd.concat([X_raw, X_ratio], axis=1)

classifiers = {
    "Logistic regression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
    ),
    "Random forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                               max_features=0.7, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    ),
    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4,
                             max_features=0.7, class_weight="balanced",
                             random_state=42, n_jobs=-1)
    ),
    "RBF SVM": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        SVC(C=0.5, kernel="rbf", class_weight="balanced", probability=True,
            random_state=42)
    ),
}
regressors = {
    "Ridge": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)
    ),
    "Random forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=400, min_samples_leaf=4,
                              max_features=0.7, random_state=42, n_jobs=-1)
    ),
    "Extra Trees": make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=400, min_samples_leaf=4,
                            max_features=0.7, random_state=42, n_jobs=-1)
    ),
}

cls_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reg_cv = KFold(n_splits=5, shuffle=True, random_state=42)
cls_rows, reg_rows = [], []

for feature_name, X in [
    ("Raw spectra", X_raw),
    ("Other-period ratios", X_ratio),
    ("Combined", X_combined),
]:
    for model_name, model in classifiers.items():
        prob = cross_val_predict(model, X, y, cv=cls_cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        cls_rows.append({
            "features": feature_name, "model": model_name,
            "roc_auc": roc_auc_score(y, prob),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })

    for model_name, model in regressors.items():
        pred = cross_val_predict(model, X, yr, cv=reg_cv)
        reg_rows.append({
            "features": feature_name, "model": model_name,
            "r2": r2_score(yr, pred),
            "rmse_log_ratio": np.sqrt(mean_squared_error(yr, pred)),
            "mae_log_ratio": mean_absolute_error(yr, pred),
        })

cls = pd.DataFrame(cls_rows).sort_values("roc_auc", ascending=False)
reg = pd.DataFrame(reg_rows).sort_values("mae_log_ratio")
cls.to_csv(OUT / "classification_comparison.csv", index=False)
reg.to_csv(OUT / "regression_comparison.csv", index=False)

print("\nCLASSIFICATION — ranked by ROC-AUC")
print(cls.round(4).to_string(index=False))
print("\nREGRESSION — ranked by MAE")
print(reg.round(4).to_string(index=False))
print("\nSaved to", OUT)
print("Validation is shuffled record-level CV; event-independent performance is not established.")
