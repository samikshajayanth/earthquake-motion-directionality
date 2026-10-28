import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)

src = Path("outputs_196/rotd_cache.csv")
out = Path("outputs_196_analysis")
out.mkdir(exist_ok=True)

df = pd.read_csv(src)
df["ratio"] = df["RotD100_T1.000"] / df["RotD50_T1.000"]
df["log_ratio"] = np.log(df["ratio"])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["ratio", "log_ratio"]).reset_index(drop=True)
df["strong_directionality"] = (df["ratio"] >= 1.3).astype(int)

periods = ["0.050", "0.100", "0.200", "0.300", "0.500",
           "0.750", "1.000", "1.500", "2.000", "3.000",
           "4.000", "5.000", "7.500", "10.000"]

# Use RotD50 and RotD100 values at periods other than 1 second.
# Exclude target-defining 1-second values to prevent direct leakage.
features = [
    c for c in df.columns
    if c.startswith(("RotD50_T", "RotD100_T"))
    and not c.endswith("_T1.000")
]
X = df[features]
y_cls = df["strong_directionality"]
y_reg = df["log_ratio"]

# RSN is record identity, not an earthquake/event identifier.
# No event metadata is present in this cache, so use shuffled stratified
# folds at record level and explicitly document this limitation.
from sklearn.model_selection import StratifiedKFold, KFold
n_splits = 5
cls_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
reg_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cls_models = {
    "Majority baseline": make_pipeline(SimpleImputer(strategy="median"),
                                       DummyClassifier(strategy="prior")),
    "Logistic regression": make_pipeline(SimpleImputer(strategy="median"),
                                         StandardScaler(),
                                         LogisticRegression(max_iter=2000,
                                                            class_weight="balanced")),
    "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=300,
                                                          min_samples_leaf=5,
                                                          class_weight="balanced",
                                                          random_state=42,
                                                          n_jobs=-1)),
}
reg_models = {
    "Mean baseline": make_pipeline(SimpleImputer(strategy="median"),
                                   DummyRegressor()),
    "Ridge regression": make_pipeline(SimpleImputer(strategy="median"),
                                      StandardScaler(), Ridge(alpha=1.0)),
    "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestRegressor(n_estimators=300,
                                                         min_samples_leaf=5,
                                                         random_state=42,
                                                         n_jobs=-1)),
}

cls_rows = []
for name, model in cls_models.items():
    if name == "Majority baseline":
        prob = cross_val_predict(model, X, y_cls, cv=cls_cv, method="predict_proba")[:, 1]
    else:
        prob = cross_val_predict(model, X, y_cls, cv=cls_cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    cls_rows.append({
        "model": name,
        "roc_auc": roc_auc_score(y_cls, prob),
        "accuracy": accuracy_score(y_cls, pred),
        "precision": precision_score(y_cls, pred, zero_division=0),
        "recall": recall_score(y_cls, pred, zero_division=0),
        "f1": f1_score(y_cls, pred, zero_division=0),
    })

reg_rows = []
for name, model in reg_models.items():
    pred = cross_val_predict(model, X, y_reg, cv=reg_cv)
    reg_rows.append({
        "model": name,
        "r2": r2_score(y_reg, pred),
        "rmse_log_ratio": float(np.sqrt(mean_squared_error(y_reg, pred))),
        "mae_log_ratio": mean_absolute_error(y_reg, pred),
    })

cls_df = pd.DataFrame(cls_rows).round(4)
reg_df = pd.DataFrame(reg_rows).round(4)
cls_df.to_csv(out / "cache_classification_metrics.csv", index=False)
reg_df.to_csv(out / "cache_regression_metrics.csv", index=False)
df.to_csv(out / "directionality_labels.csv", index=False)

summary = {
    "records": int(len(df)),
    "features": features,
    "period_seconds_for_target": 1.0,
    "threshold": 1.3,
    "strong_count": int(y_cls.sum()),
    "below_threshold_count": int((y_cls == 0).sum()),
    "validation": "5-fold shuffled record-level cross-validation",
    "limitation": (
        "Cache lacks event identifiers and metadata needed for event-grouped CV. "
        "Metrics are preliminary and may be optimistic if records share earthquakes. "
        "Spectral features from periods other than 1.0 s predict the 1.0 s ratio."
    ),
}
(out / "cache_evaluation_summary.json").write_text(json.dumps(summary, indent=2))

print("\nCLASSIFICATION RESULTS")
print(cls_df.to_string(index=False))
print("\nREGRESSION RESULTS")
print(reg_df.to_string(index=False))
print("\nSaved results to", out)
print("WARNING: record-level CV is preliminary; it is not event-independent validation.")
