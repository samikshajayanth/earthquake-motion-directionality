import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    StratifiedKFold, KFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

out = Path("outputs_tuned")
out.mkdir(exist_ok=True)
d = pd.read_csv("outputs_196/rotd_cache.csv")
d["ratio"] = d["RotD100_T1.000"] / d["RotD50_T1.000"]
d["log_ratio"] = np.log(d["ratio"])
d = d.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["ratio", "log_ratio"]
).reset_index(drop=True)
y_cls = (d["ratio"] >= 1.3).astype(int)
y_reg = d["log_ratio"]

raw_cols = [
    c for c in d.columns
    if c.startswith(("RotD50_T", "RotD100_T"))
    and not c.endswith("_T1.000")
]
ratio_cols = []
for p in ["0.050", "0.100", "0.200", "0.300", "0.500",
          "0.750", "1.500", "2.000", "3.000", "4.000",
          "5.000", "7.500", "10.000"]:
    name = f"directionality_ratio_T{p}"
    d[name] = d[f"RotD100_T{p}"] / d[f"RotD50_T{p}"]
    d[name] = d[name].replace([np.inf, -np.inf], np.nan)
    ratio_cols.append(name)

X = d[raw_cols + ratio_cols]

param_grid_cls = {
    "extratreesclassifier__n_estimators": [200, 400, 600],
    "extratreesclassifier__max_depth": [None, 3, 5, 8, 12],
    "extratreesclassifier__min_samples_leaf": [2, 4, 6, 10],
    "extratreesclassifier__max_features": [0.5, 0.7, 1.0],
}
param_grid_reg = {
    "extratreesregressor__n_estimators": [200, 400, 600],
    "extratreesregressor__max_depth": [None, 3, 5, 8, 12],
    "extratreesregressor__min_samples_leaf": [2, 4, 6, 10],
    "extratreesregressor__max_features": [0.5, 0.7, 1.0],
}

def make_model(classification):
    if classification:
        est = ExtraTreesClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    else:
        est = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    return make_pipeline(SimpleImputer(strategy="median"), est)

# Nested CV: hyperparameters are selected only within each outer training fold.
cls_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
reg_outer = KFold(n_splits=5, shuffle=True, random_state=42)
cls_prob = np.zeros(len(d))
reg_pred = np.zeros(len(d))
cls_fold, reg_fold, chosen = [], [], []

for fold, (tr, te) in enumerate(cls_outer.split(X, y_cls), 1):
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=fold)
    search = RandomizedSearchCV(
        make_model(True), param_grid_cls, n_iter=12, scoring="roc_auc",
        cv=inner, random_state=fold, n_jobs=1, refit=True
    )
    search.fit(X.iloc[tr], y_cls.iloc[tr])
    cls_prob[te] = search.predict_proba(X.iloc[te])[:, 1]
    chosen.append({"task": "classification", "fold": fold,
                   "best_inner_score": search.best_score_,
                   "parameters": json.dumps(search.best_params_)})
    cls_fold.append({
        "fold": fold, "roc_auc": roc_auc_score(y_cls.iloc[te], cls_prob[te]),
        "accuracy": accuracy_score(y_cls.iloc[te], cls_prob[te] >= 0.5),
        "f1": f1_score(y_cls.iloc[te], cls_prob[te] >= 0.5, zero_division=0)
    })

for fold, (tr, te) in enumerate(reg_outer.split(X, y_reg), 1):
    inner = KFold(n_splits=4, shuffle=True, random_state=fold)
    search = RandomizedSearchCV(
        make_model(False), param_grid_reg, n_iter=12, scoring="neg_mean_absolute_error",
        cv=inner, random_state=fold, n_jobs=1, refit=True
    )
    search.fit(X.iloc[tr], y_reg.iloc[tr])
    reg_pred[te] = search.predict(X.iloc[te])
    chosen.append({"task": "regression", "fold": fold,
                   "best_inner_score": search.best_score_,
                   "parameters": json.dumps(search.best_params_)})
    reg_fold.append({
        "fold": fold, "r2": r2_score(y_reg.iloc[te], reg_pred[te]),
        "mae_log_ratio": mean_absolute_error(y_reg.iloc[te], reg_pred[te]),
        "rmse_log_ratio": np.sqrt(mean_squared_error(y_reg.iloc[te], reg_pred[te]))
    })

cls_summary = {
    "roc_auc": roc_auc_score(y_cls, cls_prob),
    "accuracy": accuracy_score(y_cls, cls_prob >= 0.5),
    "precision": precision_score(y_cls, cls_prob >= 0.5, zero_division=0),
    "recall": recall_score(y_cls, cls_prob >= 0.5, zero_division=0),
    "f1": f1_score(y_cls, cls_prob >= 0.5, zero_division=0),
}
reg_summary = {
    "r2": r2_score(y_reg, reg_pred),
    "mae_log_ratio": mean_absolute_error(y_reg, reg_pred),
    "rmse_log_ratio": np.sqrt(mean_squared_error(y_reg, reg_pred))
}

pd.DataFrame(cls_fold).to_csv(out / "classification_outer_folds.csv", index=False)
pd.DataFrame(reg_fold).to_csv(out / "regression_outer_folds.csv", index=False)
pd.DataFrame(chosen).to_csv(out / "selected_parameters_by_fold.csv", index=False)
(out / "nested_cv_summary.json").write_text(json.dumps({
    "records": len(d),
    "classification": cls_summary,
    "regression": reg_summary,
    "validation": "5-fold outer CV with 4-fold inner tuning",
    "limitation": "Record-level CV; event-independent generalization is unverified."
}, indent=2))

print("NESTED CV CLASSIFICATION:", json.dumps(cls_summary, indent=2))
print("NESTED CV REGRESSION:", json.dumps(reg_summary, indent=2))
print("Saved to", out)
