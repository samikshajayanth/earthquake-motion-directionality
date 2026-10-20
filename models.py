"""Models and evaluation for detecting and predicting directionality.

Detection  = classification: does this record show strong directionality?
Prediction = regression: what is log(RotD100/RotD50) for this record?
Both use GroupKFold by earthquake so the test folds contain unseen events.
"""
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def classifiers():
    return {
        "Baseline (majority class)": make_pipeline(SimpleImputer(strategy="median"),
                                                   DummyClassifier(strategy="prior")),
        "Logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                              class_weight="balanced",
                                                              n_jobs=-1, random_state=SEED)),
        "Gradient boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                            max_iter=300, class_weight="balanced",
                                                            random_state=SEED),
    }


def regressors():
    return {
        "Baseline (mean)": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor()),
        "Ridge regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          Ridge(alpha=1.0)),
        "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                             n_jobs=-1, random_state=SEED)),
        "Gradient boosting": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                           max_iter=300, random_state=SEED),
    }


def _folds(n_splits, groups):
    n_splits = max(2, min(n_splits, len(set(groups))))
    return GroupKFold(n_splits=n_splits)


def cross_val_classification(X, y, groups, n_splits=5):
    """Out-of-fold probabilities and metrics for every classifier."""
    results, oof = [], {}
    for name, model in classifiers().items():
        prob = np.zeros(len(y))
        for tr, te in _folds(n_splits, groups).split(X, y, groups):
            m = _clone(model)
            m.fit(X.iloc[tr], y.iloc[tr])
            prob[te] = m.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y, prob) if y.nunique() > 1 else np.nan,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        })
        oof[name] = prob
    return results, oof


def cross_val_regression(X, y, groups, n_splits=5):
    """Out-of-fold predictions and metrics for every regressor."""
    results, oof = [], {}
    for name, model in regressors().items():
        pred = np.zeros(len(y))
        for tr, te in _folds(n_splits, groups).split(X, y, groups):
            m = _clone(model)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = m.predict(X.iloc[te])
        results.append({
            "model": name,
            "r2": r2_score(y, pred),
            "rmse_log_ratio": float(np.sqrt(np.mean((y - pred) ** 2))),
            "mae_log_ratio": mean_absolute_error(y, pred),
        })
        oof[name] = pred
    return results, oof



def feature_importance(X, y, groups, model_name="Random forest"):
    """Permutation importance averaged across held-out earthquake groups.

    Uses MAE rather than R² because individual test folds are small.
    Larger positive importance means shuffling a feature worsens MAE.
    """
    from sklearn.base import clone

    splitter = _folds(5, groups)
    model_template = regressors()[model_name]
    fold_importances = []

    for tr, te in splitter.split(X, y, groups):
        model = clone(model_template)
        model.fit(X.iloc[tr], y.iloc[tr])

        imp = permutation_importance(
            model,
            X.iloc[te],
            y.iloc[te],
            n_repeats=30,
            random_state=SEED,
            scoring="neg_mean_absolute_error"
        )

        fold_importances.append(imp.importances_mean)

    values = np.asarray(fold_importances)
    means = values.mean(axis=0)
    stds = values.std(axis=0)

    out = sorted(
        zip(X.columns, means, stds),
        key=lambda r: r[1],
        reverse=True
    )

    return [
        {
            "feature": str(feature),
            "importance": float(mean),
            "std": float(std)
        }
        for feature, mean, std in out
    ]


def _clone(model):
    from sklearn.base import clone
    return clone(model)
