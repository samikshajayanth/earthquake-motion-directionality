"""Turn record metadata into model features."""
import numpy as np
import pandas as pd


def build_features(df):
    """Return (X, feature_names). Missing values are left as NaN and are
    imputed inside the model pipelines, so no information leaks across folds."""
    X = pd.DataFrame(index=df.index)
    X["magnitude"] = df["magnitude"]
    if "rrup" in df:
        X["log_rrup"] = np.log(df["rrup"].clip(lower=0.1))
    if "rjb" in df:
        X["log_rjb"] = np.log(df["rjb"].clip(lower=0.1))
    if "repi" in df:
        X["log_repi"] = np.log(df["repi"].clip(lower=0.1))
    X["log_vs30"] = np.log(df["vs30"].clip(lower=50))
    for col in ("hypo_depth", "dip", "width"):
        if col in df:
            X[col] = df[col]

    # fault mechanism (NGA-West2 codes: 0 strike-slip, 1 normal, 2 reverse,
    # 3 reverse-oblique, 4 normal-oblique); one-hot encoded
    if "mechanism" in df:
        mech = df["mechanism"].round()
        for code in sorted(mech.dropna().unique()):
            X[f"mech_{int(code)}"] = (mech == code).astype(float)
        X.loc[mech.isna(), [c for c in X.columns if c.startswith("mech_")]] = np.nan

    # source-to-site azimuth relative to the fault strike. Directionality in
    # strike-slip events is expected to be tied to this geometry.
    if "azimuth" in df and "strike" in df:
        rel = np.deg2rad(df["azimuth"] - df["strike"])
        X["sin_rel_az"] = np.sin(rel)
        X["cos_rel_az"] = np.cos(rel)

    return X, list(X.columns)


def make_labels(df, threshold=None, quantile=None):
    """Binary label: 1 = strong directionality (ratio >= threshold)."""
    if threshold is None:
        threshold = float(df["ratio"].quantile(quantile if quantile is not None else 0.75))
    return (df["ratio"] >= threshold).astype(int), float(threshold)


def groups_for_cv(df):
    """Group by earthquake so records from one event never sit in both the
    training and test folds (they are strongly correlated)."""
    if df["event_id"].notna().any():
        return df["event_id"].astype(str).values, "earthquake"
    return df["rsn"].values, "record (no event id column found)"
