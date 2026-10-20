"""Detect and predict earthquake ground motion directionality (NGA-West2).

Main route: a time-series download from the PEER online tool (a folder of .AT2
files plus _SearchResults.csv). RotD50/RotD100 are computed from the two
horizontal components of every record:

    python run_pipeline.py --timeseries-dir path/to/UnscaledAccRecord9417 \
                           --workers 4 --period 1.0 --out outputs

    (add --limit 30 first for a quick trial run)

Alternative: flatfiles that already contain both RotD50 and RotD100 spectra:

    python run_pipeline.py --rotd50 RotD50.xlsx --rotd100 RotD100.xlsx --out outputs

Quick check on synthetic stand-in data (results are meaningless):

    python make_synthetic_timeseries.py --out synthetic_timeseries
    python run_pipeline.py --timeseries-dir synthetic_timeseries --out outputs_synthetic
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import models
import plots
from data_io import build_dataset, build_dataset_from_timeseries
from features import build_features, groups_for_cv, make_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_argument_group("data source (choose one)")
    src.add_argument("--timeseries-dir", default=None,
                     help="folder from the PEER online tool (AT2 files + _SearchResults.csv); "
                          "RotD50/RotD100 are computed from the time series")
    src.add_argument("--rotd50", default=None, help="flatfile with RotD50 spectra + metadata (.xlsx/.csv)")
    src.add_argument("--rotd100", default=None, help="flatfile with RotD100 spectra (.xlsx/.csv)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes for the time-series computation (default 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="time-series mode: only process the first N records (quick trial)")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached RotD results")
    ap.add_argument("--no-usable-filter", action="store_true",
                    help="do not drop long periods beyond a record's lowest usable frequency")
    ap.add_argument("--period", type=float, default=1.0, help="spectral period in seconds (default 1.0)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ratio at or above which directionality counts as strong")
    ap.add_argument("--quantile", type=float, default=None,
                    help="alternative to --threshold: use this quantile of the ratio (default 0.75)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    ap.add_argument("--sheet50", default=None, help="Excel sheet name/index for the RotD50 file")
    ap.add_argument("--sheet100", default=None, help="Excel sheet name/index for the RotD100 file")
    ap.add_argument("--colmap", default=None, help="JSON file mapping canonical names to column names")
    ap.add_argument("--out", default="outputs", help="output folder")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    if args.timeseries_dir:
        df, period_stats, info = build_dataset_from_timeseries(
            args.timeseries_dir, args.period, workers=args.workers,
            cache=out / "rotd_cache.csv", recompute=args.recompute, limit=args.limit,
            colmap_path=args.colmap, usable_filter=not args.no_usable_filter)
    elif args.rotd50 and args.rotd100:
        df, period_stats, info = build_dataset(args.rotd50, args.rotd100, args.period,
                                               args.sheet50, args.sheet100, args.colmap)
    else:
        ap.error("give either --timeseries-dir, or both --rotd50 and --rotd100")
    print(json.dumps(info, indent=2))

    X, feature_names = build_features(df)
    y_cls, threshold = make_labels(df, args.threshold, args.quantile)
    y_reg = df["log_ratio"]
    groups, group_kind = groups_for_cv(df)
    info.update({
        "features": feature_names,
        "ratio_median": float(df["ratio"].median()),
        "ratio_mean": float(df["ratio"].mean()),
        "strong_directionality_threshold": threshold,
        "strong_fraction": float(y_cls.mean()),
        "cv_grouped_by": group_kind,
        "folds": args.folds,
    })
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
    print(f"Strong-directionality threshold: ratio >= {threshold:.3f} "
          f"({y_cls.mean():.1%} of records); CV grouped by {group_kind}")

    print("\nDetection (classification) ...")
    cls_results, cls_oof = models.cross_val_classification(X, y_cls, groups, args.folds)
    cls_df = pd.DataFrame(cls_results).round(3)
    print(cls_df.to_string(index=False))

    print("\nPrediction (regression on log ratio) ...")
    reg_results, reg_oof = models.cross_val_regression(X, y_reg, groups, args.folds)
    reg_df = pd.DataFrame(reg_results).round(3)
    print(reg_df.to_string(index=False))

    best_reg = reg_df[~reg_df["model"].str.startswith("Baseline")].sort_values("r2").iloc[-1]
    importance = models.feature_importance(X, y_reg, groups)

    cls_df.to_csv(out / "results_classification.csv", index=False)
    reg_df.to_csv(out / "results_regression.csv", index=False)
    pd.DataFrame(importance).to_csv(out / "feature_importance.csv", index=False)
    period_stats.to_csv(out / "ratio_by_period.csv", index=False)
    (out / "summary.json").write_text(json.dumps(info, indent=2, default=str))

    t = info["period_used"]
    plots.plot_ratio_vs_period(period_stats, t, out / "fig_ratio_vs_period.png")
    plots.plot_ratio_hist(df["ratio"], threshold, t, out / "fig_ratio_hist.png")
    plots.plot_roc(y_cls, cls_oof, out / "fig_roc.png")
    plots.plot_pred_vs_actual(y_reg.values, reg_oof[best_reg["model"]], best_reg["model"],
                              best_reg["r2"], out / "fig_pred_vs_actual.png")
    plots.plot_importance(importance, out / "fig_feature_importance.png")

    print(f"\nTop features: {[r['feature'] for r in importance[:5]]}")
    print(f"Saved tables, summary.json and figures to {out}/")


if __name__ == "__main__":
    main()
