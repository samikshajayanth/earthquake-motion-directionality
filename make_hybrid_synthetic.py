"""Hybrid statistical augmentation for the earthquake directionality project.

Synthetic records are generated from the real RotD cache by bootstrap
resampling and controlled perturbations. They are NOT real earthquakes,
ground-motion simulations, or independent observations.

Usage:
    python make_hybrid_synthetic.py --n 5000 --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = [
    "0.050", "0.100", "0.200", "0.300", "0.500", "0.750",
    "1.000", "1.500", "2.000", "3.000", "4.000", "5.000",
    "7.500", "10.000",
]
SOURCE = Path("outputs_196/rotd_cache.csv")
OUT = Path("synthetic_hybrid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be positive")
    if not SOURCE.exists():
        raise SystemExit(f"Real reference cache not found: {SOURCE}")

    rng = np.random.default_rng(args.seed)
    real = pd.read_csv(SOURCE)

    if real["rsn"].duplicated().any():
        raise SystemExit("Duplicate RSNs in reference cache; stopping safely.")

    # Sample whole rows so each synthetic record begins with a real record's
    # internally consistent response spectrum and orientation measurements.
    idx = rng.integers(0, len(real), size=args.n)
    synth = real.iloc[idx].reset_index(drop=True).copy()

    # Replace identifiers; these IDs must never be interpreted as real RSNs.
    synth["rsn"] = [f"SYN_{i:06d}" for i in range(1, args.n + 1)]
    synth["h1_file"] = "SYNTHETIC"
    synth["h2_file"] = "SYNTHETIC"
    synth["data_source"] = "synthetic"
    synth["source_rsn"] = real.iloc[idx]["rsn"].to_numpy()
    synth["synthetic_seed"] = args.seed

    # Add modest, multiplicative noise to positive response-spectrum values.
    # Apply a common perturbation to each RotD50/RotD100 pair, then ensure
    # RotD100 remains >= RotD50.
    for period in PERIODS:
        c50 = f"RotD50_T{period}"
        c100 = f"RotD100_T{period}"
        if c50 not in synth or c100 not in synth:
            continue

        factor = np.exp(rng.normal(0.0, 0.06, args.n))
        synth[c50] = np.maximum(synth[c50].to_numpy() * factor, 1e-10)

        ratio = (
            real.iloc[idx][c100].to_numpy()
            / np.maximum(real.iloc[idx][c50].to_numpy(), 1e-10)
        )
        # Slightly perturb the observed ratio while respecting RotD100 >= RotD50.
        ratio_factor = np.exp(rng.normal(0.0, 0.025, args.n))
        ratio = np.maximum(ratio * ratio_factor, 1.0)
        synth[c100] = synth[c50].to_numpy() * ratio

        theta = f"ThetaRel_T{period}"
        if theta in synth:
            synth[theta] = (
                synth[theta].to_numpy() + rng.normal(0.0, 4.0, args.n)
            ) % 180.0

        max_az = f"MaxAzimuth_T{period}"
        if max_az in synth:
            synth[max_az] = (
                synth[max_az].to_numpy() + rng.normal(0.0, 4.0, args.n)
            ) % 180.0

    # Perturb sampling interval and sample count only mildly; these are
    # placeholders inherited from source records, not generated waveforms.
    if "dt" in synth:
        synth["dt"] = np.maximum(
            synth["dt"].to_numpy() * np.exp(rng.normal(0, 0.015, args.n)),
            1e-6,
        )
    if "npts" in synth:
        synth["npts"] = np.maximum(
            1, np.rint(synth["npts"].to_numpy()).astype(int)
        )

    # Recalculate ratio and binary target at the project’s 1-second period.
    c50, c100 = "RotD50_T1.000", "RotD100_T1.000"
    if c50 not in synth or c100 not in synth:
        raise SystemExit("Required 1.0-second RotD columns are missing.")

    synth["directionality_ratio_T1.000"] = synth[c100] / synth[c50]

    # Match the existing project threshold from outputs_196/summary.json.
    summary_path = Path("outputs_196/summary.json")
    threshold = 1.3367660281644023
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        threshold = float(
            summary.get("strong_directionality_threshold", threshold)
        )

    synth["strong_directionality_T1.000"] = (
        synth["directionality_ratio_T1.000"] >= threshold
    ).astype(int)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "synthetic_hybrid_5000.csv"
    if args.n != 5000:
        csv_path = out / f"synthetic_hybrid_{args.n}.csv"
    synth.to_csv(csv_path, index=False)

    # Validation checks.
    checks = {
        "requested_records": args.n,
        "generated_records": int(len(synth)),
        "unique_synthetic_ids": int(synth["rsn"].nunique()),
        "all_ids_marked_synthetic": bool(
            synth["rsn"].astype(str).str.startswith("SYN_").all()
        ),
        "all_data_source_synthetic": bool(
            (synth["data_source"] == "synthetic").all()
        ),
        "RotD100_ge_RotD50_at_1s": bool(
            (synth[c100] >= synth[c50]).all()
        ),
        "positive_RotD50_at_1s": bool((synth[c50] > 0).all()),
        "directionality_ratio_matches_spectra": bool(
            np.allclose(synth["directionality_ratio_T1.000"],
                        synth[c100] / synth[c50])
        ),
        "threshold_used": threshold,
        "positive_class_fraction": float(
            synth["strong_directionality_T1.000"].mean()
        ),
        "seed": args.seed,
        "reference_file": str(SOURCE),
        "limitations": [
            "Bootstrap-derived statistical augmentation, not physical simulation.",
            "source_rsn identifies the real row used as the starting point.",
            "dt and npts are inherited metadata; no synthetic waveform is generated.",
            "Do not evaluate final model performance on synthetic test records.",
        ],
    }

    report_path = out / "validation_report.json"
    report_path.write_text(json.dumps(checks, indent=2))

    print(f"Reference records: {len(real)}")
    print(f"Synthetic records written: {len(synth)}")
    print(f"CSV: {csv_path}")
    print(f"Validation report: {report_path}")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
