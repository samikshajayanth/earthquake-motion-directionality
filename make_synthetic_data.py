"""Create a SYNTHETIC stand-in for the NGA-West2 flatfiles.

This exists only so you can check that the pipeline runs end to end before the
real data is downloaded. The numbers it produces mean nothing scientifically:
never report results from this data in the write-up or the demo.

    python make_synthetic_data.py --out synthetic_data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    rsn = 1
    for eq in range(args.n_events):
        mag = rng.uniform(4.0, 7.6)
        mech = int(rng.choice([0, 1, 2, 3, 4], p=[0.4, 0.1, 0.3, 0.1, 0.1]))
        strike = rng.uniform(0, 360)
        depth = rng.uniform(3, 20)
        for _ in range(rng.integers(5, 40)):
            rrup = float(np.exp(rng.uniform(np.log(1), np.log(200))))
            vs30 = float(np.exp(rng.normal(np.log(400), 0.4)))
            az = rng.uniform(0, 360)
            rows.append(dict(rsn=rsn, eq=eq, mag=mag, mech=mech, strike=strike,
                             depth=depth, rrup=rrup, vs30=vs30, az=az))
            rsn += 1
    d = pd.DataFrame(rows)

    # fake directionality: stronger for strike-slip, larger magnitude, mid distance
    base = (0.12 + 0.05 * (d["mech"] == 0) + 0.02 * (d["mag"] - 6)
            + 0.04 * np.exp(-((np.log(d["rrup"]) - np.log(30)) ** 2) / 2))
    log_ratio = np.clip(base + rng.normal(0, 0.08, len(d)), 0.0, 0.8)

    meta = pd.DataFrame({
        "Record Sequence Number": d["rsn"],
        "EQID": d["eq"],
        "Earthquake Magnitude": d["mag"].round(2),
        "Mechanism Based on Rake Angle": d["mech"],
        "ClstD (km)": d["rrup"].round(2),
        "Joyner-Boore Dist. (km)": (d["rrup"] * 0.9).round(2),
        "Vs30 (m/s) selected for analysis": d["vs30"].round(1),
        "Hypocenter Depth (km)": d["depth"].round(1),
        "Strike (deg)": d["strike"].round(1),
        "Source to site azimuth": d["az"].round(1),
    })
    # a few -999 values as the real flatfile uses for missing data
    meta.loc[rng.choice(len(meta), size=len(meta) // 20, replace=False),
             "Hypocenter Depth (km)"] = -999

    spec50, spec100 = {}, {}
    for T in PERIODS:
        med = np.exp(-4.5 + 0.9 * (d["mag"] - 6) - 1.1 * np.log(d["rrup"] + 10) - 0.5 * np.log(T + 0.1))
        s50 = med * np.exp(rng.normal(0, 0.5, len(d)))
        spec50[f"T{T:.3f}S"] = s50
        spec100[f"T{T:.3f}S"] = s50 * np.exp(log_ratio + rng.normal(0, 0.02, len(d)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, pd.DataFrame(spec50)], axis=1).to_csv(out / "synthetic_rotd50.csv", index=False)
    pd.concat([meta[["Record Sequence Number"]], pd.DataFrame(spec100)], axis=1).to_csv(
        out / "synthetic_rotd100.csv", index=False)
    print(f"Wrote synthetic files to {out}/ ({len(d)} records, {args.n_events} events)")


if __name__ == "__main__":
    main()
