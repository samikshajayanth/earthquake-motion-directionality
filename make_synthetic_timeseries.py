"""Create a SYNTHETIC folder that mimics a PEER time-series download.

Only for checking that the time-series pipeline runs end to end. The data is
random: never report results computed from it.

    python make_synthetic_timeseries.py --out synthetic_timeseries
    python run_pipeline.py --timeseries-dir synthetic_timeseries --out outputs_synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MECH = ["Strike Slip", "Normal", "Reverse", "Reverse Oblique", "Normal Oblique"]


def _quake(n, dt, rng, amp):
    x = rng.normal(size=n)
    f = np.fft.rfftfreq(n, dt)
    x = np.fft.irfft(np.fft.rfft(x) * (f < 15.0), n)
    t = np.arange(n) * dt
    env = (t / 3.0) * np.exp(1 - t / 3.0)
    return amp * x / x.std() * env


def _write_at2(path, acc, dt, label):
    lines = ["PEER NGA STRONG MOTION DATABASE RECORD",
             f"SYNTHETIC EVENT 01/01/2000, SYNTHETIC STATION, {label}",
             "ACCELERATION TIME SERIES IN UNITS OF G",
             f"NPTS= {len(acc):5d}, DT= {dt:.4f} SEC"]
    for i in range(0, len(acc), 5):
        lines.append("".join(f"{v:15.7E}" for v in acc[i:i + 5]))
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_timeseries")
    ap.add_argument("--n-events", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dt, n = 0.01, 2500
    rows, rsn = [], 1
    for eq in range(args.n_events):
        mag = rng.uniform(4.5, 7.5)
        mech = int(rng.choice(5, p=[0.4, 0.1, 0.3, 0.1, 0.1]))
        az1 = float(rng.choice([0, 45, 90, 180, 225]))
        for _ in range(rng.integers(4, 12)):
            rrup = float(np.exp(rng.uniform(np.log(2), np.log(150))))
            vs30 = float(np.exp(rng.normal(np.log(400), 0.4)))
            # more correlated, more unequal components -> stronger directionality
            rho = np.clip(0.15 + 0.4 * (mech == 0) + 0.05 * (mag - 6), 0, 0.9)
            k = rng.uniform(0.5, 1.0)
            a = _quake(n, dt, rng, 0.15)
            b = _quake(n, dt, rng, 0.15)
            h1 = a
            h2 = k * (rho * a + np.sqrt(1 - rho ** 2) * b)
            v = _quake(n, dt, rng, 0.08)
            az2 = (az1 + 90) % 360
            stem = f"RSN{rsn}_SYNTH{eq}_STA{rsn}"
            _write_at2(out / f"{stem}-C{int(az1):03d}.AT2", h1, dt, f"{int(az1)}")
            _write_at2(out / f"{stem}-C{int(az2):03d}.AT2", h2, dt, f"{int(az2)}")
            _write_at2(out / f"{stem}-CUP.AT2", v, dt, "UP")
            for ext in ("DT2", "VT2"):  # real downloads also contain these
                (out / f"{stem}-C{int(az1):03d}.{ext}").write_text("not used\n")
            rows.append({
                " Record Sequence Number": rsn,
                " Earthquake Name": f"Synthetic Event {eq}",
                " Earthquake Magnitude": round(mag, 2),
                " Mechanism": MECH[mech],
                " Rrup (km)": round(rrup, 2),
                " Rjb (km)": round(rrup * 0.9, 2),
                " Vs30 (m/s)": round(vs30, 1),
                " Lowest Usable Freq - Ave. Component (Hz)": round(float(rng.uniform(0.05, 1.0)), 3),
            })
            rsn += 1
    pd.DataFrame(rows).to_csv(out / "_SearchResults.csv", index=False)
    print(f"Wrote {rsn - 1} synthetic records to {out}/")


if __name__ == "__main__":
    main()
