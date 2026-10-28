"""Compute RotD50 / RotD100 response spectra from PEER NGA-West2 .AT2 files.

For every record (RSN) the download holds three components. We take the two
horizontal ones (H1, H2), compute the response of a 5%-damped single-degree-of-
freedom oscillator to each, and combine them at every orientation theta:

    u(theta, t) = u1(t) cos(theta) + u2(t) sin(theta)      (linear system)

The peak of |u| at each orientation gives a "response vs orientation" curve.
Following Boore (2010):

    RotD50  = median over orientations of the peak response
    RotD100 = maximum over orientations of the peak response

Directionality ratio = RotD100 / RotD50 (>= 1). We also keep the orientation of
the maximum (measured from H1 toward H2, and as a compass azimuth when the
component azimuths are known).

The oscillator is solved exactly for ground acceleration that is piecewise
linear between samples (first-order-hold discretisation) and run as a digital
filter, which makes it fast.
"""
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.signal import lfilter, ss2tf

DEFAULT_PERIODS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
DAMPING = 0.05
ANGLE_STEP_DEG = 2.0

_NPTS_RE = re.compile(r"NPTS\s*=\s*(\d+)\s*,\s*DT\s*=\s*([0-9.eE+\-]+)", re.IGNORECASE)
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_RSN_RE = re.compile(r"^RSN(\d+)_", re.IGNORECASE)
_VERTICAL_TOKENS = {"UP", "DWN", "DOWN", "DN", "V", "VER", "VERT", "VERTICAL", "Z"}


# ----------------------------------------------------------------- reading
def read_header(path):
    """Return (label, azimuth) from the 2nd header line of an AT2 file.

    The last comma-separated item of that line is the component: an azimuth
    such as '180' or a vertical marker such as 'UP' / 'DWN'.
    """
    with open(path, "r", errors="replace") as f:
        head = [f.readline() for _ in range(4)]
    label = head[1].strip().split(",")[-1].strip().upper() if len(head) > 1 else ""
    az = None
    try:
        az = float(label) % 360
    except ValueError:
        pass
    return label, az


def is_vertical(path, label):
    stem = Path(path).stem.upper()
    last = re.split(r"[-_.]", stem)[-1]
    if label in _VERTICAL_TOKENS or last in _VERTICAL_TOKENS:
        return True
    return last.endswith(("UP", "DWN", "DOWN"))


def read_at2(path):
    """Read an AT2 file. Returns (acceleration in g, dt in seconds)."""
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines[:8]):
        m = _NPTS_RE.search(line)
        if m:
            break
    else:
        raise ValueError(f"no NPTS/DT header in {path}")
    npts, dt = int(m.group(1)), float(m.group(2))
    vals = np.array(_FLOAT_RE.findall("\n".join(lines[i + 1:])), dtype=float)
    if len(vals) < npts:
        npts = len(vals)
    return vals[:npts], dt


# --------------------------------------------------------------- oscillator
def sdof_filter(T, dt, xi=DAMPING):
    """Digital filter (b, a) mapping ground acceleration to relative
    displacement of a SDOF oscillator with period T and damping xi."""
    w = 2 * np.pi / T
    A = np.array([[0.0, 1.0], [-w * w, -2 * xi * w]])
    M = np.zeros((4, 4))
    M[:2, :2] = A
    M[:2, 2] = [0.0, -1.0]  # input matrix B: u'' + 2 xi w u' + w^2 u = -a_g
    M[2, 3] = 1.0 / dt
    E = expm(M * dt)
    Phi, G1, G2 = E[:2, :2], E[:2, 2:3], E[:2, 3:4]
    Bw = Phi @ G2 + G1 - G2
    C = np.array([[1.0, 0.0]])
    D = C @ G2
    num, den = ss2tf(Phi, Bw, C, D)
    return num[0], den


def sdof_displacement(acc, dt, T, xi=DAMPING):
    b, a = sdof_filter(T, dt, xi)
    return lfilter(b, a, acc)


# -------------------------------------------------------------------- RotD
def rotd_spectra(h1, h2, dt, periods=DEFAULT_PERIODS, xi=DAMPING, step=ANGLE_STEP_DEG):
    """Return dict with arrays 'rotd50', 'rotd100', 'theta' (deg from H1 toward
    H2), one value per period. Spectral values are in the units of the input
    acceleration (g)."""
    n = min(len(h1), len(h2))
    pad = int(np.ceil(2 * max(periods) / dt))  # let long-period systems ring down
    a1 = np.concatenate([h1[:n], np.zeros(pad)])
    a2 = np.concatenate([h2[:n], np.zeros(pad)])
    ang = np.deg2rad(np.arange(0.0, 180.0, step))
    cs, sn = np.cos(ang)[:, None], np.sin(ang)[:, None]

    out = {"rotd50": [], "rotd100": [], "theta": []}
    for T in periods:
        u1 = sdof_displacement(a1, dt, T, xi)[None, :]
        u2 = sdof_displacement(a2, dt, T, xi)[None, :]
        peak = np.abs(cs * u1 + sn * u2).max(axis=1)  # peak vs orientation
        w2 = (2 * np.pi / T) ** 2  # displacement -> pseudo-spectral acceleration
        out["rotd50"].append(w2 * np.median(peak))
        out["rotd100"].append(w2 * peak.max())
        out["theta"].append(float(np.rad2deg(ang[peak.argmax()])))
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------- folder driver
def find_records(folder):
    """Map RSN -> list of (path, label, azimuth, is_vertical) for AT2 files."""
    groups = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.upper() != ".AT2":
            continue
        m = _RSN_RE.match(p.name)
        if not m:
            continue
        label, az = read_header(p)
        groups.setdefault(int(m.group(1)), []).append((p, label, az, is_vertical(p, label)))
    return groups


def list_rsns(folder):
    """RSNs that have at least one .AT2 file in the folder (names only, fast)."""
    out = set()
    for p in Path(folder).iterdir():
        if p.suffix.upper() == ".AT2":
            m = _RSN_RE.match(p.name)
            if m:
                out.add(int(m.group(1)))
    return out


def pick_horizontals(files):
    """Return (h1, h2) entries or None if the pair cannot be identified."""
    horiz = [f for f in files if not f[3]]
    if len(horiz) != 2:
        return None
    return horiz[0], horiz[1]


def _absolute_azimuth(az1, az2, theta_rel):
    """Compass azimuth (0-180) of the maximum orientation, when the two
    component azimuths are 90 degrees apart; otherwise NaN."""
    if az1 is None or az2 is None:
        return np.nan
    d = (az2 - az1) % 360
    if abs(d - 90) < 1:
        return (az1 + theta_rel) % 180
    if abs(d - 270) < 1:
        return (az1 - theta_rel) % 180
    return np.nan


def process_record(args):
    rsn, files, periods = args
    try:
        pair = pick_horizontals(files)
        if pair is None:
            return rsn, None, "could not identify two horizontal components"
        (p1, _, az1, _), (p2, _, az2, _) = pair
        h1, dt1 = read_at2(p1)
        h2, dt2 = read_at2(p2)
        if abs(dt1 - dt2) > 1e-9:
            return rsn, None, f"components have different time steps ({dt1} vs {dt2})"
        res = rotd_spectra(h1, h2, dt1, periods)
        row = {"rsn": rsn, "h1_file": p1.name, "h2_file": p2.name, "dt": dt1,
               "npts": min(len(h1), len(h2)), "az_h1": az1, "az_h2": az2}
        for T, a50, a100, th in zip(periods, res["rotd50"], res["rotd100"], res["theta"]):
            row[f"RotD50_T{T:.3f}"] = a50
            row[f"RotD100_T{T:.3f}"] = a100
            row[f"ThetaRel_T{T:.3f}"] = th
            row[f"MaxAzimuth_T{T:.3f}"] = _absolute_azimuth(az1, az2, th)
        return rsn, row, None
    except Exception as exc:  # keep going; report at the end
        return rsn, None, f"{type(exc).__name__}: {exc}"


def compute_folder(folder, periods=DEFAULT_PERIODS, workers=1, limit=None, progress_every=250):
    """Compute RotD spectra for every record in a PEER download folder.

    Returns (DataFrame with one row per record, list of (rsn, reason) failures).
    """
    groups = find_records(folder)
    if not groups:
        raise FileNotFoundError(f"No RSN*_*.AT2 files found in {folder}")
    rsns = sorted(groups)[:limit] if limit else sorted(groups)
    jobs = [(r, groups[r], list(periods)) for r in rsns]
    print(f"Computing RotD50/RotD100 for {len(jobs)} records at {len(periods)} periods "
          f"({workers} worker{'s' if workers != 1 else ''}) ...", flush=True)

    rows, failures = [], []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = ex.map(process_record, jobs, chunksize=8)
            for k, (rsn, row, err) in enumerate(results, 1):
                (rows.append(row) if row else failures.append((rsn, err)))
                if k % progress_every == 0:
                    print(f"  {k}/{len(jobs)} records", flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            rsn, row, err = process_record(job)
            (rows.append(row) if row else failures.append((rsn, err)))
            if k % progress_every == 0:
                print(f"  {k}/{len(jobs)} records", flush=True)
    return pd.DataFrame(rows), failures
