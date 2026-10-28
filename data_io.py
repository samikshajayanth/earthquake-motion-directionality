"""Load the PEER NGA-West2 flatfiles and build the directionality dataset.

Directionality is measured here by the ratio

    ratio(T) = RotD100(T) / RotD50(T)

where RotD50 is the median spectral acceleration over all horizontal
orientations and RotD100 is the maximum. A ratio of 1.0 means no
directionality; larger values mean the ground motion is stronger in some
horizontal orientations than in others.

The Updated NGA-West2 flatfile download is split in two zips. One holds the
RotD50 spectra (with the record metadata), the other the RotD100 spectra.
Both are keyed by "Record Sequence Number" (RSN).

IMPORTANT: column names differ slightly between flatfile versions. The
loader matches columns by a list of known aliases and prints what it found.
If something is not matched, pass --colmap with a JSON file such as
{"magnitude": "Earthquake Magnitude", "rrup": "ClstD (km)"}.
"""
import csv
import json
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# canonical name -> accepted column names (compared after normalising)
ALIASES = {
    "rsn": ["record sequence number", "record sequence #", "record seq. #", "record seq #",
            "record sequence no.", "rsn"],
    "event_id": ["earthquake id", "eqid", "earthquake name", "event id"],
    "magnitude": ["earthquake magnitude", "magnitude", "mw"],
    "rrup": ["closest d (km)", "clstd (km)", "clstd", "rrup (km)", "rrup"],
    "rjb": ["joyner-boore dist. (km)", "joyner-boore dist (km)", "rjb (km)", "rjb"],
    "repi": ["epicentral distance (km)", "epicentral dist. (km)", "repi (km)", "repi"],
    "vs30": ["vs30 (m/s) selected for analysis", "vs30 (m/s)", "vs30 (m/sec)", "vs30"],
    "mechanism": ["mechanism based on rake angle", "fault type", "mechanism"],
    "hypo_depth": ["hypocenter depth (km)", "hypocenter depth", "hypo depth (km)"],
    "dip": ["dip (deg)", "dip"],
    "strike": ["strike (deg)", "strike"],
    "width": ["fault rupture width (km)", "rupture width (km)", "width (km)"],
    "azimuth": ["source to site azimuth", "epicenter-station azimuth", "site azimuth"],
}
REQUIRED = ["rsn", "magnitude", "vs30"]  # plus one of rrup / rjb
SPECTRAL_RE = re.compile(r"^T(\d+(?:\.\d+)?)S$", re.IGNORECASE)


def _norm(name):
    return re.sub(r"\s+", " ", str(name).strip().lower())


def read_table(path, sheet=None):
    """Read a csv or Excel flatfile."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    return pd.read_csv(path, low_memory=False)


def match_columns(columns, overrides=None):
    """Map canonical names to the real column names in a flatfile."""
    overrides = overrides or {}
    by_norm = {_norm(c): c for c in columns}
    found = {}
    for canon, aliases in ALIASES.items():
        if canon in overrides:
            if overrides[canon] not in columns:
                raise KeyError(f"colmap says {canon} -> {overrides[canon]!r}, not in file")
            found[canon] = overrides[canon]
            continue
        for alias in aliases:
            if alias in by_norm:
                found[canon] = by_norm[alias]
                break

    # High-pass / lowest usable frequency column (PEER uses both spellings).
    if "hp_freq" not in found and "hp_freq" not in overrides:
        for norm, col in by_norm.items():
            if "lowest usable freq" in norm or "lowest useable freq" in norm:
                found["hp_freq"] = col
                break
    return found


def spectral_columns(columns):
    """Return {period_in_seconds: column_name} for columns like T1.000S."""
    out = {}
    for c in columns:
        m = SPECTRAL_RE.match(str(c).strip())
        if m:
            out[float(m.group(1))] = c
    return dict(sorted(out.items()))


def _clean_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s <= -900)  # flatfiles use -999 for missing


_MECH_TEXT = [("reverse oblique", 3), ("reverse-oblique", 3), ("normal oblique", 4),
              ("normal-oblique", 4), ("strike", 0), ("reverse", 2), ("thrust", 2), ("normal", 1)]


def _encode_mechanism(series):
    """Fault mechanism as NGA-West2 codes (0 SS, 1 N, 2 R, 3 RO, 4 NO).
    Accepts numeric codes or text such as 'Strike Slip' / 'Reverse Oblique'."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5:
        return num.mask(num <= -900)

    def code(x):
        s = str(x).strip().lower().replace("_", " ")
        for key, c in _MECH_TEXT:
            if key in s:
                return c
        return np.nan

    return series.map(code).astype(float)


def build_dataset(rotd50_path, rotd100_path, period=1.0, sheet50=None,
                  sheet100=None, colmap_path=None):
    """Return (dataset, period_stats, info) from two flatfile-style tables.

    dataset: one row per record, with metadata, 'ratio' and 'log_ratio'.
    period_stats: median and 16/84th percentile of the ratio at every period.
    info: dict with the column mapping and cleaning counts (for the write-up).
    """
    f50 = read_table(rotd50_path, sheet50)
    f100 = read_table(rotd100_path, sheet100)
    return assemble_dataset(f50, f100, period, colmap_path,
                            sources=(str(rotd50_path), str(rotd100_path)))


def assemble_dataset(f50, f100, period=1.0, colmap_path=None, sources=("", ""),
                     usable_filter=False):
    """Core of the loader. f50 holds metadata + RotD50 spectra (columns like
    T1.000S); f100 holds the RotD100 spectra and the RSN."""
    overrides = json.loads(Path(colmap_path).read_text()) if colmap_path else {}

    f50 = f50.copy()
    f100 = f100.copy()
    f50.columns = [str(c).strip() for c in f50.columns]
    f100.columns = [str(c).strip() for c in f100.columns]

    cm50 = match_columns(f50.columns, overrides)
    cm100 = match_columns(f100.columns, overrides)
    missing = [k for k in REQUIRED if k not in cm50]
    if "rrup" not in cm50 and "rjb" not in cm50:
        missing.append("rrup or rjb")
    if "rsn" not in cm100:
        missing.append("rsn (in RotD100 file)")
    if missing:
        raise KeyError(
            f"Could not find columns for: {missing}. "
            f"Columns in the RotD50 file start with: {list(f50.columns)[:40]}. "
            "Use --colmap to map them."
        )

    spec50 = spectral_columns(f50.columns)
    spec100 = spectral_columns(f100.columns)
    common = sorted(set(spec50) & set(spec100))
    if not common:
        raise ValueError("No matching spectral columns (e.g. T1.000S) in both files")
    periods = np.array(common)
    t_used = float(periods[np.argmin(np.abs(periods - period))])

    # align both files on RSN
    f50 = f50.copy()
    f50["_rsn"] = pd.to_numeric(f50[cm50["rsn"]], errors="coerce")
    f100 = f100.copy()
    f100["_rsn"] = pd.to_numeric(f100[cm100["rsn"]], errors="coerce")
    f50 = f50.dropna(subset=["_rsn"]).drop_duplicates("_rsn").set_index("_rsn")
    f100 = f100.dropna(subset=["_rsn"]).drop_duplicates("_rsn").set_index("_rsn")
    rsns = f50.index.intersection(f100.index)
    f50, f100 = f50.loc[rsns], f100.loc[rsns]

    # ratio at every period (for the overview plot)
    s50 = np.column_stack([_clean_numeric(f50[spec50[t]]).values for t in periods])
    s100 = np.column_stack([_clean_numeric(f100[spec100[t]]).values for t in periods])
    with np.errstate(divide="ignore", invalid="ignore"):
        all_ratio = np.where((s50 > 0) & (s100 > 0), s100 / s50, np.nan)
    all_ratio[(all_ratio < 0.95) | (all_ratio > 5)] = np.nan

    # long periods are unreliable when the record was high-pass filtered at a
    # frequency close to 1/T: keep a period only if T <= 1 / lowest usable freq
    n_unusable = None
    if usable_filter and "hp_freq" in cm50:
        hp = _clean_numeric(f50[cm50["hp_freq"]]).values
        with np.errstate(divide="ignore", invalid="ignore"):
            tmax = np.where(hp > 0, 1.0 / hp, np.inf)
        too_long = periods[None, :] > tmax[:, None]
        n_unusable = int(too_long[:, int(np.argmin(np.abs(periods - period)))].sum())
        all_ratio[too_long] = np.nan
    period_stats = pd.DataFrame({
        "period": periods,
        "p16": np.nanpercentile(all_ratio, 16, axis=0),
        "median": np.nanpercentile(all_ratio, 50, axis=0),
        "p84": np.nanpercentile(all_ratio, 84, axis=0),
    })

    # metadata table with canonical names
    data = pd.DataFrame(index=rsns)
    data["rsn"] = rsns.values
    for canon, col in cm50.items():
        if canon == "rsn":
            continue
        data[canon] = f50[col].values
        if canon == "mechanism":
            data[canon] = _encode_mechanism(data[canon])
        elif canon != "event_id":
            data[canon] = _clean_numeric(data[canon])
    if "event_id" not in data:
        data["event_id"] = np.nan  # handled later (falls back to RSN groups)

    j = int(np.where(periods == t_used)[0][0])
    data["ratio"] = all_ratio[:, j]
    n_before = len(data)
    data = data.dropna(subset=["ratio", "magnitude", "vs30"])
    if "rrup" in data:
        data = data.dropna(subset=["rrup"])
    data = data.reset_index(drop=True)
    data["log_ratio"] = np.log(data["ratio"])

    info = {
        "rotd50_source": sources[0],
        "rotd100_source": sources[1],
        "columns_matched": cm50,
        "dropped_for_unusable_period": n_unusable,
        "period_requested": period,
        "period_used": t_used,
        "records_matched_by_rsn": int(len(rsns)),
        "records_after_cleaning": int(len(data)),
        "records_dropped": int(n_before - len(data)),
        "n_events": int(data["event_id"].nunique()) if data["event_id"].notna().any() else None,
    }
    return data, period_stats, info


def find_search_results(folder):
    """Locate the _SearchResults.csv that PEER puts in a time-series download."""
    folder = Path(folder)
    exact = folder / "_SearchResults.csv"
    if exact.exists():
        return exact
    hits = sorted(folder.glob("*SearchResults*.csv"))
    if not hits:
        raise FileNotFoundError(f"No *SearchResults*.csv found in {folder}")
    return hits[0]


def build_dataset_from_timeseries(folder, period=1.0, periods=None, workers=1,
                                  cache=None, recompute=False, limit=None,
                                  colmap_path=None, usable_filter=True):
    """Build the dataset from a PEER time-series download folder.

    RotD50 and RotD100 are computed from the horizontal .AT2 files (rotd.py);
    magnitude, distance, Vs30 etc. come from the folder's _SearchResults.csv.
    The slow step (response spectra) is cached in a CSV so it only runs once.
    """
    from rotd import DEFAULT_PERIODS, compute_folder, list_rsns

    folder = Path(folder)
    periods = sorted(set(round(p, 3) for p in (periods or DEFAULT_PERIODS)) | {round(period, 3)})
    # a --limit trial neither reads nor writes the cache, so it can never
    # leave a partial cache behind
    cache = Path(cache) if (cache and limit is None) else None
    skipped_file = cache.with_name("rotd_skipped.csv") if cache else None

    rot, failures = None, []
    if cache and cache.exists() and not recompute:
        cached = pd.read_csv(cache)
        skipped = (set(pd.read_csv(skipped_file)["rsn"]) if skipped_file.exists() else set())
        have = {round(float(c.split("_T")[1]), 3) for c in cached.columns if c.startswith("RotD50_T")}
        missing_rsns = list_rsns(folder) - set(cached["rsn"]) - skipped
        if not set(periods) <= have:
            print("Cache does not cover the requested periods; recomputing.")
        elif missing_rsns:
            print(f"Cache is missing {len(missing_rsns)} records now in the folder; recomputing.")
        else:
            rot = cached
            print(f"Using cached RotD results: {cache} ({len(rot)} records)")
    if rot is None:
        rot, failures = compute_folder(folder, periods, workers=workers, limit=limit)
        if failures:
            print(f"  {len(failures)} records skipped, e.g. {failures[:5]}")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            rot.to_csv(cache, index=False)
            pd.DataFrame(failures, columns=["rsn", "reason"]).to_csv(skipped_file, index=False)
    n_computed = len(rot)

    search_file = find_search_results(folder)
    with open(search_file, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        search_lines = f.readlines()

    # PEER reports contain a two-column search-criteria preamble, followed by
    # the actual 22-column metadata table and sometimes additional report
    # sections. Find the exact metadata header, then keep only its data rows.
    header_row = next(
        (
            i for i, line in enumerate(search_lines)
            if line.strip().lower().startswith(
                "result id, spectral ordinate, record sequence number"
            )
        ),
        None,
    )

    # Accept either a full PEER report or a standalone CSV containing the
    # 22-column metadata table.
    if header_row is None:
        first_line = search_lines[0] if search_lines else ""
        if first_line.strip().lower().startswith(
            "result id,spectral ordinate,record sequence number"
        ):
            header_row = 0
        else:
            raise ValueError(f"Could not find the metadata table header in {search_file}")

    header = next(csv.reader([search_lines[header_row]], skipinitialspace=True))
    expected_fields = len(header)
    metadata_rows = []
    started_rows = False
    for line in search_lines[header_row + 1:]:
        stripped = line.strip()
        if not stripped:
            if started_rows:
                break
            continue
        if stripped.startswith("--"):
            if started_rows:
                break
            continue
        try:
            row = next(csv.reader([line], skipinitialspace=True))
        except csv.Error:
            if started_rows:
                break
            continue

        # Each selected-record row begins with a numeric Result ID and should
        # have the same number of fields as the header. Stop at the next report
        # section rather than letting it corrupt pandas' CSV parsing.
        is_result_row = bool(row) and row[0].strip().isdigit()
        if is_result_row and len(row) == expected_fields:
            metadata_rows.append(row)
            started_rows = True
        elif started_rows:
            break

    if not metadata_rows:
        raise ValueError(f"No metadata records found after the header in {search_file}")

    meta = pd.DataFrame(metadata_rows, columns=[c.strip() for c in header])
    overrides = json.loads(Path(colmap_path).read_text()) if colmap_path else {}
    cm = match_columns(meta.columns, overrides)
    if "rsn" not in cm:
        raise KeyError(f"No RSN column in _SearchResults.csv. Columns: {list(meta.columns)}. "
                       "Use --colmap to map 'rsn'.")
    meta["_rsn"] = pd.to_numeric(meta[cm["rsn"]], errors="coerce")
    meta = meta.dropna(subset=["_rsn"]).drop_duplicates("_rsn")
    rot = rot.drop_duplicates("rsn").set_index("rsn")
    meta = meta[meta["_rsn"].isin(rot.index)].copy()

    f50, f100 = meta.drop(columns=["_rsn"]).copy(), pd.DataFrame({"RSN": meta["_rsn"].values})
    rsn_series = meta["_rsn"].astype(int)
    for T in periods:
        f50[f"T{T:.3f}S"] = rsn_series.map(rot[f"RotD50_T{T:.3f}"]).values
        f100[f"T{T:.3f}S"] = rsn_series.map(rot[f"RotD100_T{T:.3f}"]).values

    data, period_stats, info = assemble_dataset(
        f50, f100, period, colmap_path,
        sources=(f"computed from {folder}", f"computed from {folder}"),
        usable_filter=usable_filter)
    info["records_with_time_series"] = n_computed
    info["records_in_search_results"] = int(len(meta))
    return data, period_stats, info
