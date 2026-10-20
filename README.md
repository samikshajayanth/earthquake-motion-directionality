# Detecting and Predicting Earthquake Ground Motion Directionality

UE24CS352A Machine Learning mini-project.

Ground motion is not equally strong in every horizontal direction. We measure
this with the **directionality ratio**

    ratio(T) = RotD100(T) / RotD50(T)

where RotD50 is the median spectral acceleration over all horizontal
orientations and RotD100 is the maximum (Boore, 2010). A ratio of 1.0 means no
directionality; the largest possible value is about 1.41.

The project asks two questions of each recorded ground motion:

| Task | Type | Target |
|------|------|--------|
| **Detect** strong directionality | classification | `ratio >= threshold` (default: top 25% of records) |
| **Predict** the amount of directionality | regression | `log(ratio)` |

Features come from earthquake and site metadata: magnitude, rupture distance,
Vs30, fault mechanism, and (when available) hypocentre depth and the
source-to-site azimuth relative to the fault strike.

Models: logistic/ridge regression, random forest, histogram gradient boosting,
plus a naive baseline for each task. Evaluation uses **GroupKFold by
earthquake**, so records from one earthquake never appear in both training and
test folds (records from the same event are strongly correlated, and a random
split would give over-optimistic scores).

## How the data is produced

The PEER flatfiles do not give us RotD100, so we compute it ourselves from the
recorded time series (`rotd.py`):

1. For each record, take the two horizontal acceleration components (H1, H2).
   They are identified from the file header, not the file name; the vertical
   (UP / DWN) component is ignored.
2. Compute the response of a 5%-damped single-degree-of-freedom oscillator to
   each component, for 14 periods from 0.05 s to 10 s. The oscillator is solved
   exactly for acceleration that is piecewise linear between samples.
3. For every orientation theta (0 to 178 degrees in 2-degree steps) combine the
   responses, `u(theta) = u1 cos(theta) + u2 sin(theta)`, and take the peak.
4. RotD50 is the median of those peaks over theta, RotD100 is the maximum. The
   orientation of the maximum is saved too.

`tests/test_rotd.py` checks the oscillator against an independent ODE solver,
that two identical components give a ratio of exactly sqrt(2), and that
rotating the sensor axes leaves RotD50 and RotD100 unchanged.

## Setup

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python tests/test_rotd.py          # should print "5 tests passed"

## Get the data (PEER NGA-West2 online tool)

1. Go to <https://ngawest2.berkeley.edu/> and log in (free registration).
2. Open the NGA-West2 tool, go to **New Spectras**, click **Load Sample Values**
   and **Submit**. The target spectrum is only the entry point to the record
   search; it does not affect which records we use.
3. Open the submitted entry from the list, set wide search ranges so that many
   records match, and run the search.
4. Click **Create Time Series Records (metadata+spectra+traces)** and download
   the zip. Unzip it. The folder (for example `UnscaledAccRecord9417`) holds
   `.AT2` files for every record plus `_SearchResults.csv` with the metadata.
   Keep it out of GitHub (large).

## Run

Quick trial on 30 records (no cache is written; with so few records the
results are only a smoke test):

    python run_pipeline.py --timeseries-dir ~/Downloads/UnscaledAccRecord9417 \
        --limit 30 --out outputs_trial

Full run (the slow step, response spectra, runs once and is cached in
`outputs/rotd_cache.csv`; later runs reuse it and take seconds):

    python run_pipeline.py --timeseries-dir ~/Downloads/UnscaledAccRecord9417 \
        --workers 4 --period 1.0 --out outputs

Useful options: `--period` (spectral period in seconds), `--threshold 1.3` or
`--quantile 0.8` (what counts as "strong"), `--folds`, `--workers`,
`--recompute`, `--no-usable-filter`, `--colmap cols.json`.

If you already have flatfiles that contain both RotD50 and RotD100 spectra, use
`--rotd50 FILE --rotd100 FILE` instead of `--timeseries-dir`.

### If a column is not found

Column names vary between downloads. The loader prints which columns it
matched. If one is missing or wrong, create a JSON file mapping our names to the
real column names and pass it with `--colmap`:

    {"magnitude": "Earthquake Magnitude", "rrup": "Rrup (km)"}

Names that can be mapped: rsn, event_id, magnitude, rrup, rjb, repi, vs30,
mechanism, hypo_depth, dip, strike, width, azimuth, hp_freq.

## Outputs (in `outputs/`)

- `results_classification.csv`, `results_regression.csv`: cross-validated metrics
- `feature_importance.csv`: permutation importance on held-out earthquakes
- `ratio_by_period.csv`: median and 16th/84th percentile of the ratio at each period
- `rotd_cache.csv`: RotD50, RotD100 and orientation of the maximum for every record
- `summary.json`: dataset details (records, events, columns matched, threshold)
- `fig_*.png`: figures for the write-up and slides

## Check the code runs without the real data

    python make_synthetic_timeseries.py --out synthetic_timeseries
    python run_pipeline.py --timeseries-dir synthetic_timeseries --out outputs_synthetic

The synthetic data is random and only proves the code runs. **Never report its
numbers.**

## Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | entry point |
| `rotd.py` | reads .AT2 files, computes RotD50 / RotD100 |
| `data_io.py` | loads metadata, matches columns, builds the dataset |
| `features.py` | feature engineering, labels, grouping |
| `models.py` | models, grouped cross-validation, feature importance |
| `plots.py` | figures |
| `tests/test_rotd.py` | checks of the RotD calculation |
| `make_synthetic_timeseries.py` | stand-in data for a quick code check |
| `make_synthetic_data.py` | stand-in flatfiles for the `--rotd50/--rotd100` route |

## Limitations

- The target is the RotD100/RotD50 ratio. It measures how much stronger the
  maximum orientation is than the median, not the orientation itself.
- Records are not independent samples (many stations per earthquake), hence the
  grouped cross-validation.
- Long periods are dropped for a record when T is longer than 1 / (its lowest
  usable frequency), because high-pass filtering makes them unreliable. This
  needs that column in `_SearchResults.csv`; if it is missing, no filtering is
  applied (`dropped_for_unusable_period` is `null` in `summary.json`).
- Angles are sampled every 2 degrees, so RotD100 is accurate to well under 1%.
- The strong-directionality threshold is a modelling choice; report results for
  more than one.
