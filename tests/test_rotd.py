"""Checks for rotd.py. Run with:  python tests/test_rotd.py   (or pytest)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rotd  # noqa: E402


def _noise(n, dt, seed, fmax=15.0):
    """Band-limited random 'earthquake-like' acceleration (g) with an envelope."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    f = np.fft.rfftfreq(n, dt)
    spec = np.fft.rfft(x) * (f < fmax)
    x = np.fft.irfft(spec, n)
    t = np.arange(n) * dt
    env = (t / 4.0) * np.exp(1 - t / 4.0)
    return 0.2 * x / x.std() * env


def test_sdof_matches_ode_solver():
    """The digital filter must agree with a brute-force ODE integration."""
    dt, n = 0.01, 1500
    acc = _noise(n, dt, 1)
    t = np.arange(n) * dt
    for T in (0.1, 0.5, 1.0, 3.0):
        w, xi = 2 * np.pi / T, rotd.DAMPING

        def rhs(tt, y):
            a = np.interp(tt, t, acc)
            return [y[1], -a - 2 * xi * w * y[1] - w * w * y[0]]

        sol = solve_ivp(rhs, (0, t[-1]), [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12,
                        max_step=dt / 5)
        ref = sol.y[0]
        got = rotd.sdof_displacement(acc, dt, T)
        err = np.abs(got - ref).max() / np.abs(ref).max()
        assert err < 1e-4, f"T={T}: relative error {err:.2e}"


def test_identical_components_give_sqrt2():
    dt = 0.01
    a = _noise(2000, dt, 2)
    res = rotd.rotd_spectra(a, a.copy(), dt, [0.2, 1.0, 3.0])
    ratio = res["rotd100"] / res["rotd50"]
    assert np.allclose(ratio, np.sqrt(2), atol=0.01), ratio
    assert np.allclose(res["theta"], 45.0, atol=2.0), res["theta"]


def test_ratio_bounds_for_independent_components():
    dt = 0.01
    res = rotd.rotd_spectra(_noise(2000, dt, 3), _noise(2000, dt, 4), dt)
    ratio = res["rotd100"] / res["rotd50"]
    assert (ratio >= 1.0).all() and (ratio <= np.sqrt(2) + 1e-6).all(), ratio


def test_rotation_invariance():
    """Rotating the sensor axes must not change RotD50 / RotD100."""
    dt = 0.01
    h1, h2 = _noise(2000, dt, 5), 0.6 * _noise(2000, dt, 6)
    base = rotd.rotd_spectra(h1, h2, dt, [0.3, 1.0, 2.0])
    phi = np.deg2rad(30.0)
    r1 = np.cos(phi) * h1 + np.sin(phi) * h2
    r2 = -np.sin(phi) * h1 + np.cos(phi) * h2
    rot = rotd.rotd_spectra(r1, r2, dt, [0.3, 1.0, 2.0])
    assert np.allclose(base["rotd50"], rot["rotd50"], rtol=1e-3)
    assert np.allclose(base["rotd100"], rot["rotd100"], rtol=1e-3)
    shift = (base["theta"] - rot["theta"]) % 180
    assert np.allclose(shift, 30.0, atol=2.5) or np.allclose(shift, 30.0 - 180, atol=2.5), shift


def _write_at2(path, acc, dt, label):
    lines = ["PEER NGA STRONG MOTION DATABASE RECORD",
             f"TEST EARTHQUAKE 01/01/2000, TEST STATION, {label}",
             "ACCELERATION TIME SERIES IN UNITS OF G",
             f"NPTS= {len(acc):5d}, DT= {dt:.4f} SEC"]
    for i in range(0, len(acc), 5):
        lines.append("".join(f"{v:15.7E}" for v in acc[i:i + 5]))
    Path(path).write_text("\n".join(lines) + "\n")


def test_folder_pairing_and_azimuth():
    dt = 0.01
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        h1, h2, v = _noise(1500, dt, 7), _noise(1500, dt, 8), _noise(1500, dt, 9)
        # record 1: azimuths 180 / 270, vertical 'UP'
        _write_at2(d / "RSN1_TEST.A_A-STA180.AT2", h1, dt, "180")
        _write_at2(d / "RSN1_TEST.A_A-STA270.AT2", h2, dt, "270")
        _write_at2(d / "RSN1_TEST.A_A-STAUP.AT2", v, dt, "UP")
        # record 2: vertical named DWN, horizontals 45 / 135
        _write_at2(d / "RSN2_TEST.B_B-XYZDWN.AT2", v, dt, "DWN")
        _write_at2(d / "RSN2_TEST.B_B-XYZ045.AT2", h1, dt, "45")
        _write_at2(d / "RSN2_TEST.B_B-XYZ135.AT2", h2, dt, "135")
        # other file types in the folder must be ignored
        (d / "RSN1_TEST.A_A-STA180.DT2").write_text("junk")
        (d / "_SearchResults.csv").write_text("a,b\n1,2\n")

        df, failures = rotd.compute_folder(d, periods=[0.5, 1.0], workers=1)
        assert not failures, failures
        assert sorted(df["rsn"]) == [1, 2]
        r1 = df[df.rsn == 1].iloc[0]
        assert {r1.h1_file, r1.h2_file} == {"RSN1_TEST.A_A-STA180.AT2", "RSN1_TEST.A_A-STA270.AT2"}
        assert r1["RotD100_T1.000"] >= r1["RotD50_T1.000"] > 0
        assert 0 <= r1["MaxAzimuth_T1.000"] < 180


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")
