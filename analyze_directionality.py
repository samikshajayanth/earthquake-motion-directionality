
import re
import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path.home() / "Downloads" / "UnscaledAccRecord9417"
OUT_FILE = Path("outputs_trial") / "orientation_estimates.csv"


def read_at2(filepath):
    text = filepath.read_text(errors="replace")
    lines = text.splitlines()

    npts = None
    dt = None

    for line in lines:
        match = re.search(
            r"NPTS\s*=\s*(\d+)\s*,?\s*DT\s*=\s*([+\-0-9.Ee]+)",
            line,
            re.IGNORECASE,
        )
        if match:
            npts = int(match.group(1))
            dt = float(match.group(2))
            break

    if npts is None or dt is None:
        raise ValueError(f"Could not read NPTS/DT: {filepath.name}")

    values = []
    started = False

    for line in lines:
        if re.search(r"NPTS\s*=", line, re.IGNORECASE):
            started = True
            continue
        if started:
            for token in line.split():
                try:
                    values.append(float(token.replace("D", "E")))
                except ValueError:
                    pass

    if len(values) < npts:
        raise ValueError(
            f"{filepath.name}: expected {npts} samples, got {len(values)}"
        )

    return np.asarray(values[:npts], dtype=float), dt



def orientation_from_two_components(a, b, azimuth_a, azimuth_b):
    """
    Estimate the north-referenced orientation that maximizes horizontal PGA.

    azimuth_a and azimuth_b are the component azimuths in degrees,
    measured clockwise from north. The result is an axis orientation
    in [0, 180), because opposite directions describe the same axis.
    """
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]

    # Check that the two components are approximately orthogonal.
    separation = (azimuth_b - azimuth_a) % 180
    if not np.isclose(separation, 90, atol=2):
        raise ValueError(
            f"Components are not approximately orthogonal: "
            f"{azimuth_a}° and {azimuth_b}°"
        )

    angles = np.arange(180, dtype=float)
    best_angle = None
    best_pga = -np.inf

    alpha = np.deg2rad(azimuth_a)
    beta = np.deg2rad(azimuth_b)

    for phi in angles:
        direction = np.deg2rad(phi)

        # Project each recorded component onto the candidate direction.
        rotated = (
            a * np.cos(direction - alpha)
            + b * np.cos(direction - beta)
        )

        pga = float(np.max(np.abs(rotated)))

        if pga > best_pga:
            best_pga = pga
            best_angle = int(phi)

    return best_angle, best_pga, n



def parse_component(filename):
    """
    Return RSN and final component direction token where detectable.
    This is only a filename heuristic; it is not a verified orientation map.
    """
    name = Path(filename).stem
    match = re.match(r"RSN(\d+)_", name, re.IGNORECASE)
    rsn = match.group(1) if match else ""

    suffix = re.search(r"(\d{3})$", name)
    direction = suffix.group(1) if suffix else ""

    return rsn, direction


def main():
    files = sorted(DATA_DIR.glob("*.AT2"))
    print("AT2 files found:", len(files))

    # Group records by RSN. Each RSN may have multiple files and directions.
    groups = {}
    for path in files:
        rsn, direction = parse_component(path.name)
        if rsn:
            groups.setdefault(rsn, []).append((path, direction))

    rows = []
    skipped = []

    for rsn, components in sorted(groups.items(), key=lambda x: int(x[0])):
        # Use files with numeric three-digit component suffixes.
        directional = [(p, d) for p, d in components if d]
        if len(directional) < 2:
            skipped.append((rsn, "fewer than two numeric-angle files"))
            continue

        # Take the first two distinct numeric-angle files for this RSN.
        # The actual component identity and orthogonality must be verified.
        first, second = directional[0][0], directional[1][0]

        try:
            a, dt_a = read_at2(first)
            b, dt_b = read_at2(second)

            if not np.isclose(dt_a, dt_b):
                skipped.append((rsn, "different sampling intervals"))
                continue

            angle, pga, n = orientation_from_two_components(
            a, b, float(directional[0][1]), float(directional[1][1])
            )

            rows.append({
                "rsn": rsn,
                "file_component_1": first.name,
                "file_component_2": second.name,
                "component_1_suffix": directional[0][1],
                "component_2_suffix": directional[1][1],
                "dt_seconds": dt_a,
                "samples_used": n,
                "orientation_relative_degrees": angle,
                "max_rotated_pga_g": pga,
            })
        except Exception as exc:
            skipped.append((rsn, str(exc)))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "rsn", "file_component_1", "file_component_2",
        "component_1_suffix", "component_2_suffix", "dt_seconds",
        "samples_used", "orientation_relative_degrees", "max_rotated_pga_g"
    ]

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("RSNs with provisional orientation estimates:", len(rows))
    print("RSNs skipped:", len(skipped))
    print("Output:", OUT_FILE.resolve())

    print("\nFirst 10 estimates:")
    for row in rows[:10]:
        print(
            row["rsn"], "|",
            row["orientation_relative_degrees"], "degrees |",
            f"PGA={row['max_rotated_pga_g']:.5f} g |",
            row["file_component_1"], "+", row["file_component_2"]
        )

    if skipped:
        print("\nFirst 10 skipped RSNs:")
        for item in skipped[:10]:
            print(item)


if __name__ == "__main__":
    main()
