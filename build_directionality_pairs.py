import csv
import re
from pathlib import Path
from itertools import combinations
from math import radians, sin, cos, asin, sqrt

from openpyxl import load_workbook

BASE = Path.home() / "Downloads"
SITE_FILE = (
    BASE / "NGA_West2_supporting_data_for_flatfile"
    / "NGA_West2_SiteDatabase_V032.xlsm"
)
METADATA_FILE = BASE / "UnscaledAccRecord9417" / "_SearchResults.csv"
ORIENTATION_FILE = Path("outputs_trial") / "orientation_estimates.csv"
OUT_FILE = Path("outputs_trial") / "station_directionality_pairs.csv"
MAX_DISTANCE_KM = 5.0


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        radians, [float(lat1), float(lon1), float(lat2), float(lon2)]
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(h))


def read_metadata(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            header = [v.strip() for v in row]
            if (
                header and header[0] == "Result ID"
                and "Station Name" in header
                and "Record Sequence Number" in header
                and "Earthquake Name" in header
            ):
                break
        else:
            raise ValueError(f"Metadata header not found in {path}")

        indexes = {
            key: header.index(key)
            for key in (
                "Record Sequence Number", "Earthquake Name", "Station Name"
            )
        }

        records = []
        for row in reader:
            if len(row) != len(header):
                continue
            try:
                rsn = str(int(row[indexes["Record Sequence Number"]].strip()))
            except (ValueError, IndexError):
                continue

            records.append({
                "rsn": rsn,
                "event": row[indexes["Earthquake Name"]].strip(),
                "station": row[indexes["Station Name"]].strip(),
            })
        return records


def read_stations(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["V032"]
    rows = ws.iter_rows(values_only=True)
    next(rows)
    next(rows)
    header = [
        str(v).strip() if v is not None else ""
        for v in next(rows)
    ]
    col = {name: i for i, name in enumerate(header)}

    stations = {}
    for row in rows:
        name = row[col["Station Name"]]
        lat = row[col["Latitude"]]
        lon = row[col["Longitude"]]
        
        if not name or lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            continue

        # Reject invalid geographic coordinates.
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        stations.setdefault(normalize(name), []).append({
            "name": str(name),
            "lat": lat,
            "lon": lon,
        })

    wb.close()
    return stations


def read_orientations(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run analyze_directionality.py first."
        )

    result = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rsn = str(int(row["rsn"]))
            angle = row.get("orientation_relative_degrees", "").strip()
            if angle:
                result[rsn] = float(angle) % 180
    return result


def angular_axis_difference(a, b):
    """Smallest difference between two unoriented axes, in degrees."""
    difference = abs((a - b) % 180)
    return min(difference, 180 - difference)


def main():
    for path in (SITE_FILE, METADATA_FILE, ORIENTATION_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    stations = read_stations(SITE_FILE)
    metadata = read_metadata(METADATA_FILE)
    orientations = read_orientations(ORIENTATION_FILE)

    records = []
    unmatched = 0
    ambiguous = 0

    for record in metadata:
        hits = stations.get(normalize(record["station"]), [])
        if len(hits) != 1:
            if len(hits) == 0:
                unmatched += 1
            else:
                ambiguous += 1
            continue

        station = hits[0]
        records.append({
            **record,
            "lat": station["lat"],
            "lon": station["lon"],
            "orientation": orientations.get(record["rsn"]),
        })

    pairs = []
    for event in sorted({r["event"] for r in records}):
        event_records = [r for r in records if r["event"] == event]

        for a, b in combinations(event_records, 2):
            if normalize(a["station"]) == normalize(b["station"]):
                continue

            # A station pair is useful only if both orientations exist.
            if a["orientation"] is None or b["orientation"] is None:
                continue

            distance = distance_km(
                a["lat"], a["lon"], b["lat"], b["lon"]
            )
            if distance > MAX_DISTANCE_KM:
                continue

            pairs.append({
                "event": event,
                "rsn_1": a["rsn"],
                "station_1": a["station"],
                "orientation_1_deg": round(a["orientation"], 2),
                "rsn_2": b["rsn"],
                "station_2": b["station"],
                "orientation_2_deg": round(b["orientation"], 2),
                "distance_km": round(distance, 3),
                "delta_orientation_deg": round(
                    angular_axis_difference(
                        a["orientation"], b["orientation"]
                    ), 2
                ),
            })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event", "rsn_1", "station_1", "orientation_1_deg",
        "rsn_2", "station_2", "orientation_2_deg", "distance_km",
        "delta_orientation_deg",
    ]
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    print("\n========== DIRECTIONALITY PAIR DATASET ==========")
    print("Metadata records:", len(metadata))
    print("Records with unique station-coordinate matches:", len(records))
    print("Station names without a match:", unmatched)
    print("Station names with ambiguous matches:", ambiguous)
    print("Orientation estimates available:", len(orientations))
    print("Eligible same-event pairs within 5 km:", len(pairs))
    print("Output:", OUT_FILE.resolve())

    for pair in pairs[:10]:
        print(
            f"{pair['event']} | {pair['station_1']} - {pair['station_2']} "
            f"| {pair['distance_km']} km "
            f"| Δθ={pair['delta_orientation_deg']}°"
        )

    if len(pairs) < 30:
        print(
            "\nWARNING: This is a small preliminary dataset; "
            "it is not sufficient to establish reliable model performance."
        )


if __name__ == "__main__":
    main()
