
import csv
import re
from pathlib import Path
from itertools import combinations
from math import radians, sin, cos, asin, sqrt
from openpyxl import load_workbook

base = Path.home() / "Downloads"
site_file = (
    base / "NGA_West2_supporting_data_for_flatfile"
    / "NGA_West2_SiteDatabase_V032.xlsm"
)
csv_file = base / "UnscaledAccRecord9417" / "_SearchResults.csv"


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def distance_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two geographic coordinates."""
    lat1, lon1, lat2, lon2 = map(
        radians, [float(lat1), float(lon1), float(lat2), float(lon2)]
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 6371.0 * 2 * asin(sqrt(a))


# Read station catalogue
wb = load_workbook(site_file, read_only=True, data_only=True)
ws = wb["V032"]
rows = ws.iter_rows(values_only=True)
next(rows)
next(rows)
headers = [
    str(v).strip() if v is not None else ""
    for v in next(rows)
]
col = {name: i for i, name in enumerate(headers)}

stations = {}
for row in rows:
    name = row[col["Station Name"]]
    lat = row[col["Latitude"]]
    lon = row[col["Longitude"]]
    if name and lat is not None and lon is not None:
        stations.setdefault(normalize(name), []).append(
            {"name": str(name), "lat": lat, "lon": lon}
        )
wb.close()

# Read the actual PEER metadata table
records = []
with open(csv_file, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        clean = [v.strip() for v in row]
        if clean and clean[0] == "Result ID" and "Station Name" in clean:
            headers = clean
            break
    else:
        raise ValueError("Metadata header not found")

    idx = {name: headers.index(name) for name in [
        "Record Sequence Number", "Earthquake Name", "Station Name"
    ]}

    for row in reader:
        if len(row) != len(headers):
            continue
        try:
            int(row[0].strip())
        except (ValueError, IndexError):
            continue

        name = row[idx["Station Name"]].strip()
        hits = stations.get(normalize(name), [])

        # Keep only unique name matches for this first diagnostic.
        if len(hits) == 1:
            records.append({
                "rsn": row[idx["Record Sequence Number"]].strip(),
                "event": row[idx["Earthquake Name"]].strip(),
                "station": name,
                "lat": float(hits[0]["lat"]),
                "lon": float(hits[0]["lon"]),
            })

# Find pairs from the same earthquake within 5 km.
pairs = []
for event in sorted({r["event"] for r in records}):
    event_records = [r for r in records if r["event"] == event]

    for a, b in combinations(event_records, 2):
        # Do not pair a station with itself.
        if normalize(a["station"]) == normalize(b["station"]):
            continue

        d = distance_km(a["lat"], a["lon"], b["lat"], b["lon"])
        if d <= 5:
            pairs.append((event, a, b, d))

print("\n========== COORDINATE / PAIR DIAGNOSTIC ==========")
print("Records with unique station-name matches:", len(records))
print("Distinct earthquakes represented:", len({r["event"] for r in records}))
print("Same-earthquake station pairs within 5 km:", len(pairs))

print("\nFirst 20 candidate pairs:")
for event, a, b, d in pairs[:20]:
    print(
        f"{event} | {a['station']} - {b['station']} | "
        f"{d:.2f} km | RSNs {a['rsn']}, {b['rsn']}"
    )

print("\nNOTE: These are candidate pairs only. Station identity and")
print("earthquake/event labels must be verified before modeling.")
