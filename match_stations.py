
import csv
import re
from pathlib import Path
from openpyxl import load_workbook

# Files on your Mac
base = Path.home() / "Downloads"

site_file = (
    base
    / "NGA_West2_supporting_data_for_flatfile"
    / "NGA_West2_SiteDatabase_V032.xlsm"
)

csv_file = base / "UnscaledAccRecord9417" / "_SearchResults.csv"


def normalize_name(value):
    """Normalize station names for comparison."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


# Check that both files exist
for file_path in (site_file, csv_file):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

# Read the station-coordinate workbook
workbook = load_workbook(
    site_file, read_only=True, data_only=True
)
sheet = workbook["V032"]
rows = sheet.iter_rows(values_only=True)

# The first two rows precede the actual column headers
next(rows)
next(rows)
headers = next(rows)
headers = [
    str(value).strip() if value is not None else ""
    for value in headers
]
column = {name: i for i, name in enumerate(headers)}

required = ["Station Name", "Latitude", "Longitude"]
for name in required:
    if name not in column:
        raise ValueError(f"Missing workbook column: {name}")

stations = {}

for row in rows:
    name = row[column["Station Name"]]
    lat = row[column["Latitude"]]
    lon = row[column["Longitude"]]

    if name and lat is not None and lon is not None:
        key = normalize_name(name)
        stations.setdefault(key, []).append(
            (str(name), lat, lon)
        )


# Find the metadata header in the PEER CSV
records = []

with open(csv_file, newline="", encoding="utf-8-sig") as file:
    reader = csv.reader(file)
    headers = None

    for row in reader:
        cleaned = [value.strip() for value in row]

        if (
            cleaned
            and cleaned[0] == "Result ID"
            and "Station Name" in cleaned
            and "Record Sequence Number" in cleaned
        ):
            headers = cleaned
            break

    if headers is None:
        workbook.close()
        raise ValueError("Could not find the metadata header.")

    station_col = headers.index("Station Name")
    event_col = headers.index("Earthquake Name")
    rsn_col = headers.index("Record Sequence Number")

    for row in reader:
        if len(row) != len(headers):
            continue

        try:
            int(row[0].strip())
        except (ValueError, IndexError):
            continue

        records.append({
            "rsn": row[rsn_col].strip(),
            "event": row[event_col].strip(),
            "station": row[station_col].strip(),
        })

# Compare normalized station names
matched = []
unmatched = []

for record in records:
    hits = stations.get(normalize_name(record["station"]), [])

    if hits:
        matched.append((record, hits))
    else:
        unmatched.append(record)

# Report results
print("\n========== STATION MATCH REPORT ==========")
print("Records in metadata:", len(records))
print("Records whose station name matches:", len(matched))
print("Records with no exact normalized match:", len(unmatched))

print("\n========== MATCH EXAMPLES ==========")
for record, hits in matched[:15]:
    print(
        f"RSN {record['rsn']} | "
        f"{record['event']} | "
        f"{record['station']} | "
        f"Coordinates: {hits[0][1]}, {hits[0][2]} | "
        f"Possible coordinate matches: {len(hits)}"
    )

print("\n========== UNMATCHED EXAMPLES ==========")
for record in unmatched[:15]:
    print(
        f"RSN {record['rsn']} | "
        f"{record['event']} | {record['station']}"
    )

print("\nNext step: verify ambiguous matches before pairing stations.")
