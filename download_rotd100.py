
import getpass
import time
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth

BASE_URL = "https://www.nga-gm.org"
PAGE_SIZE = 50
OUTPUT = "rotd100_full.csv"

username = input("NGA GM username: ")
password = getpass.getpass("NGA GM password: ")

session = requests.Session()
session.headers.update({"Accept": "application/json"})

def get_with_retry(url, **kwargs):
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(15, 90), **kwargs)
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == 3:
                raise
            wait = 2 ** (attempt + 1)
            print(f"Connection issue: {exc}")
            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)

# Authenticate
response = get_with_retry(
    f"{BASE_URL}/users/login",
    auth=HTTPBasicAuth(username, password),
)
token = response.json()["token"]
session.headers["Authorization"] = f"Bearer {token}"

all_records = []
page = 1

while True:
    print(f"Requesting page {page}...")

    response = get_with_retry(
        f"{BASE_URL}/psaRotd100",
        params={"limit": PAGE_SIZE, "page": page},
    )
    try:
        data = response.json()
    except ValueError:
        print("Response is not JSON:")
        print(response.text[:2000])
        raise SystemExit

    if not isinstance(data, list):
        print("Response type:", type(data).__name__)
        print("Response content:")
        print(data)
        raise SystemExit

    records = data

    if not records:
        break

    all_records.extend(records)

    # Save progress after every successful page
    pd.DataFrame(all_records).to_csv(OUTPUT, index=False)
    print(f"Saved {len(all_records)} records so far.")

    if len(records) < PAGE_SIZE:
        break

    page += 1
    time.sleep(0.5)

print(f"Finished. Total records: {len(all_records)}")
print(f"File: {OUTPUT}")
