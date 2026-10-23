import csv
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace")
MANIFEST = ROOT / "outputs" / "manifest.csv"
OUTPUT = ROOT / "splits" / "split_assignments.csv"

SPLIT_BY_DATASET = {
    "dataset1": "test",
    "dataset2": "train",
    "dataset3": "validation",
}

EXPECTED_COUNTS = {
    "dataset1": 22,
    "dataset2": 40,
    "dataset3": 5,
}

with MANIFEST.open(newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fields = reader.fieldnames

if len(rows) != 67:
    raise ValueError(f"Expected 67 records, found {len(rows)}")

counts = Counter(row["dataset"] for row in rows)

if dict(counts) != EXPECTED_COUNTS:
    raise ValueError(f"Unexpected dataset counts: {counts}")

for row in rows:
    if row["dimensions_match"].lower() != "true":
        raise ValueError(f"Dimension mismatch: {row['frame_id']}")

    if int(row["unknown_label_pixels"]) != 0:
        raise ValueError(f"Unknown label pixels: {row['frame_id']}")

for row in rows:
    row["split"] = SPLIT_BY_DATASET[row["dataset"]]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["split", *fields])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved split assignments to: {OUTPUT}")

for split, count in Counter(
    row["split"] for row in rows
).items():
    print(f"{split}: {count} images")