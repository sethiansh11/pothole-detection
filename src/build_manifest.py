import csv
from pathlib import Path
from PIL import Image

ROOT = Path("/workspace/data/stereo_pothole_datasets")
OUTPUT = Path("/workspace/outputs/manifest.csv")

POTHOLE = (153, 0, 0)
BACKGROUND = (0, 0, 0)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
rows = []

for dataset in sorted(ROOT.glob("dataset*")):
    rgb_dir = dataset / "rgb"
    label_dir = dataset / "label"

    rgb_files = {
        p.stem: p for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }
    label_files = {
        p.stem: p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

    if set(rgb_files) != set(label_files):
        raise ValueError(f"Unmatched RGB/label files in {dataset.name}")

    for frame_id in sorted(rgb_files):
        rgb_path = rgb_files[frame_id]
        label_path = label_files[frame_id]

        with Image.open(rgb_path) as rgb:
            width, height = rgb.size

        with Image.open(label_path) as image:
            label = image.convert("RGB")
            colors = label.getcolors(maxcolors=1_000_000)

        if colors is None:
            raise ValueError(f"Too many label colours: {label_path}")

        color_counts = {color: count for count, color in colors}
        total_pixels = width * height
        pothole_pixels = color_counts.get(POTHOLE, 0)
        background_pixels = color_counts.get(BACKGROUND, 0)
        unknown_pixels = total_pixels - pothole_pixels - background_pixels

        with Image.open(label_path) as label:
            dimensions_match = label.size == (width, height)

        rows.append({
            "dataset": dataset.name,
            "frame_id": frame_id,
            "rgb_path": str(rgb_path.relative_to(ROOT)),
            "label_path": str(label_path.relative_to(ROOT)),
            "width": width,
            "height": height,
            "dimensions_match": dimensions_match,
            "pothole_pixels": pothole_pixels,
            "total_pixels": total_pixels,
            "pothole_percent": round(
                100 * pothole_pixels / total_pixels, 4
            ),
            "unknown_label_pixels": unknown_pixels,
        })

if not rows:
    raise RuntimeError(f"No image pairs found under {ROOT}")

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Images recorded: {len(rows)}")
print(f"Manifest saved to: {OUTPUT}")

for dataset in sorted({row["dataset"] for row in rows}):
    count = sum(row["dataset"] == dataset for row in rows)
    print(f"{dataset}: {count} images")