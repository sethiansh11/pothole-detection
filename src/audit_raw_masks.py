
from pathlib import Path
from collections import Counter
import csv

import numpy as np
from PIL import Image


# Project root is automatically inferred from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Number of masks to inspect from each dataset.
MAX_SAMPLES_PER_SOURCE = 8

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def find_mask_directories(root):
    """Find mask folders that have a corresponding images folder."""

    results = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        if folder.name.lower() not in {"mask", "masks"}:
            continue

        sibling_images = [
            child for child in folder.parent.iterdir()
            if child.is_dir() and child.name.lower() in {"image", "images"}
        ]

        if sibling_images:
            results.append(folder)

    return sorted(results)


def get_mask_files(folder):
    """Return all supported image files inside a mask folder."""

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_evenly(files, limit):
    """Select samples spread across the folder."""

    if len(files) <= limit:
        return files

    indices = np.linspace(
        0, len(files) - 1, limit, dtype=int
    )

    return [files[i] for i in indices]


def inspect_mask(mask_path):
    """Inspect original mask without modifying or binarising it."""

    with Image.open(mask_path) as image:
        arr = np.asarray(image.convert("RGB"))

    pixels = arr.reshape(-1, 3)

    colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True
    )

    order = np.argsort(counts)[::-1]

    color_counts = {
        tuple(map(int, colors[i])): int(counts[i])
        for i in order
    }

    total_pixels = arr.shape[0] * arr.shape[1]

    non_black = np.any(arr != 0, axis=2)

    # Approximate colour groups for investigation only.
    red_like = (
        (arr[:, :, 0] > 100) &
        (arr[:, :, 0] > arr[:, :, 1] * 1.5) &
        (arr[:, :, 0] > arr[:, :, 2] * 1.5)
    )

    green_like = (
        (arr[:, :, 1] > 100) &
        (arr[:, :, 1] > arr[:, :, 0] * 1.5) &
        (arr[:, :, 1] > arr[:, :, 2] * 1.5)
    )

    result = {
        "mask_name": mask_path.name,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "unique_colors": len(colors),
        "non_black_pct": non_black.mean() * 100,
        "red_like_pct": red_like.mean() * 100,
        "green_like_pct": green_like.mean() * 100,
        "top_colors": color_counts,
        "total_pixels": total_pixels,
    }

    return result


def main():

    print("=" * 65)
    print("POTHOLEMIX RAW MASK AUDIT")
    print("=" * 65)

    mask_directories = find_mask_directories(PROJECT_ROOT)

    if not mask_directories:
        print("\nNo mask directories found.")
        print(f"Search location: {PROJECT_ROOT}")
        print("Check that your dataset is inside the project folder.")
        return

    print(f"\nFound {len(mask_directories)} mask directories.")

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mask_audit.csv"

    csv_rows = []

    for mask_dir in mask_directories:

        source = mask_dir.parent.name
        files = get_mask_files(mask_dir)

        if not files:
            continue

        samples = sample_evenly(
            files,
            MAX_SAMPLES_PER_SOURCE
        )

        aggregate_colors = Counter()
        total_sample_pixels = 0

        source_results = []

        for mask_path in samples:

            result = inspect_mask(mask_path)
            source_results.append(result)

            aggregate_colors.update(
                {
                    color: count
                    for color, count in result["top_colors"].items()
                }
            )

            total_sample_pixels += result["total_pixels"]

            top_colors = list(
                result["top_colors"].items()
            )[:5]

            color_text = "; ".join(
                f"{color}: {count}"
                for color, count in top_colors
            )

            csv_rows.append({
                "source": source,
                "mask_name": result["mask_name"],
                "width": result["width"],
                "height": result["height"],
                "unique_colors": result["unique_colors"],
                "non_black_pct": round(result["non_black_pct"], 4),
                "red_like_pct": round(result["red_like_pct"], 4),
                "green_like_pct": round(result["green_like_pct"], 4),
                "top_colors": color_text,
            })

        print("\n" + "-" * 65)
        print(f"DATASET: {source}")
        print(f"Total masks: {len(files)}")
        print(f"Audited masks: {len(samples)}")

        print("\nMost common RGB colours in sampled masks:")

        for color, count in aggregate_colors.most_common(10):

            percentage = (
                count / total_sample_pixels
            ) * 100

            print(f"  {color}: {percentage:.4f}%")

        print("\nIndividual samples:")

        for result in source_results[:4]:

            print(
                f"  {result['mask_name']} | "
                f"{result['width']}x{result['height']} | "
                f"unique={result['unique_colors']} | "
                f"red-like={result['red_like_pct']:.2f}% | "
                f"green-like={result['green_like_pct']:.2f}%"
            )

    if csv_rows:

        with open(csv_path, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=csv_rows[0].keys()
            )

            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print(f"Report saved at: {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
