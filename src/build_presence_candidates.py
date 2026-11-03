
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

MASK_NAMES = {"mask", "masks"}
IMAGE_NAMES = {"image", "images"}


def find_mask_directories(root):
    """Find mask folders with sibling image folders."""

    results = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        if folder.name.lower() not in MASK_NAMES:
            continue

        image_dirs = [
            child for child in folder.parent.iterdir()
            if child.is_dir()
            and child.name.lower() in IMAGE_NAMES
        ]

        if image_dirs:
            results.append((folder, image_dirs))

    return sorted(results, key=lambda item: str(item[0]))


def index_images(image_dirs):
    """Index images by filename stem."""

    image_map = defaultdict(list)

    for image_dir in image_dirs:
        for path in image_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_map[path.stem.lower()].append(path)

    return image_map


def image_hash(path):
    """Calculate SHA256 to identify exact duplicate images."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def inspect_mask(mask_path):
    """Inspect the mask and identify colour-coded annotations."""

    with Image.open(mask_path) as image:
        arr = np.asarray(image.convert("RGB"))

    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    # Red-like annotation: candidate pothole.
    red = (
        (r > 100) &
        (r > g * 1.5) &
        (r > b * 1.5)
    )

    # Green-like annotation: candidate crack.
    green = (
        (g > 100) &
        (g > r * 1.5) &
        (g > b * 1.5)
    )

    foreground = (r != 0) | (g != 0) | (b != 0)

    # Pixels that don't fit either known annotation colour.
    unknown = foreground & ~(red | green)

    red_count = int(red.sum())
    green_count = int(green.sum())
    unknown_count = int(unknown.sum())

    # Candidate label, not yet a verified training label.
    if red_count > 0:
        candidate_label = 1
        status = "positive_candidate_red_found"

    elif unknown_count > 0:
        candidate_label = ""
        status = "review_unknown_colours"

    else:
        candidate_label = 0
        status = "negative_candidate_no_red_found"

    return {
        "width": arr.shape[1],
        "height": arr.shape[0],
        "red_pixels": red_count,
        "green_pixels": green_count,
        "unknown_pixels": unknown_count,
        "unique_colours": len(np.unique(
            arr.reshape(-1, 3), axis=0
        )),
        "candidate_label": candidate_label,
        "label_status": status,
    }


def write_csv(path, rows):
    if not rows:
        print(f"No records to write: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():

    print("=" * 70)
    print("FULL POTHOLE PRESENCE DATASET AUDIT")
    print("=" * 70)

    directories = find_mask_directories(PROJECT_ROOT)

    if not directories:
        print("No matching mask/image folder pairs found.")
        return

    print(f"\nFound {len(directories)} mask directories.")

    all_rows = []
    summary_rows = []
    unmatched_rows = []

    hash_to_images = defaultdict(list)

    for mask_dir, image_dirs in directories:

        image_map = index_images(image_dirs)

        masks = sorted(
            path for path in mask_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        positive = 0
        negative = 0
        review = 0
        matched = 0
        ambiguous = 0

        for mask_path in masks:

            result = inspect_mask(mask_path)

            candidates = image_map.get(
                mask_path.stem.lower(), []
            )

            if len(candidates) == 1:

                image_path = candidates[0]
                matched += 1
                image_digest = image_hash(image_path)

                hash_to_images[image_digest].append(
                    str(image_path)
                )

            else:

                image_path = None
                image_digest = ""

                if len(candidates) > 1:
                    ambiguous += 1

            if result["candidate_label"] == 1:
                positive += 1

            elif result["candidate_label"] == 0:
                negative += 1

            else:
                review += 1

            row = {
                "source": mask_dir.parent.name,
                "mask_directory": str(
                    mask_dir.relative_to(PROJECT_ROOT)
                ),
                "mask_path": str(mask_path),
                "image_path": (
                    str(image_path) if image_path else ""
                ),
                "image_sha256": image_digest,
                "paired": bool(image_path),
                **result,
            }

            all_rows.append(row)

            if not image_path:
                unmatched_rows.append(row)

        summary_rows.append({
            "mask_directory": str(
                mask_dir.relative_to(PROJECT_ROOT)
            ),
            "total_masks": len(masks),
            "paired": matched,
            "unmatched": len(masks) - matched,
            "ambiguous": ambiguous,
            "positive_candidates": positive,
            "negative_candidates": negative,
            "manual_review": review,
        })

        print("\n" + "-" * 70)
        print(f"Folder: {mask_dir.relative_to(PROJECT_ROOT)}")
        print(f"Masks: {len(masks)}")
        print(f"Paired images: {matched}")
        print(f"Positive candidates: {positive}")
        print(f"Negative candidates: {negative}")
        print(f"Review required: {review}")
        print(f"Ambiguous image matches: {ambiguous}")

    # Identify exact duplicate image files.
    duplicate_rows = []

    for digest, paths in hash_to_images.items():

        unique_paths = sorted(set(paths))

        if len(unique_paths) > 1:
            duplicate_rows.append({
                "sha256": digest,
                "duplicate_count": len(unique_paths),
                "image_paths": " | ".join(unique_paths),
            })

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "presence_candidates.csv",
        all_rows
    )

    write_csv(
        output_dir / "presence_directory_summary.csv",
        summary_rows
    )

    write_csv(
        output_dir / "presence_unmatched_masks.csv",
        unmatched_rows
    )

    write_csv(
        output_dir / "presence_exact_duplicates.csv",
        duplicate_rows
    )

    print("\n" + "=" * 70)
    print("FULL AUDIT COMPLETE")
    print("=" * 70)

    print(f"Total masks: {len(all_rows)}")
    print(f"Total paired images: {sum(r['paired'] for r in all_rows)}")
    print(f"Duplicate image groups: {len(duplicate_rows)}")

    print("\nGenerated files:")

    print("outputs/presence_candidates.csv")
    print("outputs/presence_directory_summary.csv")
    print("outputs/presence_unmatched_masks.csv")
    print("outputs/presence_exact_duplicates.csv")


if __name__ == "__main__":
    main()
