
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "pothole_mix" / "pothole-mix"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("training", "validation", "testing")


def get_files(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def inspect_mask(mask_path):
    with Image.open(mask_path) as mask:
        raw = np.asarray(mask)
        print(f"      Mask mode: {mask.mode}, size: {mask.size}")

        if raw.ndim == 2:
            values, counts = np.unique(raw, return_counts=True)
            order = np.argsort(counts)[::-1][:20]
            print("      Raw mask values (value: pixels):")
            print({
                int(values[i]): int(counts[i])
                for i in order
            })

        rgb = np.asarray(mask.convert("RGB"))
        colors, counts = np.unique(
            rgb.reshape(-1, 3), axis=0, return_counts=True
        )
        order = np.argsort(counts)[::-1][:10]

        print("      Dominant RGB colors (color: pixels):")
        print({
            tuple(int(v) for v in colors[i]): int(counts[i])
            for i in order
        })


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    print("Dataset root:", DATA_ROOT)

    for split in SPLITS:
        split_dir = DATA_ROOT / split
        print(f"\n{'=' * 15} {split.upper()} {'=' * 15}")

        total_images = total_masks = total_pairs = 0
        total_missing_images = total_missing_masks = 0

        for source in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            image_dir = source / "images"
            mask_dir = source / "masks"

            images = get_files(image_dir)
            masks = get_files(mask_dir)

            image_map = {p.stem: p for p in images}
            mask_map = {p.stem: p for p in masks}

            image_ids = set(image_map)
            mask_ids = set(mask_map)
            pairs = image_ids & mask_ids

            missing_masks = sorted(image_ids - mask_ids)
            missing_images = sorted(mask_ids - image_ids)

            total_images += len(images)
            total_masks += len(masks)
            total_pairs += len(pairs)
            total_missing_masks += len(missing_masks)
            total_missing_images += len(missing_images)

            print(f"\n{source.name}:")
            print(
                f"  images={len(images)}, masks={len(masks)}, "
                f"matched_pairs={len(pairs)}"
            )
            print(
                f"  images_without_masks={len(missing_masks)}, "
                f"masks_without_images={len(missing_images)}"
            )

            if missing_masks:
                print("  Example unmatched images:", missing_masks[:5])
            if missing_images:
                print("  Example unmatched masks:", missing_images[:5])

            # Inspect one actual pair from each source in the training split.
            if split == "training" and pairs:
                stem = sorted(pairs)[0]
                image_path = image_map[stem]
                mask_path = mask_map[stem]

                with Image.open(image_path) as image:
                    print(
                        f"  Sample image: {image_path.name}, "
                        f"mode={image.mode}, size={image.size}"
                    )
                    image_size = image.size

                with Image.open(mask_path) as mask:
                    mask_size = mask.size

                print(f"  Sample pair: {stem}")
                print("  Image/mask dimensions match:", image_size == mask_size)
                inspect_mask(mask_path)

        print(f"\n{split.upper()} TOTALS:")
        print("  Images:", total_images)
        print("  Masks:", total_masks)
        print("  Matched pairs:", total_pairs)
        print("  Images without masks:", total_missing_masks)
        print("  Masks without images:", total_missing_images)


if __name__ == "__main__":
    main()
