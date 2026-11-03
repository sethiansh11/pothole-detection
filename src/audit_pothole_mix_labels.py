
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "pothole_mix" / "pothole-mix"

SPLITS = ("training", "validation", "testing")

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
}


def main():
    for split in SPLITS:
        print(f"\n{'=' * 15} {split.upper()} {'=' * 15}")

        split_dir = DATA_ROOT / split

        for source in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            mask_dir = source / "masks"
            masks = sorted(
                p for p in mask_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )

            stats = defaultdict(int)
            pixel_totals = defaultdict(int)

            for path in masks:
                with Image.open(path) as file:
                    mask = np.asarray(file.convert("RGB"))

                # Count the known label colours in this mask.
                present = []

                for name, color in COLORS.items():
                    pixels = np.all(mask == color, axis=2)
                    count = int(np.count_nonzero(pixels))
                    pixel_totals[name] += count

                    if count:
                        stats[f"masks_with_{name}"] += 1
                        present.append(name)

                background = np.all(mask == (0, 0, 0), axis=2)
                foreground = ~background

                known = np.zeros(foreground.shape, dtype=bool)
                for color in COLORS.values():
                    known |= np.all(mask == color, axis=2)

                unknown = foreground & ~known

                if np.any(unknown):
                    stats["masks_with_other_colors"] += 1
                    stats["other_color_pixels"] += int(unknown.sum())

                if len(present) > 1:
                    stats["masks_with_multiple_label_colors"] += 1

                stats["total_masks"] += 1

            print(f"\n{source.name} ({len(masks)} masks)")
            for name in COLORS:
                print(
                    f"  {name}: masks={stats[f'masks_with_{name}']}, "
                    f"pixels={pixel_totals[name]}"
                )

            print(
                "  Masks containing multiple label colours:",
                stats["masks_with_multiple_label_colors"],
            )
            print(
                "  Masks containing other colours:",
                stats["masks_with_other_colors"],
            )
            print(
                "  Other-colour pixels:",
                stats["other_color_pixels"],
            )


if __name__ == "__main__":
    main()
