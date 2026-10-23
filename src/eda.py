from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path("/workspace/data/stereo_pothole_datasets")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for dataset in sorted(ROOT.glob("dataset*")):
    print(f"\n{'=' * 50}")
    print(dataset.name)

    rgb_dir = dataset / "rgb"
    label_dir = dataset / "label"

    rgb_files = {
        p.stem: p for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    label_files = {
        p.stem: p for p in label_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    rgb_names = set(rgb_files)
    label_names = set(label_files)
    matched = sorted(rgb_names & label_names)

    print("RGB images:", len(rgb_names))
    print("Label images:", len(label_names))
    print("Matched pairs:", len(matched))
    print("RGB without label:", sorted(rgb_names - label_names))
    print("Labels without RGB:", sorted(label_names - rgb_names))

    dimensions = Counter()
    mask_colors = Counter()
    dimension_mismatches = []

    for name in matched:
        with Image.open(rgb_files[name]) as rgb:
            rgb_size = rgb.size

        with Image.open(label_files[name]) as label:
            label_size = label.size
            colors = label.convert("RGB").getcolors(
                maxcolors=1_000_000
            )

        dimensions[rgb_size] += 1

        if rgb_size != label_size:
            dimension_mismatches.append(name)

        if colors is not None:
            for pixel_count, color in colors:
                mask_colors[color] += pixel_count

    print("Image dimensions:", dimensions)
    print("Dimension mismatches:", dimension_mismatches)
    print("Most common mask colours:")

    total_pixels = sum(mask_colors.values())

    for color, count in mask_colors.most_common(10):
        percentage = 100 * count / total_pixels
        print(f"  {color}: {count:,} pixels ({percentage:.2f}%)")