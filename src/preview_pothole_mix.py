
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "pothole_mix" / "pothole-mix"
OUTPUT = ROOT / "outputs" / "pothole_mix_label_preview.png"

RED = (255, 0, 0)    # Pothole target
GREEN = (0, 255, 0)  # Crack
THUMB_W, THUMB_H = 320, 180
ROW_H = 225


def find_example(split, source, kind):
    base = DATA_ROOT / split / source
    image_dir = base / "images"
    mask_dir = base / "masks"

    for mask_path in sorted(mask_dir.glob("*")):
        if not mask_path.is_file():
            continue

        image_candidates = [
            path for path in image_dir.glob(mask_path.stem + ".*")
            if path.is_file()
        ]
        if not image_candidates:
            continue

        with Image.open(mask_path) as file:
            mask = np.asarray(file.convert("RGB"))

        red = np.all(mask == RED, axis=2).any()
        green = np.all(mask == GREEN, axis=2).any()

        matches = {
            "red_only": red and not green,
            "green_only": green and not red,
            "both": red and green,
        }

        if matches.get(kind, False):
            with Image.open(image_candidates[0]) as file:
                image = file.convert("RGB").copy()

            return image, mask_path, red, green

    raise RuntimeError(
        f"No {kind} example found in {split}/{source}"
    )


def make_overlay(image, mask):
    image_array = np.asarray(image).copy()
    overlay = image_array.astype(np.float32)

    red_mask = np.all(mask == RED, axis=2)
    green_mask = np.all(mask == GREEN, axis=2)

    overlay[red_mask] = (
        0.45 * overlay[red_mask] + 0.55 * np.array(RED)
    )
    overlay[green_mask] = (
        0.45 * overlay[green_mask] + 0.55 * np.array(GREEN)
    )

    return Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    )


def fit_image(image):
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), "white")
    fitted = ImageOps.contain(image, (THUMB_W, THUMB_H))
    x = (THUMB_W - fitted.width) // 2
    y = (THUMB_H - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def main():
    cases = [
        ("Pothole-600: pothole", "pothole600", "red_only"),
        ("CNR: pothole", "cnr-road-dataset", "red_only"),
        ("Crack500: crack", "crack500", "green_only"),
        ("Mixed source: pothole only", "cracks-and-potholes-in-road", "red_only"),
        ("Mixed source: crack only", "cracks-and-potholes-in-road", "green_only"),
        ("Mixed source: both classes", "cracks-and-potholes-in-road", "both"),
    ]

    canvas = Image.new(
        "RGB", (THUMB_W * 3, ROW_H * len(cases)), "white"
    )
    draw = ImageDraw.Draw(canvas)

    for row, (title, source, kind) in enumerate(cases):
        image, mask_path, has_red, has_green = find_example(
            "training", source, kind
        )

        with Image.open(mask_path) as file:
            mask = np.asarray(file.convert("RGB"))

        binary = np.all(mask == RED, axis=2).astype(np.uint8) * 255
        binary_image = Image.fromarray(binary, mode="L").convert("RGB")

        y = row * ROW_H
        draw.text((8, y + 4), title, fill="black")

        panels = [
            (image, "Original image"),
            (make_overlay(image, mask), "Red=pothole; green=crack"),
            (binary_image, "Binary target: red only"),
        ]

        for col, (panel, label) in enumerate(panels):
            x = col * THUMB_W
            draw.text((x + 5, y + 24), label, fill="black")
            fitted = fit_image(panel)
            canvas.paste(fitted, (x, y + 43))

        print(
            f"{title}: red={has_red}, green={has_green}, "
            f"file={mask_path.name}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)
    print("\nPreview saved to:", OUTPUT)


if __name__ == "__main__":
    main()
