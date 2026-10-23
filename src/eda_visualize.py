from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path("/workspace/data/stereo_pothole_datasets")
OUTPUT = Path("/workspace/outputs")
OUTPUT.mkdir(parents=True, exist_ok=True)

POTHOLE_COLOR = (153, 0, 0)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

thumb_w, thumb_h = 420, 250
samples = []

for dataset in sorted(ROOT.glob("dataset*")):
    rgb_files = {
        p.stem: p for p in (dataset / "rgb").iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }
    label_files = {
        p.stem: p for p in (dataset / "label").iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

    names = sorted(rgb_files.keys() & label_files.keys())
    ratios = []

    for name in names:
        with Image.open(label_files[name]) as image:
            label = image.convert("RGB")
            colors = label.getcolors(maxcolors=1_000_000)

            if colors is None:
                raise ValueError(f"Too many mask colours: {name}")

            counts = {color: count for count, color in colors}
            percentage = (
                100 * counts.get(POTHOLE_COLOR, 0)
                / (label.width * label.height)
            )

        ratios.append((name, percentage))

    values = [percentage for _, percentage in ratios]

    print(f"\n{dataset.name}: {len(names)} frames")
    if values:
        print(
            f"Min: {min(values):.2f}% | "
            f"Median: {sorted(values)[len(values)//2]:.2f}% | "
            f"Max: {max(values):.2f}%"
        )
        print("Smallest labelled areas:", sorted(ratios, key=lambda x: x[1])[:3])
        print("Largest labelled areas:", sorted(ratios, key=lambda x: x[1], reverse=True)[:3])

    # Select the first, middle and last frames for visual inspection.
    for i in sorted({0, len(names) // 2, len(names) - 1}):
        if names:
            name = names[i]
            ratio = dict(ratios)[name]
            samples.append(
                (dataset.name, name, ratio,
                 rgb_files[name], label_files[name])
            )

# Create one image containing RGB, mask and overlay for each sample.
row_h = thumb_h + 28
sheet = Image.new("RGB", (thumb_w * 3, 36 + row_h * len(samples)), "white")
draw = ImageDraw.Draw(sheet)

for col, title in enumerate(["RGB input", "Ground-truth mask", "Overlay"]):
    draw.text((col * thumb_w + 10, 8), title, fill="black")

for i, (dataset_name, name, ratio, rgb_path, label_path) in enumerate(samples):
    y = 36 + i * row_h
    draw.text(
        (10, y),
        f"{dataset_name}/{name} — labelled pixels: {ratio:.2f}%",
        fill="black"
    )

    with Image.open(rgb_path) as image:
        rgb = image.convert("RGB").resize(
            (thumb_w, thumb_h), Image.Resampling.BILINEAR
        )

    with Image.open(label_path) as image:
        label = image.convert("RGB").resize(
            (thumb_w, thumb_h), Image.Resampling.NEAREST
        )

    # Visualization only: black = background, red = pothole.
    binary = label.convert("L").point(lambda p: 255 if p > 0 else 0)
    mask_display = ImageOps.colorize(
        binary, black=(20, 20, 20), white=(255, 40, 40)
    )

    red = Image.new("RGB", rgb.size, (255, 0, 0))
    blended = Image.blend(rgb, red, 0.35)
    overlay = Image.composite(blended, rgb, binary)

    sheet.paste(rgb, (0, y + 20))
    sheet.paste(mask_display, (thumb_w, y + 20))
    sheet.paste(overlay, (thumb_w * 2, y + 20))

output_file = OUTPUT / "eda_contact_sheet.png"
sheet.save(output_file)
print(f"\nContact sheet saved to: {output_file}")