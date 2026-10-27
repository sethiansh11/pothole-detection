import sys
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset
from model import build_model

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

ROOT = Path("/workspace")
DATA_ROOT = ROOT / "data" / "stereo_pothole_datasets"
OUTPUT_DIR = ROOT / "outputs"

THRESHOLD = 0.80
PATCH_SIZE = 128
STRIDE = 64
MIN_FP_PIXELS = 128
MAX_REGIONS = 10
MAX_PER_FRAME = 3

device = torch.device("cuda:0")

checkpoint = torch.load(
    OUTPUT_DIR / "lraspp_frozen_best.pth",
    map_location=device,
    weights_only=True,
)

model = build_model().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dataset = PotholeDataset("validation")

mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)

candidates = []
predictions_by_frame = {}

# Find image regions containing many false-positive pixels.
with torch.inference_mode():
    for index in range(len(dataset)):
        image, mask = dataset[index]
        sample = dataset.samples[index]

        inputs = image.unsqueeze(0).to(device)
        inputs = (inputs - mean) / std

        logits = model(inputs)["out"]
        probability = logits.softmax(dim=1)[0, 1].cpu().numpy()

        prediction = probability >= THRESHOLD
        target = mask.squeeze(0).numpy() >= 0.5

        false_positives = prediction & ~target
        predictions_by_frame[index] = prediction

        height, width = false_positives.shape

        for y in range(0, height - PATCH_SIZE + 1, STRIDE):
            for x in range(0, width - PATCH_SIZE + 1, STRIDE):
                fp_count = int(
                    false_positives[
                        y:y + PATCH_SIZE,
                        x:x + PATCH_SIZE
                    ].sum()
                )

                if fp_count >= MIN_FP_PIXELS:
                    candidates.append({
                        "frame_index": index,
                        "dataset": sample["dataset"],
                        "frame_id": sample["frame_id"],
                        "x": x,
                        "y": y,
                        "fp_pixels": fp_count,
                    })

# Prefer high-FP regions but avoid inspecting overlapping patches.
candidates.sort(key=lambda c: c["fp_pixels"], reverse=True)

selected = []
per_frame = {}

for candidate in candidates:
    index = candidate["frame_index"]

    if per_frame.get(index, 0) >= MAX_PER_FRAME:
        continue

    overlaps = False

    for old in selected:
        if old["frame_index"] != index:
            continue

        dx = max(
            0,
            min(candidate["x"], old["x"])
            + PATCH_SIZE
            - max(candidate["x"], old["x"])
        )
        dy = max(
            0,
            min(candidate["y"], old["y"])
            + PATCH_SIZE
            - max(candidate["y"], old["y"])
        )

        if dx * dy / (PATCH_SIZE * PATCH_SIZE) > 0.20:
            overlaps = True
            break

    if overlaps:
        continue

    selected.append(candidate)
    per_frame[index] = per_frame.get(index, 0) + 1

    if len(selected) >= MAX_REGIONS:
        break

if not selected:
    raise RuntimeError(
        "No suitable regions found. Lower MIN_FP_PIXELS and retry."
    )

# Build a contact sheet: road image, annotation, error overlay.
tile = 280
header_height = 40
row_height = tile + 44

sheet = Image.new(
    "RGB",
    (tile * 3, header_height + row_height * len(selected)),
    "white",
)
draw = ImageDraw.Draw(sheet)

for col, title in enumerate([
    "Road image",
    "Ground-truth mask",
    "Prediction errors",
]):
    draw.text((col * tile + 8, 12), title, fill="black")

audit_rows = []

for number, candidate in enumerate(selected, start=1):
    index = candidate["frame_index"]
    sample = dataset.samples[index]
    x, y = candidate["x"], candidate["y"]

    rgb_path = DATA_ROOT / sample["rgb_path"]
    label_path = DATA_ROOT / sample["label_path"]

    with Image.open(rgb_path) as file:
        rgb = file.convert("RGB")

    with Image.open(label_path) as file:
        label = file.convert("RGB")

    original_w, original_h = rgb.size

    # Map the model-resolution patch to original-resolution coordinates.
    left = round(x * original_w / 640)
    top = round(y * original_h / 384)
    right = round((x + PATCH_SIZE) * original_w / 640)
    bottom = round((y + PATCH_SIZE) * original_h / 384)
    box = (left, top, right, bottom)

    rgb_crop = rgb.crop(box)
    label_crop = label.crop(box)

    gt_array = np.asarray(label_crop)
    ground_truth = np.all(
        gt_array == np.array([153, 0, 0]),
        axis=2,
    )

    # Convert model predictions back to original image resolution.
    predicted_small = predictions_by_frame[index].astype(np.uint8) * 255

    predicted_full = Image.fromarray(predicted_small).resize(
        (original_w, original_h),
        Image.Resampling.NEAREST,
    )

    prediction_crop = np.asarray(
        predicted_full.crop(box)
    ) > 0

    true_positive = prediction_crop & ground_truth
    false_positive = prediction_crop & ~ground_truth
    false_negative = ~prediction_crop & ground_truth

    rgb_array = np.asarray(rgb_crop).copy()

    # Green = matches label; red = extra prediction;
    # blue = annotated pothole pixel missed by the model.
    overlay = rgb_array.astype(np.float32)

    overlay[true_positive] = (
        0.45 * overlay[true_positive]
        + 0.55 * np.array([0, 220, 0])
    )
    overlay[false_positive] = (
        0.45 * overlay[false_positive]
        + 0.55 * np.array([255, 0, 0])
    )
    overlay[false_negative] = (
        0.45 * overlay[false_negative]
        + 0.55 * np.array([0, 80, 255])
    )

    mask_display = np.zeros_like(rgb_array)
    mask_display[ground_truth] = (220, 40, 40)

    visuals = [
        rgb_crop,
        Image.fromarray(mask_display),
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)),
    ]

    row_y = header_height + (number - 1) * row_height

    draw.text(
        (8, row_y),
        f"{number:02d} {candidate['dataset']}/{candidate['frame_id']} "
        f"| FP pixels: {candidate['fp_pixels']}",
        fill="black",
    )

    for col, visual in enumerate(visuals):
        visual = visual.resize(
            (tile, tile),
            Image.Resampling.BILINEAR if col == 0
            else Image.Resampling.NEAREST,
        )
        sheet.paste(visual, (col * tile, row_y + 20))

    audit_rows.append({
        "audit_id": number,
        "dataset": candidate["dataset"],
        "frame_id": candidate["frame_id"],
        "threshold": THRESHOLD,
        "x_model": x,
        "y_model": y,
        "false_positive_pixels": candidate["fp_pixels"],
        "classification": "",
        "notes": "",
    })

sheet_path = OUTPUT_DIR / "false_positive_audit.png"
sheet.save(sheet_path)

csv_path = OUTPUT_DIR / "false_positive_audit.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=audit_rows[0].keys())
    writer.writeheader()
    writer.writerows(audit_rows)

print("Validation images:", len(dataset))
print("Candidate windows:", len(candidates))
print("Selected regions:", len(selected))
print("Contact sheet:", sheet_path)
print("Audit CSV:", csv_path)