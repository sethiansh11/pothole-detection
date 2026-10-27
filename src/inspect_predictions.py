import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset
from model import build_model

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

device = torch.device("cuda:0")
root = Path("/workspace")
output_dir = root / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = output_dir / "lraspp_frozen_best.pth"
checkpoint = torch.load(
    checkpoint_path, map_location=device, weights_only=True
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

thumb_w, thumb_h = 320, 192
row_h = thumb_h + 42
headers = ["RGB image", "Ground truth", "Prediction @ 0.50", "Prediction overlay"]

sheet = Image.new("RGB", (thumb_w * 4, 40 + row_h * len(dataset)), "white")
draw = ImageDraw.Draw(sheet)

for col, header in enumerate(headers):
    draw.text((col * thumb_w + 8, 10), header, fill="black")

total_tp = total_fp = total_fn = 0

for i in range(len(dataset)):
    image, mask = dataset[i]
    sample = dataset.samples[i]

    inputs = image.unsqueeze(0).to(device)
    targets = mask.squeeze(0).numpy().astype(bool)

    inputs = (inputs - mean) / std

    with torch.inference_mode():
        logits = model(inputs)["out"]
        probabilities = logits.softmax(dim=1)[0, 1]
        predictions = (probabilities >= 0.50).cpu().numpy()

    tp = int(np.logical_and(predictions, targets).sum())
    fp = int(np.logical_and(predictions, ~targets).sum())
    fn = int(np.logical_and(~predictions, targets).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    total_tp += tp
    total_fp += fp
    total_fn += fn

    print(
        f"{sample['dataset']}/{sample['frame_id']} | "
        f"precision={precision:.3f} recall={recall:.3f} "
        f"dice={dice:.3f} | "
        f"predicted_pixels={int(predictions.sum())} "
        f"actual_pixels={int(targets.sum())}"
    )

    rgb = np.clip(
        image.permute(1, 2, 0).numpy() * 255, 0, 255
    ).astype(np.uint8)

    gt_image = np.zeros_like(rgb)
    gt_image[targets] = (153, 0, 0)

    pred_image = np.zeros_like(rgb)
    pred_image[predictions] = (0, 220, 0)

    overlay = rgb.copy()
    overlay[predictions] = (
        0.55 * overlay[predictions] + 0.45 * np.array([255, 40, 40])
    ).astype(np.uint8)

    y = 40 + i * row_h
    draw.text(
        (8, y),
        f"{sample['dataset']} / frame {sample['frame_id']}",
        fill="black",
    )

    visuals = [
        Image.fromarray(rgb),
        Image.fromarray(gt_image),
        Image.fromarray(pred_image),
        Image.fromarray(overlay),
    ]

    for col, visual in enumerate(visuals):
        visual = visual.resize(
            (thumb_w, thumb_h), Image.Resampling.NEAREST
        )
        sheet.paste(visual, (col * thumb_w, y + 20))

overall_precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp else 0.0
)
overall_recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn else 0.0
)
overall_dice = (
    2 * total_tp / (2 * total_tp + total_fp + total_fn)
    if 2 * total_tp + total_fp + total_fn else 0.0
)

sheet_path = output_dir / "validation_predictions.png"
sheet.save(sheet_path)

print(f"\nOverall precision: {overall_precision:.4f}")
print(f"Overall recall:    {overall_recall:.4f}")
print(f"Overall Dice:      {overall_dice:.4f}")
print(f"Saved visualization to: {sheet_path}")