import sys
import csv
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset
from model import build_model

# Require NVIDIA GPU.
if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required.")

device = torch.device("cuda:0")
root = Path("/workspace")
output_dir = root / "outputs"
checkpoint_path = output_dir / "lraspp_finetune_best.pth"
# Load the best checkpoint selected during training.
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

model = build_model().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dataset = PotholeDataset("validation")

# ImageNet normalization used by the pretrained backbone.
mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Store predictions on CPU so GPU memory stays low.
all_probabilities = []
all_targets = []

with torch.inference_mode():
    for index in range(len(dataset)):
        image, mask = dataset[index]

        image = image.unsqueeze(0).to(device)
        image = (image - mean) / std

        logits = model(image)["out"]

        # Convert class logits into pothole probabilities.
        probabilities = logits.softmax(dim=1)[0, 1].cpu()

        targets = mask.squeeze(0).bool()

        all_probabilities.append(probabilities)
        all_targets.append(targets)

rows = []

for threshold in thresholds:
    tp = fp = fn = tn = 0
    predicted_positive = 0
    actual_positive = 0

    for probabilities, targets in zip(
        all_probabilities, all_targets
    ):
        predictions = probabilities >= threshold

        tp += int((predictions & targets).sum())
        fp += int((predictions & ~targets).sum())
        fn += int((~predictions & targets).sum())
        tn += int((~predictions & ~targets).sum())

        predicted_positive += int(predictions.sum())
        actual_positive += int(targets.sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )

    total_pixels = tp + fp + fn + tn

    rows.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "predicted_pothole_percent": (
            100 * predicted_positive / total_pixels
        ),
        "actual_pothole_percent": (
            100 * actual_positive / total_pixels
        ),
    })

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Validation images: {len(dataset)}")
print()
print(
    f"{'Threshold':>9} {'Precision':>10} {'Recall':>10} "
    f"{'Dice':>8} {'IoU':>8} {'Predicted %':>12}"
)

for row in rows:
    print(
        f"{row['threshold']:>9.2f} "
        f"{row['precision']:>10.4f} "
        f"{row['recall']:>10.4f} "
        f"{row['dice']:>8.4f} "
        f"{row['iou']:>8.4f} "
        f"{row['predicted_pothole_percent']:>11.2f}%"
    )

output_path = output_dir / "finetune_threshold_metrics.csv"
with output_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nResults saved to: {output_path}")