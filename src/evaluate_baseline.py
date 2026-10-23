import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

device = torch.device("cuda:0")

dataset = PotholeDataset("validation")
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

tp = fp = tn = fn = 0

for images, masks in loader:
    masks = masks.to(device) >= 0.5

    # Deliberately predict background for every pixel.
    predictions = torch.zeros_like(masks)

    tp += (predictions & masks).sum().item()
    fp += (predictions & ~masks).sum().item()
    tn += (~predictions & ~masks).sum().item()
    fn += (~predictions & masks).sum().item()

total = tp + fp + tn + fn

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

results = {
    "baseline": "all_background",
    "validation_images": len(dataset),
    "pixel_accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "iou": iou,
    "dice": dice,
    "tp": tp,
    "fp": fp,
    "tn": tn,
    "fn": fn,
}

output = Path("/workspace/outputs")
output.mkdir(parents=True, exist_ok=True)

with (output / "baseline_validation.json").open("w") as file:
    json.dump(results, file, indent=2)

for key, value in results.items():
    print(f"{key}: {value}")

print("Saved baseline results.")