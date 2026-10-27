import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset
from model import build_model

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

ROOT = Path("/workspace")
OUTPUT = ROOT / "outputs"
DEVICE = torch.device("cuda:0")

EPOCHS = 8
BATCH_SIZE = 1
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
POTHOLE_WEIGHT = 8.0
THRESHOLD = 0.50

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load the best model from the frozen-backbone experiment.
previous = torch.load(
    OUTPUT / "lraspp_frozen_best.pth",
    map_location="cpu",
    weights_only=True,
)

model = build_model()
model.load_state_dict(previous["model_state_dict"])
model = model.to(DEVICE)

# Freeze every parameter initially.
for parameter in model.parameters():
    parameter.requires_grad = False

# Train the segmentation head.
for parameter in model.classifier.parameters():
    parameter.requires_grad = True

# Also fine-tune the final MobileNetV3 feature blocks.
unfrozen_layers = set()

for name, parameter in model.backbone.named_parameters():
    layer_index = int(name.split(".", 1)[0])

    if layer_index >= 12:
        parameter.requires_grad = True
        unfrozen_layers.add(layer_index)

if not unfrozen_layers:
    raise RuntimeError("No backbone layers were unfrozen")

print("Fine-tuned backbone layers:", sorted(unfrozen_layers))

backbone_parameters = [
    p for p in model.backbone.parameters() if p.requires_grad
]
head_parameters = [
    p for p in model.classifier.parameters() if p.requires_grad
]

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_parameters, "lr": BACKBONE_LR},
        {"params": head_parameters, "lr": HEAD_LR},
    ],
    weight_decay=1e-4,
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        [1.0, POTHOLE_WEIGHT], device=DEVICE
    )
)

train_loader = DataLoader(
    PotholeDataset("train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    PotholeDataset("validation"),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

mean = torch.tensor(
    [0.485, 0.456, 0.406], device=DEVICE
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=DEVICE
).view(1, 3, 1, 1)


def prepare(images, masks):
    images = images.to(DEVICE)
    masks = masks.to(DEVICE).squeeze(1).long()
    return (images - mean) / std, masks


def evaluate():
    model.eval()
    loss_sum = 0.0
    tp = fp = tn = fn = 0

    with torch.inference_mode():
        for images, masks in val_loader:
            images, targets = prepare(images, masks)

            logits = model(images)["out"]
            loss_sum += criterion(logits, targets).item()

            predictions = logits.softmax(dim=1)[:, 1] >= THRESHOLD
            actual = targets == 1

            tp += int((predictions & actual).sum())
            fp += int((predictions & ~actual).sum())
            tn += int((~predictions & ~actual).sum())
            fn += int((~predictions & actual).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )

    return {
        "val_loss": loss_sum / len(val_loader),
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
    }


history = []
best_dice = -1.0
checkpoint_path = OUTPUT / "lraspp_finetune_best.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()

    # Keep BatchNorm statistics in the pretrained backbone fixed.
    model.backbone.eval()

    train_loss = 0.0

    for images, masks in train_loader:
        images, targets = prepare(images, masks)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)["out"]
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    metrics = evaluate()

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        **metrics,
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={metrics['val_loss']:.4f} | "
        f"Dice={metrics['dice']:.4f} | "
        f"IoU={metrics['iou']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f}"
    )

    if metrics["dice"] > best_dice:
        best_dice = metrics["dice"]

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_metrics": metrics,
                "threshold": THRESHOLD,
                "backbone_lr": BACKBONE_LR,
                "head_lr": HEAD_LR,
                "pothole_weight": POTHOLE_WEIGHT,
                "fine_tuned_layers": sorted(unfrozen_layers),
            },
            checkpoint_path,
        )

with (OUTPUT / "finetune_history.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

print("\nBest validation Dice:", best_dice)
print("Saved checkpoint:", checkpoint_path)
print("Saved history: outputs/finetune_history.csv")