import sys
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset
from model import build_model

# Experiment configuration
SEED = 42
EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
POTHOLE_WEIGHT = 8.0
THRESHOLD = 0.50

if not torch.cuda.is_available():
    raise RuntimeError("This project requires an NVIDIA GPU.")

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0")
output_dir = Path("/workspace/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

train_dataset = PotholeDataset("train")
val_dataset = PotholeDataset("validation")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = build_model().to(device)

# Experiment 1: freeze pretrained feature extraction.
for parameter in model.backbone.parameters():
    parameter.requires_grad = False

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

class_weights = torch.tensor(
    [1.0, POTHOLE_WEIGHT],
    dtype=torch.float32,
    device=device,
)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Normalization expected by the pretrained ImageNet backbone.
mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)


def prepare(images, masks):
    images = images.to(device)
    masks = masks.to(device).squeeze(1).long()
    images = (images - mean) / std
    return images, masks


def get_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def validate():
    model.eval()
    loss_sum = 0.0
    tp = fp = tn = fn = 0

    with torch.inference_mode():
        for images, masks in val_loader:
            images, targets = prepare(images, masks)
            logits = model(images)["out"]
            loss = criterion(logits, targets)
            loss_sum += loss.item()

            probabilities = logits.softmax(dim=1)[:, 1]
            predictions = probabilities >= THRESHOLD
            actual = targets == 1

            tp += int((predictions & actual).sum().item())
            fp += int((predictions & ~actual).sum().item())
            tn += int((~predictions & ~actual).sum().item())
            fn += int((~predictions & actual).sum().item())

    metrics = get_metrics(tp, fp, tn, fn)
    metrics["val_loss"] = loss_sum / len(val_loader)
    metrics["predicted_pothole_pixels"] = tp + fp
    metrics["actual_pothole_pixels"] = tp + fn

    return metrics


history = []
best_dice = -1.0
best_checkpoint = output_dir / "lraspp_frozen_best.pth"

for epoch in range(1, EPOCHS + 1):
    model.train()

    # Keep the frozen backbone's batch-normalization statistics fixed.
    model.backbone.eval()

    loss_sum = 0.0

    for images, masks in train_loader:
        images, targets = prepare(images, masks)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)["out"]
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    train_loss = loss_sum / len(train_loader)
    metrics = validate()

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

    # The final test dataset is never loaded or evaluated here.
    if metrics["dice"] > best_dice:
        best_dice = metrics["dice"]

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_metrics": metrics,
                "config": {
                    "seed": SEED,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "pothole_weight": POTHOLE_WEIGHT,
                    "threshold": THRESHOLD,
                    "backbone_frozen": True,
                },
            },
            best_checkpoint,
        )

history_file = output_dir / "training_history.csv"

with history_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

print("\nBest validation Dice:", best_dice)
print("Checkpoint:", best_checkpoint)
print("Training history:", history_file)