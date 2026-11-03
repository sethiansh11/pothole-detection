
import argparse
import csv
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PotholeDataset
from model_unetpp import build_unetpp_model


ROOT = Path("/workspace")
OUTPUT_DIR = ROOT / "outputs"

SEED = 42
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
POTHOLE_WEIGHT = 8.0
THRESHOLD = 0.50


def get_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn

    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "dice": (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn else 0.0
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Limit training batches per epoch; 0 means all batches.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.max_train_steps < 0:
        raise ValueError("Invalid epoch or training-step count.")

    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA GPU is required.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Build U-Net++ with an ImageNet-pretrained ResNet18 encoder.
    model = build_unetpp_model().to(device)

    # First experiment: freeze the encoder and train the decoder/head.
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    class_weights = torch.tensor(
        [1.0, POTHOLE_WEIGHT],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)

    def prepare(images, masks):
        images = images.to(device)
        targets = masks.to(device).squeeze(1).long()
        return (images - mean) / std, targets

    def validate():
        model.eval()
        loss_sum = 0.0
        tp = fp = tn = fn = 0

        with torch.inference_mode():
            for images, masks in val_loader:
                images, targets = prepare(images, masks)

                # Mixed precision reduces GPU memory use.
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    logits = model(images)
                    loss = criterion(logits, targets)

                loss_sum += loss.item()

                probabilities = logits.float().softmax(dim=1)[:, 1]
                predictions = probabilities >= THRESHOLD
                actual = targets == 1

                tp += int((predictions & actual).sum().item())
                fp += int((predictions & ~actual).sum().item())
                tn += int((~predictions & ~actual).sum().item())
                fn += int((~predictions & actual).sum().item())

        metrics = get_metrics(tp, fp, tn, fn)
        metrics["val_loss"] = loss_sum / len(val_loader)
        return metrics

    scaler = torch.amp.GradScaler("cuda")

    checkpoint_path = OUTPUT_DIR / "unetpp_resnet18_frozen_best.pth"
    history_path = OUTPUT_DIR / "unetpp_resnet18_history.csv"

    history = []
    best_dice = -1.0

    print("GPU:", torch.cuda.get_device_name(0))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Trainable parameters:", sum(
        p.numel() for p in trainable_parameters
    ))
    print("AMP: enabled")
    print("Frozen encoder: ResNet18")
    print("Maximum batches per epoch:", args.max_train_steps or "all")

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Do not update the frozen encoder's BatchNorm statistics.
        model.encoder.eval()

        train_loss_sum = 0.0
        steps = 0

        torch.cuda.reset_peak_memory_stats(device)

        for images, masks in train_loader:
            if args.max_train_steps and steps >= args.max_train_steps:
                break

            images, targets = prepare(images, masks)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            steps += 1

        if steps == 0:
            raise RuntimeError("No training batches were processed.")

        train_loss = train_loss_sum / steps
        metrics = validate()

        record = {
            "epoch": epoch,
            "train_steps": steps,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(record)

        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"steps={steps} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"Dice={metrics['dice']:.4f} | "
            f"IoU={metrics['iou']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"peak_GPU_MiB={peak_memory_mib:.1f}"
        )

        # Keep the best validation-Dice checkpoint for this experiment.
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": "UnetPlusPlus",
                    "encoder": "resnet18",
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": metrics,
                    "config": {
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "pothole_weight": POTHOLE_WEIGHT,
                        "threshold": THRESHOLD,
                        "encoder_frozen": True,
                        "mixed_precision": True,
                    },
                },
                checkpoint_path,
            )

        with history_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

    print("\nBest validation Dice:", best_dice)
    print("Checkpoint:", checkpoint_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()
