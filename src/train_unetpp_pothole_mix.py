import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# The script is in src/, so its directory is already on Python's import path.
from model_unetpp import build_unetpp_model
from pothole_mix_dataset import PotholeMixDataset

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_BEST = OUTPUT_DIR / "unetpp_pothole_mix_best.pth"
CHECKPOINT_LAST = OUTPUT_DIR / "unetpp_pothole_mix_last.pth"
HISTORY_CSV = OUTPUT_DIR / "unetpp_pothole_mix_history.csv"

SEED = 42
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 2
ENCODER_LR = 1e-5
DECODER_LR = 3e-4
POTHOLE_WEIGHT = 4.0
THRESHOLD = 0.50


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, device):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    # Persistent workers avoid recreating workers every epoch.
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def prepare(images, masks, device, mean, std):
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True).squeeze(1).long()
    images = (images - mean) / std
    return images, masks


def calculate_metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"dice": dice, "iou": iou, "recall": recall, "precision": precision}


def evaluate(model, loader, criterion, device, mean, std, threshold, amp_enabled):
    model.eval()
    loss_sum = 0.0
    batch_count = 0
    tp = fp = fn = 0

    with torch.inference_mode():
        for batch_index, (images, masks) in enumerate(loader, start=1):
            images, targets = prepare(images, masks, device, mean, std)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            loss_sum += float(loss.item())
            batch_count += 1
            predictions = logits.softmax(dim=1)[:, 1] >= threshold
            actual = targets == 1
            tp += int((predictions & actual).sum().item())
            fp += int((predictions & ~actual).sum().item())
            fn += int((~predictions & actual).sum().item())

            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f"  Validation batches: {batch_index}/{len(loader)}", flush=True)

    metrics = calculate_metrics(tp, fp, fn)
    metrics["val_loss"] = loss_sum / max(batch_count, 1)
    return metrics


def save_checkpoint(path, model, optimizer, scaler, epoch, best_dice, metrics, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "architecture": "UnetPlusPlus",
            "encoder": "resnet18",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_dice": best_dice,
            "validation_metrics": metrics,
            "history": history,
            "config": {
                "batch_size": BATCH_SIZE_FOR_CHECKPOINT,
                "encoder_lr": ENCODER_LR,
                "decoder_lr": DECODER_LR,
                "pothole_weight": POTHOLE_WEIGHT,
                "threshold": THRESHOLD,
            },
        },
        path,
    )


def write_history(history):
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Preserve columns from older checkpoint/history formats as well as new ones.
    fields = [
        "epoch", "steps", "train_loss", "val_loss", "dice", "iou",
        "recall", "precision", "elapsed_seconds", "peak_gpu_mib"
    ]
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)

    with HISTORY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net++ on Pothole Mix.")
    parser.add_argument("--epochs", type=int, default=15, help="Total target epoch number.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Limit training batches per epoch for a quick test; 0 means all.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. Without a path, uses last checkpoint if present, else best.")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    # Used only to record the actual batch size in checkpoints.
    global BATCH_SIZE_FOR_CHECKPOINT
    BATCH_SIZE_FOR_CHECKPOINT = args.batch_size

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Run this script in Docker with --gpus all.")

    device = torch.device("cuda:0")
    amp_enabled = device.type == "cuda"
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    train_dataset = PotholeMixDataset("train")
    val_dataset = PotholeMixDataset("validation")
    print(f"Training images: {len(train_dataset)}", flush=True)
    print(f"Validation images: {len(val_dataset)}", flush=True)

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, True, device)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, False, device)

    model = build_unetpp_model().to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)
    print(f"Batch size: {args.batch_size}; DataLoader workers: {args.num_workers}", flush=True)
    print(f"Encoder learning rate: {ENCODER_LR}; decoder/head learning rate: {DECODER_LR}", flush=True)
    print(f"Pothole class weight: {POTHOLE_WEIGHT}; mixed precision: {amp_enabled}", flush=True)
    print(f"Maximum batches per epoch: {args.max_train_steps or 'all'}", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": model.decoder.parameters(), "lr": DECODER_LR},
            {"params": model.segmentation_head.parameters(), "lr": DECODER_LR},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, POTHOLE_WEIGHT], device=device)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    start_epoch = 1
    best_dice = -1.0
    history = []
    if args.resume is not None:
        if args.resume == "auto":
            resume_path = CHECKPOINT_LAST if CHECKPOINT_LAST.exists() else CHECKPOINT_BEST
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = ROOT / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))
        if best_dice < 0:
            best_dice = float(checkpoint.get("validation_metrics", {}).get("dice", -1.0))
        history = checkpoint.get("history", [])
        # Older checkpoints may not contain the run history; preserve the existing CSV.
        if not history and HISTORY_CSV.exists():
            with HISTORY_CSV.open("r", newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        print(f"Resumed from: {resume_path}", flush=True)
        print(f"Next epoch: {start_epoch}; best Dice so far: {best_dice:.4f}", flush=True)

    if start_epoch > args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; --epochs {args.epochs} has nothing more to run.", flush=True)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        # Keep pretrained encoder BatchNorm running statistics stable.
        model.encoder.eval()
        train_loss_sum = 0.0
        steps = 0
        total_steps = min(len(train_loader), args.max_train_steps) if args.max_train_steps else len(train_loader)

        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            if args.max_train_steps and batch_index > args.max_train_steps:
                break

            images, targets = prepare(images, masks, device, mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item())
            steps += 1

            if batch_index % 100 == 0 or batch_index == total_steps:
                elapsed = time.time() - epoch_started
                print(
                    f"  Epoch {epoch:02d} train batch {batch_index}/{total_steps} "
                    f"| loss={train_loss_sum / steps:.4f} | elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )

        train_loss = train_loss_sum / max(steps, 1)
        metrics = evaluate(model, val_loader, criterion, device, mean, std, THRESHOLD, amp_enabled)
        elapsed_seconds = time.time() - epoch_started
        peak_gpu_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        row = {
            "epoch": epoch,
            "steps": steps,
            "train_loss": round(train_loss, 6),
            "val_loss": round(metrics["val_loss"], 6),
            "dice": round(metrics["dice"], 6),
            "iou": round(metrics["iou"], 6),
            "recall": round(metrics["recall"], 6),
            "precision": round(metrics["precision"], 6),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "peak_gpu_mib": round(peak_gpu_mib, 1),
        }
        history.append(row)

        # Save model state before writing the CSV. If history export fails,
        # the completed epoch is still safely resumable from the last checkpoint.
        improved = metrics["dice"] > best_dice
        if improved:
            best_dice = metrics["dice"]
            save_checkpoint(CHECKPOINT_BEST, model, optimizer, scaler, epoch, best_dice, metrics, history)
            print(f"  New best checkpoint saved: {CHECKPOINT_BEST}", flush=True)

        save_checkpoint(CHECKPOINT_LAST, model, optimizer, scaler, epoch, best_dice, metrics, history)
        write_history(history)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | steps={steps} "
            f"| train_loss={train_loss:.4f} | val_loss={metrics['val_loss']:.4f} "
            f"| Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f} "
            f"| Recall={metrics['recall']:.4f} | Precision={metrics['precision']:.4f} "
            f"| peak_GPU_MiB={peak_gpu_mib:.1f} | elapsed={elapsed_seconds / 60:.1f} min",
            flush=True,
        )


    print(f"\nBest validation Dice: {best_dice:.4f}", flush=True)
    print(f"Best checkpoint: {CHECKPOINT_BEST}", flush=True)
    print(f"Latest checkpoint (resume from this): {CHECKPOINT_LAST}", flush=True)
    print(f"Training history: {HISTORY_CSV}", flush=True)


if __name__ == "__main__":
    main()
