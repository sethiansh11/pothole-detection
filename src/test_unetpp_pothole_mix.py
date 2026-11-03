import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_unetpp import build_unetpp_model
from pothole_mix_dataset import PotholeMixDataset


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_CHECKPOINT = OUTPUT_DIR / "unetpp_pothole_mix_best.pth"

RESULTS_CSV = OUTPUT_DIR / "unetpp_pothole_mix_test_results.csv"
RESULTS_JSON = OUTPUT_DIR / "unetpp_pothole_mix_test_results.json"


@torch.inference_mode()
def evaluate(model, loader, device, mean, std, threshold, amp_enabled):

    model.eval()

    tp = fp = tn = fn = 0

    loss_sum = 0.0
    image_count = 0

    start_time = time.time()
    total_batches = len(loader)

    for batch_index, (images, masks) in enumerate(loader, start=1):

        batch_size = images.shape[0]

        images = images.to(device, non_blocking=True)

        targets = masks.to(
            device,
            non_blocking=True
        ).squeeze(1).long()

        images = (images - mean) / std

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled
        ):

            logits = model(images)

            batch_loss = torch.nn.functional.cross_entropy(
                logits,
                targets,
                weight=torch.tensor(
                    [1.0, 4.0],
                    device=device
                )
            )

        loss_sum += float(batch_loss.item()) * batch_size
        image_count += batch_size

        predictions = logits.softmax(dim=1)[:, 1] >= threshold
        actual = targets == 1

        tp += int((predictions & actual).sum().item())
        fp += int((predictions & ~actual).sum().item())
        tn += int((~predictions & ~actual).sum().item())
        fn += int((~predictions & actual).sum().item())

        if batch_index % 25 == 0 or batch_index == total_batches:

            elapsed = (time.time() - start_time) / 60

            print(
                f"Test batches: {batch_index}/{total_batches} "
                f"| images={image_count} "
                f"| elapsed={elapsed:.1f} min",
                flush=True
            )

    precision = tp / (tp + fp) if tp + fp else 0.0

    recall = tp / (tp + fn) if tp + fn else 0.0

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0.0
    )

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    return {
        "images": image_count,
        "threshold": threshold,
        "test_loss": loss_sum / max(image_count, 1),
        "dice": dice,
        "iou": iou,
        "recall": recall,
        "precision": precision,
        "pixel_accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "actual_pothole_pixels": tp + fn,
        "predicted_pothole_pixels": tp + fp,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate U-Net++ on the Pothole Mix test split."
    )

    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the best model checkpoint."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Run Docker with --gpus all."
        )

    device = torch.device("cuda:0")
    amp_enabled = True

    torch.backends.cudnn.benchmark = True

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}",
        flush=True
    )

    print(
        f"Checkpoint: {checkpoint_path}",
        flush=True
    )

    print(
        "Loading the test split. No training will be performed...",
        flush=True
    )

    test_dataset = PotholeMixDataset("test")

    print(
        f"Test images: {len(test_dataset)}",
        flush=True
    )

    loader_kwargs = {
        "dataset": test_dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }

    if args.num_workers > 0:

        loader_kwargs["persistent_workers"] = True

        loader_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(**loader_kwargs)

    # Build the same architecture used during training.
    model = build_unetpp_model().to(device)

    # Load the trained checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain model_state_dict. "
            "Use the Pothole Mix U-Net++ checkpoint."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    checkpoint_epoch = checkpoint.get("epoch", "unknown")

    validation_metrics = checkpoint.get(
        "validation_metrics",
        {}
    )

    print(
        f"Checkpoint epoch: {checkpoint_epoch}",
        flush=True
    )

    if validation_metrics:

        print(
            "Saved validation metrics: "
            f"Dice={validation_metrics.get('dice', 'n/a')} | "
            f"IoU={validation_metrics.get('iou', 'n/a')} | "
            f"Recall={validation_metrics.get('recall', 'n/a')} | "
            f"Precision={validation_metrics.get('precision', 'n/a')}",
            flush=True
        )

    # Same ImageNet normalization used during training.
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    ).view(1, 3, 1, 1)

    print(
        f"Batch size: {args.batch_size}; "
        f"workers: {args.num_workers}; "
        f"threshold: {args.threshold:.2f}; "
        f"mixed precision: {amp_enabled}",
        flush=True
    )

    # Evaluate the untouched test dataset.
    results = evaluate(
        model,
        test_loader,
        device,
        mean,
        std,
        args.threshold,
        amp_enabled
    )

    results["checkpoint"] = str(checkpoint_path)

    results["checkpoint_epoch"] = checkpoint_epoch

    results["validation_dice_at_checkpoint"] = (
        validation_metrics.get("dice")
    )

    # Save results.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with RESULTS_JSON.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            results,
            handle,
            indent=2
        )

    csv_fields = list(results.keys())

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields
        )

        writer.writeheader()

        writer.writerow(results)

    # Print final results.
    print("\n========== TEST RESULTS ==========", flush=True)

    print(
        f"Images evaluated: {results['images']}",
        flush=True
    )

    print(
        f"Test loss:       {results['test_loss']:.4f}",
        flush=True
    )

    print(
        f"Dice:            {results['dice']:.4f}",
        flush=True
    )

    print(
        f"IoU:             {results['iou']:.4f}",
        flush=True
    )

    print(
        f"Recall:          {results['recall']:.4f}",
        flush=True
    )

    print(
        f"Precision:       {results['precision']:.4f}",
        flush=True
    )

    print(
        f"Pixel accuracy:  {results['pixel_accuracy']:.4f}",
        flush=True
    )

    print(
        f"TP / FP / TN / FN: "
        f"{results['tp']} / {results['fp']} / "
        f"{results['tn']} / {results['fn']}",
        flush=True
    )

    print(
        f"Elapsed: {results['elapsed_seconds'] / 60:.1f} min",
        flush=True
    )

    print(
        f"\nJSON results: {RESULTS_JSON}",
        flush=True
    )

    print(
        f"CSV results: {RESULTS_CSV}",
        flush=True
    )


if __name__ == "__main__":
    main()