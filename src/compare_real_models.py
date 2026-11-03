
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from model import build_model as build_lraspp_model
from model_unetpp import build_unetpp_model


ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "data" / "real_photos"
OUTPUT_DIR = ROOT / "outputs" / "real_model_comparison"

LRASPP_CHECKPOINT = ROOT / "outputs" / "lraspp_finetune_best.pth"
UNETPP_CHECKPOINT = ROOT / "outputs" / "unetpp_pothole_mix_best.pth"

# Same input dimensions used during training.
IMAGE_SIZE = (640, 384)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

# ImageNet normalization used by both models.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {path}")

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict missing in {path}")

    return checkpoint


def load_models(device):
    print("\nLoading LR-ASPP model...", flush=True)

    lraspp_checkpoint = load_checkpoint(
        LRASPP_CHECKPOINT, device
    )

    lraspp = build_lraspp_model().to(device)

    lraspp.load_state_dict(
        lraspp_checkpoint["model_state_dict"]
    )

    lraspp.eval()

    print(
        f"LR-ASPP checkpoint epoch: "
        f"{lraspp_checkpoint.get('epoch', 'unknown')}",
        flush=True
    )

    lraspp_threshold = float(
        lraspp_checkpoint.get("threshold", 0.50)
    )

    print("\nLoading U-Net++ model...", flush=True)

    unetpp_checkpoint = load_checkpoint(
        UNETPP_CHECKPOINT, device
    )

    unetpp = build_unetpp_model().to(device)

    unetpp.load_state_dict(
        unetpp_checkpoint["model_state_dict"]
    )

    unetpp.eval()

    print(
        f"U-Net++ checkpoint epoch: "
        f"{unetpp_checkpoint.get('epoch', 'unknown')}",
        flush=True
    )

    return (
        lraspp,
        unetpp,
        lraspp_threshold,
        0.50,
        lraspp_checkpoint,
        unetpp_checkpoint
    )


def prepare_image(path):
    """Resize exactly as in the training dataset pipeline."""

    with Image.open(path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")

    original_size = original.size

    resized = original.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(array)

    tensor = tensor.permute(2, 0, 1).contiguous()

    return original, original_size, tensor


@torch.inference_mode()
def predict(model, inputs, model_type, device, threshold):
    """Run one model and return its binary pothole prediction."""

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda")
    ):
        output = model(inputs)

    # Torchvision LR-ASPP returns a dictionary.
    # U-Net++ returns the segmentation tensor directly.
    if isinstance(output, dict):
        logits = output["out"]
    else:
        logits = output

    if logits.shape[1] != 2:
        raise ValueError(
            f"{model_type} returned {logits.shape[1]} classes; "
            "expected 2."
        )

    if logits.shape[-2:] != inputs.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    probability = torch.softmax(
        logits.float(),
        dim=1
    )[0, 1]

    prediction = (
        probability >= threshold
    ).to(torch.uint8).cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    return prediction, elapsed


def restore_mask(mask_array, original_size):
    mask = Image.fromarray(
        mask_array * 255,
        mode="L"
    )

    return mask.resize(
        original_size,
        Image.Resampling.NEAREST
    )


def create_overlay(original, mask, alpha=0.42):
    base = original.convert("RGBA")

    mask_array = np.asarray(mask.convert("L")) > 0

    overlay_array = np.zeros(
        (mask_array.shape[0], mask_array.shape[1], 4),
        dtype=np.uint8
    )

    overlay_array[mask_array] = (
        255, 35, 35, int(255 * alpha)
    )

    overlay = Image.fromarray(overlay_array)

    return Image.alpha_composite(
        base, overlay
    ).convert("RGB")


def create_comparison(original, lraspp_overlay, unetpp_overlay):
    width, height = original.size
    header_height = 45

    canvas = Image.new(
        "RGB",
        (width * 3, height + header_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    headers = [
        "Original photograph",
        "LR-ASPP + MobileNetV3",
        "U-Net++ + ResNet18"
    ]

    draw.text((10, 15), headers[0], fill="black")
    draw.text((width + 10, 15), headers[1], fill="black")
    draw.text((width * 2 + 10, 15), headers[2], fill="black")

    canvas.paste(original, (0, header_height))

    canvas.paste(
        lraspp_overlay,
        (width, header_height)
    )

    canvas.paste(
        unetpp_overlay,
        (width * 2, header_height)
    )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Compare LR-ASPP and U-Net++ on real road photographs."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional common threshold for both models."
    )

    args = parser.parse_args()

    if args.threshold is not None:
        if not 0 <= args.threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Photo folder not found: {INPUT_DIR}"
        )

    image_paths = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No photographs found in {INPUT_DIR}"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True
        )
    else:
        print("WARNING: Running on CPU.", flush=True)

    print(f"Photographs found: {len(image_paths)}")

    (
        lraspp,
        unetpp,
        lraspp_threshold,
        unetpp_threshold,
        lraspp_checkpoint,
        unetpp_checkpoint
    ) = load_models(device)

    if args.threshold is not None:
        lraspp_threshold = args.threshold
        unetpp_threshold = args.threshold

    print("\n========== COMPARISON SETTINGS ==========")

    print(f"Input dimensions: {IMAGE_SIZE}")
    print(f"LR-ASPP threshold: {lraspp_threshold:.2f}")
    print(f"U-Net++ threshold: {unetpp_threshold:.2f}")
    print(f"Photos: {len(image_paths)}")

    mean = torch.tensor(
        MEAN,
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        STD,
        device=device
    ).view(1, 3, 1, 1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    total_start = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"Processing {image_path.name}",
            flush=True
        )

        original, original_size, image_tensor = prepare_image(
            image_path
        )

        inputs = image_tensor.unsqueeze(0).to(
            device,
            non_blocking=True
        )

        inputs = (inputs - mean) / std

        # Predict using both models on the exact same input.
        lraspp_prediction, lraspp_time = predict(
            lraspp,
            inputs,
            "LR-ASPP",
            device,
            lraspp_threshold
        )

        unetpp_prediction, unetpp_time = predict(
            unetpp,
            inputs,
            "U-Net++",
            device,
            unetpp_threshold
        )

        # Restore predictions to the original photograph size.
        lraspp_mask = restore_mask(
            lraspp_prediction,
            original_size
        )

        unetpp_mask = restore_mask(
            unetpp_prediction,
            original_size
        )

        # Create visual overlays.
        lraspp_overlay = create_overlay(
            original,
            lraspp_mask
        )

        unetpp_overlay = create_overlay(
            original,
            unetpp_mask
        )

        comparison = create_comparison(
            original,
            lraspp_overlay,
            unetpp_overlay
        )

        stem = image_path.stem

        lraspp_mask.save(
            OUTPUT_DIR / f"{stem}_lraspp_mask.png"
        )

        unetpp_mask.save(
            OUTPUT_DIR / f"{stem}_unetpp_mask.png"
        )

        lraspp_overlay.save(
            OUTPUT_DIR / f"{stem}_lraspp_overlay.jpg",
            quality=92
        )

        unetpp_overlay.save(
            OUTPUT_DIR / f"{stem}_unetpp_overlay.jpg",
            quality=92
        )

        comparison.save(
            OUTPUT_DIR / f"{stem}_comparison.jpg",
            quality=92
        )

        lraspp_area = float(
            lraspp_prediction.mean() * 100
        )

        unetpp_area = float(
            unetpp_prediction.mean() * 100
        )

        rows.append({
            "image": image_path.name,
            "lraspp_threshold": lraspp_threshold,
            "unetpp_threshold": unetpp_threshold,
            "lraspp_predicted_area_percent": round(
                lraspp_area, 3
            ),
            "unetpp_predicted_area_percent": round(
                unetpp_area, 3
            ),
            "lraspp_inference_seconds": round(
                lraspp_time, 4
            ),
            "unetpp_inference_seconds": round(
                unetpp_time, 4
            )
        })

        print(
            f"LR-ASPP: {lraspp_area:.2f}% predicted area, "
            f"{lraspp_time:.3f}s",
            flush=True
        )

        print(
            f"U-Net++: {unetpp_area:.2f}% predicted area, "
            f"{unetpp_time:.3f}s",
            flush=True
        )

        print(
            f"Comparison saved: {stem}_comparison.jpg",
            flush=True
        )

    csv_path = OUTPUT_DIR / "comparison_summary.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.perf_counter() - total_start

    print("\n========== COMPARISON COMPLETE ==========")

    print(f"Photographs processed: {len(image_paths)}")

    print(
        f"Total elapsed: {total_elapsed / 60:.2f} minutes"
    )

    print(f"Results folder: {OUTPUT_DIR}")

    print(f"Summary CSV: {csv_path}")

    print(
        "\nOpen the *_comparison.jpg images "
        "to compare both models."
    )


if __name__ == "__main__":
    main()
