import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from model_unetpp import build_unetpp_model


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = ROOT / "data" / "real_photos"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "real_photo_predictions"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "unetpp_pothole_mix_best.pth"

# Same input dimensions used during training.
IMAGE_SIZE = (640, 384)  # width, height

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def load_image(path: Path):
    """Load an image, correct its orientation and resize it."""

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    original_size = image.size

    resized = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(array)

    tensor = tensor.permute(2, 0, 1).contiguous()

    return image, original_size, tensor


def make_overlay(
    original: Image.Image,
    mask: Image.Image,
    alpha: float = 0.42
):
    """Overlay predicted pothole pixels in red."""

    base = original.convert("RGBA")

    mask_array = np.asarray(mask.convert("L")) > 0

    overlay_array = np.zeros(
        (mask_array.shape[0], mask_array.shape[1], 4),
        dtype=np.uint8
    )

    overlay_array[mask_array] = (
        255,
        40,
        40,
        int(255 * alpha)
    )

    overlay = Image.fromarray(
        overlay_array,
        mode="RGBA"
    )

    return Image.alpha_composite(
        base,
        overlay
    ).convert("RGB")


def make_comparison(
    original: Image.Image,
    overlay: Image.Image
):
    """Create a side-by-side comparison."""

    width, height = original.size

    header_height = 34

    canvas = Image.new(
        "RGB",
        (width * 2, height + header_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 10),
        "Original",
        fill="black"
    )

    draw.text(
        (width + 10, 10),
        "Predicted pothole pixels (red)",
        fill="black"
    )

    canvas.paste(original, (0, header_height))

    canvas.paste(overlay, (width, header_height))

    return canvas


@torch.inference_mode()
def predict_one(
    model,
    image_tensor,
    device,
    mean,
    std,
    threshold,
    amp_enabled
):
    """Run inference on one image."""

    inputs = image_tensor.unsqueeze(0).to(
        device,
        non_blocking=True
    )

    inputs = (inputs - mean) / std

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp_enabled
    ):

        logits = model(inputs)

        pothole_probability = torch.softmax(
            logits,
            dim=1
        )[0, 1]

    probability = pothole_probability.float().cpu().numpy()

    mask = (
        probability >= threshold
    ).astype(np.uint8) * 255

    predicted_fraction = float(
        (mask > 0).mean()
    )

    return Image.fromarray(mask, mode="L"), predicted_fraction


def main():

    parser = argparse.ArgumentParser(
        description="Run trained U-Net++ on real road photographs."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR)
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT)
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1"
        )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    checkpoint_path = Path(args.checkpoint)

    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}\n"
            "Create data/real_photos and put road photos inside it."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {input_dir}"
        )

    # Select GPU when available.
    use_cuda = torch.cuda.is_available()

    device = torch.device(
        "cuda:0" if use_cuda else "cpu"
    )

    amp_enabled = use_cuda

    if use_cuda:

        torch.backends.cudnn.benchmark = True

        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True
        )

    else:

        print(
            "CUDA unavailable. Running on CPU.",
            flush=True
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input folder: {input_dir}")
    print(f"Photos found: {len(image_paths)}")
    print(f"Threshold: {args.threshold:.2f}")

    # Build the same architecture used during training.
    model = build_unetpp_model().to(device)

    # Load trained model weights.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch: "
        f"{checkpoint.get('epoch', 'unknown')}",
        flush=True
    )

    # ImageNet normalization, same as training.
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device
    ).view(1, 3, 1, 1)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_rows = []

    total_started = time.time()

    # Process each real-world photograph.
    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        started = time.time()

        original, original_size, image_tensor = load_image(
            image_path
        )

        mask_small, predicted_fraction = predict_one(
            model,
            image_tensor,
            device,
            mean,
            std,
            args.threshold,
            amp_enabled
        )

        # Restore mask to original photo dimensions.
        mask = mask_small.resize(
            original_size,
            Image.Resampling.NEAREST
        )

        overlay = make_overlay(
            original,
            mask
        )

        comparison = make_comparison(
            original,
            overlay
        )

        stem = image_path.stem

        mask_path = output_dir / f"{stem}_mask.png"

        overlay_path = output_dir / f"{stem}_overlay.jpg"

        comparison_path = output_dir / f"{stem}_comparison.jpg"

        mask.save(mask_path)

        overlay.save(
            overlay_path,
            quality=92
        )

        comparison.save(
            comparison_path,
            quality=92
        )

        elapsed = time.time() - started

        row = {
            "image": image_path.name,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "threshold": args.threshold,
            "predicted_pothole_area_percent": round(
                predicted_fraction * 100,
                3
            ),
            "inference_seconds": round(elapsed, 3),
            "mask_file": mask_path.name,
            "overlay_file": overlay_path.name,
            "comparison_file": comparison_path.name
        }

        summary_rows.append(row)

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"predicted area={predicted_fraction * 100:.2f}% | "
            f"time={elapsed:.2f}s",
            flush=True
        )

    # Save CSV summary.
    summary_path = output_dir / "prediction_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(summary_rows)

    print("\n========== REAL-PHOTO INFERENCE COMPLETE ==========")

    print(f"Photos processed: {len(image_paths)}")

    print(
        f"Total elapsed: {time.time() - total_started:.1f} seconds"
    )

    print(f"Results folder: {output_dir}")

    print(f"Summary CSV: {summary_path}")

    print(
        "Open the *_comparison.jpg files to inspect predictions."
    )


if __name__ == "__main__":
    main()