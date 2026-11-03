
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "pothole_mix" / "pothole-mix"

SPLIT_FOLDERS = {
    "train": "training",
    "validation": "validation",
    "test": "testing",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

BACKGROUND = (0, 0, 0)
POTHOLE = (255, 0, 0)
CRACK = (0, 255, 0)

ALLOWED_COLORS = {BACKGROUND, POTHOLE, CRACK}


class PotholeMixDataset(Dataset):
    """
    Binary segmentation dataset.

    Target 0: background and cracks
    Target 1: potholes

    Returns:
        image: float tensor [3, H, W], values in [0, 1]
        mask:  float tensor [1, H, W], values 0 or 1
    """

    def __init__(self, split, image_size=(640, 384)):
        if split not in SPLIT_FOLDERS:
            raise ValueError(
                f"Invalid split {split!r}. "
                f"Choose from {list(SPLIT_FOLDERS)}"
            )

        if not DATA_ROOT.is_dir():
            raise FileNotFoundError(
                f"Pothole Mix directory not found: {DATA_ROOT}"
            )

        self.split = split
        self.image_size = image_size  # (width, height)
        self.split_dir = DATA_ROOT / SPLIT_FOLDERS[split]
        self.samples = []

        for source_dir in sorted(
            path for path in self.split_dir.iterdir()
            if path.is_dir()
        ):
            image_dir = source_dir / "images"
            mask_dir = source_dir / "masks"

            if not image_dir.is_dir() or not mask_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing images/ or masks/ in {source_dir}"
                )

            images = sorted(
                path for path in image_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            masks = sorted(
                path for path in mask_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            mask_by_stem = {path.stem: path for path in masks}
            image_stems = {path.stem for path in images}

            if len(mask_by_stem) != len(masks):
                raise ValueError(
                    f"Duplicate mask filenames in {mask_dir}"
                )

            missing_masks = image_stems - set(mask_by_stem)
            missing_images = set(mask_by_stem) - image_stems

            if missing_masks or missing_images:
                raise ValueError(
                    f"Pairing error in {source_dir.name}: "
                    f"{len(missing_masks)} images lack masks, "
                    f"{len(missing_images)} masks lack images"
                )

            for image_path in images:
                self.samples.append(
                    (
                        image_path,
                        mask_by_stem[image_path.stem],
                        source_dir.name,
                    )
                )

        if not self.samples:
            raise ValueError(f"No samples found for split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, source = self.samples[index]

        with Image.open(image_path) as file:
            image = file.convert("RGB")

        with Image.open(mask_path) as file:
            mask = file.convert("RGB")

        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch: {image_path} "
                f"and {mask_path}"
            )

        mask_array = np.asarray(mask, dtype=np.uint8)

        observed_colors = set(
            map(
                tuple,
                np.unique(mask_array.reshape(-1, 3), axis=0),
            )
        )

        unexpected = observed_colors - ALLOWED_COLORS

        if unexpected:
            raise ValueError(
                f"Unexpected mask colors in {mask_path}: "
                f"{unexpected}"
            )

        # Only red pixels are positive pothole labels.
        # Green crack pixels and black background are class 0.
        pothole_pixels = np.all(
            mask_array == POTHOLE, axis=2
        )

        binary_mask = Image.fromarray(
            pothole_pixels.astype(np.uint8) * 255
        )

        image = image.resize(
            self.image_size, Image.Resampling.BILINEAR
        )
        binary_mask = binary_mask.resize(
            self.image_size, Image.Resampling.NEAREST
        )

        image_array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        mask_array = (
            np.asarray(binary_mask, dtype=np.uint8) > 0
        ).astype(np.float32)

        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1).contiguous()

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0).contiguous()

        return image_tensor, mask_tensor


if __name__ == "__main__":
    expected_counts = {
        "train": 3340,
        "validation": 496,
        "test": 504,
    }

    for split, expected in expected_counts.items():
        dataset = PotholeMixDataset(split)

        source_counts = Counter(
            source for _, _, source in dataset.samples
        )

        print(f"\n===== {split.upper()} =====")
        print("Samples:", len(dataset))
        print("Expected:", expected)
        print("Sources:", dict(source_counts))

        if len(dataset) != expected:
            raise RuntimeError(
                f"Expected {expected} {split} samples, "
                f"found {len(dataset)}"
            )

        # Check one sample from every source in each split.
        inspected_sources = set()

        for index, (image_path, mask_path, source) in enumerate(
            dataset.samples
        ):
            if source in inspected_sources:
                continue

            image, mask = dataset[index]
            inspected_sources.add(source)

            print(f"\nSource: {source}")
            print("Image:", image_path.name)
            print("Mask:", mask_path.name)
            print("Image tensor:", tuple(image.shape))
            print("Mask tensor:", tuple(mask.shape))
            print("Mask values:", torch.unique(mask).tolist())
            print(
                "Pothole pixel fraction:",
                round(mask.mean().item(), 5),
            )

        assert len(inspected_sources) == len(source_counts)
        print(f"\n{split} loader check passed.")

    print("\nAll Pothole Mix loader checks passed.")
