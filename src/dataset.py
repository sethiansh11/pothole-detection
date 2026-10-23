import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PotholeDataset(Dataset):
    def __init__(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split: {split}")

        self.data_root = Path("/workspace/data/stereo_pothole_datasets")
        manifest = Path("/workspace/splits/split_assignments.csv")
        self.image_size = (640, 384)

        with manifest.open(newline="", encoding="utf-8") as file:
            self.samples = [
                row for row in csv.DictReader(file)
                if row["split"] == split
            ]

        if not self.samples:
            raise ValueError(f"No samples found for split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = self.data_root / sample["rgb_path"]
        mask_path = self.data_root / sample["label_path"]

        with Image.open(image_path) as file:
            image = file.convert("RGB")

        with Image.open(mask_path) as file:
            mask = file.convert("RGB")

        if image.size != mask.size:
            raise ValueError(
                f"Image-mask dimensions differ: {image_path}"
            )

        mask_array = np.asarray(mask)

        allowed = {(0, 0, 0), (153, 0, 0)}
        observed = set(
            map(
                tuple,
                np.unique(mask_array.reshape(-1, 3), axis=0)
            )
        )

        if not observed.issubset(allowed):
            raise ValueError(f"Unexpected mask colours: {observed - allowed}")

        image = image.resize(
            self.image_size, Image.Resampling.BILINEAR
        )
        mask = mask.resize(
            self.image_size, Image.Resampling.NEAREST
        )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask)

        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
        )

        mask_array = np.all(
            mask_array == (153, 0, 0), axis=2
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return image_tensor, mask_tensor