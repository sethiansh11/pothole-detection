import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

device = torch.device("cuda")

train_dataset = PotholeDataset("train")

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
)

images, masks = next(iter(train_loader))

print("Image batch shape:", tuple(images.shape))
print("Mask batch shape:", tuple(masks.shape))
print("Batch image range:", images.min().item(), images.max().item())
print("Mask values:", torch.unique(masks).tolist())

images = images.to(device)
masks = masks.to(device)

print("Image device:", images.device)
print("Mask device:", masks.device)

assert images.shape == (2, 3, 384, 640)
assert masks.shape == (2, 1, 384, 640)
assert set(torch.unique(masks).tolist()) <= {0.0, 1.0}

print("Batch validation passed.")