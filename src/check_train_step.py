import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/src")

from dataset import PotholeDataset
from model import build_model

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

device = torch.device("cuda:0")

# Get one training sample.
dataset = PotholeDataset("train")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

images, masks = next(iter(loader))
images = images.to(device)
masks = masks.to(device).squeeze(1).long()

# Normalize inputs for the pretrained backbone.
mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)

images = (images - mean) / std

# Build the model and optimizer.
model = build_model().to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Forward pass.
optimizer.zero_grad(set_to_none=True)
logits = model(images)["out"]
loss = criterion(logits, masks)

# Backward pass.
loss.backward()

gradients_found = sum(
    parameter.grad is not None
    for parameter in model.parameters()
)

gradient_norm = sum(
    parameter.grad.detach().norm().item()
    for parameter in model.parameters()
    if parameter.grad is not None
)

optimizer.step()

print("GPU:", torch.cuda.get_device_name(0))
print("Input shape:", tuple(images.shape))
print("Target shape:", tuple(masks.shape))
print("Logits shape:", tuple(logits.shape))
print("Loss:", loss.item())
print("Parameters receiving gradients:", gradients_found)
print("Sum of gradient norms:", gradient_norm)

assert torch.isfinite(loss).item(), "Loss is not finite"
assert gradients_found > 0, "No gradients were calculated"
assert gradient_norm > 0, "Gradients are all zero"

print("Training-step check passed.")