import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/workspace/src")
from dataset import PotholeDataset
from model import build_model

if not torch.cuda.is_available():
    raise RuntimeError("NVIDIA GPU is required")

device = torch.device("cuda:0")
root = Path("/workspace")
output_dir = root / "outputs"

checkpoint = torch.load(
    output_dir / "lraspp_frozen_best.pth",
    map_location=device,
    weights_only=True,
)

model = build_model().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dataset = PotholeDataset("validation")
thresholds = [0.40, 0.50, 0.80]

mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)

thumb_w, thumb_h = 320, 192
headers = [
    "RGB image",
    "Ground truth",
    "Errors @ 0.40",
    "Errors @ 0.50",
    "Errors @ 0.80",
]

top_h = 40
row_h = thumb_h + 30

sheet = Image.new(
    "RGB",
    (thumb_w * len(headers), top_h + row_h * len(dataset)),
    "white",
)
draw = ImageDraw.Draw(sheet)

for col, header in enumerate(headers):
    draw.text((col * thumb_w + 8, 10), header, fill="black")

with torch.inference_mode():
    for i in range(len(dataset)):
        image, mask = dataset[i]
        sample = dataset.samples[i]

        rgb = np.clip(
            image.permute(1, 2, 0).numpy() * 255,
            0, 255,
        ).astype(np.uint8)

        targets = mask.squeeze(0).numpy().astype(bool)

        inputs = image.unsqueeze(0).to(device)
        inputs = (inputs - mean) / std

        logits = model(inputs)["out"]
        probabilities = logits.softmax(dim=1)[0, 1].cpu().numpy()

        # Ground-truth pothole pixels appear red.
        ground_truth = np.zeros_like(rgb)
        ground_truth[targets] = (220, 40, 40)

        visuals = [
            Image.fromarray(rgb),
            Image.fromarray(ground_truth),
        ]

        for threshold in thresholds:
            predictions = probabilities >= threshold

            tp = predictions & targets
            fp = predictions & ~targets
            fn = ~predictions & targets

            # Green = correctly detected.
            # Red = predicted pothole, absent from label.
            # Blue = labelled pothole, missed by prediction.
            colours = np.zeros_like(rgb)
            colours[tp] = (0, 220, 0)
            colours[fp] = (255, 0, 0)
            colours[fn] = (0, 80, 255)

            active = tp | fp | fn

            overlay = rgb.copy()
            overlay[active] = (
                0.45 * rgb[active] + 0.55 * colours[active]
            ).astype(np.uint8)

            visuals.append(Image.fromarray(overlay))

        y = top_h + i * row_h

        draw.text(
            (8, y),
            f"{sample['dataset']} / frame {sample['frame_id']}",
            fill="black",
        )

        for col, visual in enumerate(visuals):
            visual = visual.resize(
                (thumb_w, thumb_h),
                Image.Resampling.NEAREST,
            )
            sheet.paste(visual, (col * thumb_w, y + 20))

output_path = output_dir / "threshold_error_maps.png"
sheet.save(output_path)

print("Saved:", output_path)
print("Rows:", len(dataset))
print("Thresholds compared:", thresholds)