import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


def build_model():
    return lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=2,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU is required.")

    device = torch.device("cuda:0")
    model = build_model().to(device)
    model.eval()

    # Simulate one preprocessed RGB image.
    sample = torch.rand(1, 3, 384, 640, device=device)

    with torch.inference_mode():
        logits = model(sample)["out"]

    print("GPU:", torch.cuda.get_device_name(0))
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    assert logits.shape == (1, 2, 384, 640)
    print("Model shape check passed.")