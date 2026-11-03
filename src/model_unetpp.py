
import torch
import segmentation_models_pytorch as smp


def build_unetpp_model(encoder_weights="imagenet"):
    """
    U-Net++ for two-class pothole segmentation:
    class 0 = background
    class 1 = pothole
    """
    return smp.UnetPlusPlus(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
        decoder_channels=(128, 64, 32, 16, 8),
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU is required.")

    device = torch.device("cuda:0")
    model = build_unetpp_model().to(device).eval()

    sample = torch.rand(1, 3, 384, 640, device=device)

    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        logits = model(sample)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print(
        "Peak GPU memory (MiB):",
        round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1),
    )

    assert logits.shape == (1, 2, 384, 640)
    print("U-Net++ model check passed.")
