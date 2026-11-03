FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN python -m pip install --break-system-packages --no-cache-dir segmentation-models-pytorch==0.5.0

CMD ["python", "-c", "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'NVIDIA GPU is required'; print('GPU:', torch.cuda.get_device_name(0))"]