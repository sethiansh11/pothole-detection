# Pothole Detection + Geo-Severity — Colleague Setup Guide

This guide explains how to access the trained U-Net++ pothole segmentation model and the separate Geo-Location Severity Engine.

> **Current project status:** The segmentation model and Geo-Location Severity Engine are separate modules. The segmentation script predicts pothole pixels from an image. The Geo-Location Severity Engine takes GPS coordinates and estimates the geographic importance of that location. They are not yet connected into one automatic end-to-end application, and the final Priority Assignment Engine has not been implemented.

## 1. What is included

| Component | Location | Purpose |
|---|---|---|
| U-Net++ model architecture | `src/model_unetpp.py` | Defines the neural network architecture |
| Trained PotholeMix weights | `models/unetpp_potholemix_best.pth` | The trained model checkpoint used with that architecture |
| Image inference script | `src/predict_real_photos.py` | Runs segmentation inference on road images |
| Dataset loader | `src/pothole_mix_dataset.py` | Dataset loading/preprocessing code, mainly needed for training/evaluation |
| Training script | `src/train_unetpp_pothole_mix.py` | Training code; not required just to run inference |
| Geo-Location Severity Engine | `src/geo_severity.py` | Retrieves geographic context and calculates a provisional geographic score from GPS coordinates |
| Docker environment | `Dockerfile` | Defines the Python/PyTorch/CUDA environment |

The PotholeMix dataset itself is not included in the Git repository. It is not needed to run inference with the already-trained checkpoint. It is needed if you intend to retrain or reproduce training/evaluation.

## 2. Requirements on your computer

1. Windows 10/11 or Linux.
2. Git installed.
3. Git LFS installed (required to download the model checkpoint).
4. Docker Desktop installed and running. On Windows, use the WSL 2 backend.
5. For GPU execution: an NVIDIA GPU, a compatible NVIDIA driver, and Docker GPU support configured.

The Docker image uses the project's CUDA-enabled PyTorch environment. Docker does not provide a GPU or host NVIDIA driver; those must be available on the computer running Docker.

## 3. Get access to the repository

The repository owner must first create a **private, empty GitHub repository**, add you as a collaborator, and push the local project commit to it.

After accepting the GitHub invitation, open PowerShell and run (replace the URL with the repository URL shared by the owner):

```powershell
git clone https://github.com/OWNER/pothole-detection.git
cd pothole-detection
```

Download the actual checkpoint from Git LFS:

```powershell
git lfs install
git lfs pull
```

Verify that the checkpoint exists and is approximately 160 MB:

```powershell
Get-Item .\models\unetpp_potholemix_best.pth |
    Select-Object FullName, @{Name="Size_MB";Expression={[math]::Round($_.Length / 1MB, 2)}}
```

If the file is only a very small text file or the model is missing, run `git lfs pull` again and check that Git LFS is installed. Do not try to load an LFS pointer file as a PyTorch checkpoint.

## 4. Build the Docker image

Run these commands from the repository root (the folder containing `Dockerfile`):

```powershell
docker --version
docker info
docker build -t pothole-detection:v1 .
```

The first build may take time because Docker must download the base image and install dependencies. Subsequent builds can use Docker's build cache.

## 5. Start the container with GPU access

From the repository root in PowerShell:

```powershell
docker run --rm --gpus all -it `
  -v "${PWD}:/workspace" `
  -w /workspace `
  pothole-detection:v1 bash
```

This mounts the cloned project folder into `/workspace` inside the container. The source code, checkpoint and generated output files remain in the cloned folder on the host computer.

Inside the container, verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU detected')"
```

`CUDA available` should be `True` for the GPU workflow. If it is `False`, check the host NVIDIA driver, Docker Desktop GPU support and the `--gpus all` setting. Do not assume the model is broken when CUDA is unavailable.

## 6. Use the trained pothole segmentation model

The model consists of two parts that must be used together:

- Architecture: `src/model_unetpp.py`
- Weights: `models/unetpp_potholemix_best.pth`

The project's inference entry point is:

```text
src/predict_real_photos.py
```

From the container, inspect its supported arguments and input/output configuration before running it. The exact CLI flags and image input path are defined by that script; do not invent flags or rename the checkpoint. If the script has an `argparse` help option, use:

```bash
python src/predict_real_photos.py --help
```

If it does not support `--help`, open `src/predict_real_photos.py` and check its configuration constants and the `if __name__ == "__main__":` section. Set the input image path and checkpoint path there as required by the script, then run:

```bash
python src/predict_real_photos.py
```

Use a road image as input. The script uses the U-Net++ architecture and the trained PotholeMix checkpoint to generate a pixel-level pothole prediction. Follow the script's own output messages for the prediction/mask output location.

**Important:** The checkpoint is tied to the architecture and preprocessing used during training. Do not substitute the older LRASPP or ResNet18-frozen checkpoint, and do not change image normalization, resize dimensions, class-channel mapping or threshold casually. The PotholeMix dataset is not required for normal inference.

## 7. Use the Geo-Location Severity Engine

The geographic engine is a separate module. It does not need the image segmentation output to run; it needs the GPS coordinates of the pothole.

Run it inside the container from the repository root:

```bash
python src/geo_severity.py
```

It interactively asks for:

- Latitude
- Longitude
- Optional observed road use
- Optional site/context name

Example:

```text
Enter latitude: 30.34730
Enter longitude: 76.38280
Observed road use (optional):
Site/context name (optional):
```

For the observed road-use field:

- Press Enter to use the OpenStreetMap road classification.
- Use `internal_vehicle_access` only when you have verified that the mapped location is an internal road used by vehicles, even if OSM classifies the mapped way as a footway.
- Do not enter `car`; it is a vehicle type, not one of the supported road-use override values.

For the site/context field, enter a meaningful label such as `Thapar campus`, or press Enter to leave it blank. Do not type `nil` unless you literally want the text `nil` saved as the context.

The engine queries OpenStreetMap's Overpass service, so an internet connection is required. Public Overpass servers can be busy; the script retries some temporary HTTP errors. Results depend on OpenStreetMap coverage and tagging.

### What the geographic score means

The current provisional rule-based V1 score combines:

- Road importance: 40%
- Critical infrastructure proximity: 35%
- Public activity/infrastructure: 25%

The output includes the three component scores, a total `geo_severity_score` from 0 to 100, and a geographic impact label (`LOW`, `MODERATE`, `HIGH` or `VERY_HIGH`). The weights and distance bands are initial engineering heuristics and have not yet been validated against a larger set of locations.

The score is **not** the physical severity of the pothole. It does not measure pothole depth, area or road-surface damage. That belongs to the separate On-Road Severity Engine. The final Priority Assignment Engine that combines both is not yet implemented.

Missing facilities in OpenStreetMap results do not prove that those facilities do not exist in the real world. The current engine does not use live traffic counts. Straight-line distances are used for nearby facilities, not driving/walking route distances.

### Output files

The engine writes/updates these files under `outputs/`:

| File | Contents |
|---|---|
| `outputs/geo_features.csv` | Consolidated geographic features and geographic severity result for each report |
| `outputs/geo_nearby_facilities.csv` | Individual mapped facilities, their OSM tags, classifications and distances |

Keep these generated outputs locally unless the project owner specifically asks you to share them. They are not required to run the engine.

## 8. Run the two modules separately for now

The current workflow is:

1. Run `src/predict_real_photos.py` to segment the pothole pixels in an image.
2. Obtain the GPS coordinates associated with that image.
3. Run `src/geo_severity.py` and enter those coordinates to calculate geographic severity.
4. Keep the segmentation result and geographic result associated with the same pothole/report ID in your experiment records.

These steps are currently separate. There is not yet a single command that takes an image plus GPS and automatically runs both modules, and there is not yet a combined priority score.

## 9. Common problems

**`models/unetpp_potholemix_best.pth` is missing or tiny**

Run `git lfs install` and `git lfs pull` from the repository root. Confirm the model file is about 160 MB.

**Docker says it cannot access the GPU**

Check that the host NVIDIA driver is installed, Docker Desktop is running with WSL 2 on Windows, and the container is started with `--gpus all`. Test with the PyTorch CUDA command in Section 5.

**Overpass returns 429/502/503/504**

This is a temporary public map-service availability issue. The script retries some of these errors. Wait and rerun if the service remains unavailable.

**No nearby school, police station, bus stop or other facility is returned**

This means the queried OpenStreetMap data did not contain a matching tagged feature in the search radius. It does not prove the real-world facility is absent.

**The road type looks wrong**

OpenStreetMap may classify a vehicle-used internal road as `footway` or another type. Verify the road in the field/map and use the supported `internal_vehicle_access` override only when appropriate. Preserve the raw OSM classification for auditing.

## 10. Training or changing the model

For inference only, the trained checkpoint is sufficient along with the architecture, inference script and dependencies. To retrain or reproduce the training process, the colleague must separately obtain the PotholeMix dataset and the appropriate split/configuration files. The dataset is not included in this repository.

Do not overwrite `models/unetpp_potholemix_best.pth` with a new experiment unless the project owner has agreed on the replacement and the checkpoint has been evaluated.

## 11. Quick start checklist

- [ ] Accept the private GitHub repository invitation.
- [ ] Clone the repository.
- [ ] Run `git lfs pull` and verify the 160 MB checkpoint.
- [ ] Build the Docker image from the repository root.
- [ ] Start the container with `--gpus all`.
- [ ] Verify `torch.cuda.is_available()`.
- [ ] Check the inference script's documented arguments/input settings before running segmentation.
- [ ] Run the Geo-Location Severity Engine with a test latitude and longitude.
- [ ] Review `outputs/geo_features.csv` and `outputs/geo_nearby_facilities.csv`.

For help, share the exact command, full error message and relevant output. Do not share passwords, GitHub tokens or private credentials.
