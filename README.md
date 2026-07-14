# Deep-Fake-Detector
The project aim is to identify to deep fake video/audio/images that is being circulated over the internet in the Ai era

This project aims to detect deepfake media (videos and images) using transfer learning. The project is currently under development.

## Project Structure

- `deepfake-detector/`: Contains the implementation codebase.
  - `common/`: Configuration loaders, logging setups, and utility helpers.
  - `config/`: YAML files defining configuration profiles.
  - `preprocessing/`: Modulators for frame extraction, face detection, crop validation, and resizing.
  - `training/`: Scaffolding for model building (MobileNetV2, EfficientNet) and training (under development).
  - `inference/`: Scaffolding for image and video prediction (under development).
  - `main.py`: Command Line Interface (CLI) controller.

## Current Progress [UNDER DEVELOPMENT!]

- **Preprocessing Pipeline (Implemented):** Automatically extracts frames, detects/crops faces using MediaPipe, validates crop quality, resizes them to 224x224, and saves them to `datasets/processed/`.
- **Model Training & Inference (Under Development):** Skeletons for model building, training, and predicting are in place and configured with dummy datasets for architecture validation.

## Getting Started

### 1. Installation
Install the required packages:
```bash
pip install -r deepfake-detector/requirements.txt


