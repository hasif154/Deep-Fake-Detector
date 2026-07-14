# Hybrid Deepfake Detection using Transfer Learning (Phase 2 - Preprocessing)

This repository holds the implementation for a Hybrid Deepfake Detection system using Transfer Learning. Phase 2 implements a robust, configuration-driven, and modular video preprocessing pipeline.

## Preprocessing Pipeline Flow

```
Video File
   │
   ▼
1. Frame Extraction (OpenCV) -> Extracts every Nth frame
   │
   ▼
2. Face Detection (MediaPipe) -> Detects all faces, selects the largest
   │
   ▼
3. Face Cropping -> Crops bounding box with configurable padding
   │
   ▼
4. Image Validation -> Filters corrupt, blank, or low-resolution crops
   │
   ▼
5. Image Resize (224x224) -> Standardizes dimensions with cv2.INTER_AREA
   │
   ▼
6. Save Processed Image -> Groups files inside datasets/processed/<class>/<video_name>/
```

---

## Directory Layout

```
deepfake-detector/
├── common/                  # Reusable utilities (logging, config loader, file/image/video helpers)
├── config/                  # Configuration YAML files (config, dataset, training, inference, logging)
├── datasets/                # Data storage (gitignored raw & processed)
│   ├── raw/                 # Put videos here (real/ and fake/)
│   └── processed/           # Extracted face crops are saved here (real/ and fake/)
├── experiments/             # Hyperparameter settings
├── models/                  # Checkpoints, exported and pre-trained models
├── preprocessing/           # Preprocessing pipeline and individual components
├── training/                # Model structures (MobileNet, EfficientNet) and trainers
├── inference/               # Individual image and video predictors
├── scripts/                 # CLI entry point scripts (prepare_dataset.py, etc.)
├── outputs/                 # execution logs, plots, predictions
├── tests/                   # Pytest suites
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
└── main.py                  # Master CLI controller
```

---

## Execution Guide

### 1. Setup Environment
Ensure Python 3.11 is installed. Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Put Raw Data
Place your raw video files under the corresponding classes under `datasets/raw/`:
- `datasets/raw/real/`
- `datasets/raw/fake/`

### 3. Run the Preprocessing Pipeline
Execute the pipeline via the master CLI tool:
```bash
python main.py prepare
```
Or run the prepare script directly with a custom configuration:
```bash
python scripts/prepare_dataset.py --config-dir config
```

### 4. Verify Output
Processed face images will be outputted under:
- `datasets/processed/real/<video_name>/frame_xxxx.jpg`
- `datasets/processed/fake/<video_name>/frame_xxxx.jpg`

You can verify details in the execution summary logs in the console or under `outputs/logs/deepfake_detector.log`.
