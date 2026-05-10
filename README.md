# Dental X-Ray YOLOv8 Pipeline

An end-to-end pipeline for training and running YOLOv8 segmentation models on dental X-ray images — bitewing, periapical, and panoramic — with intelligent tooth detection, numbering correction, and duplicate removal.

---

## Overview

This project contains two core modules:

| File | Role |
|---|---|
| `preprocess_bitewing.py` | **Pre-processing** — standardises images before training |
| `yolo_processing.py` | **Post-processing** — refines raw model output into clean annotations |

---

## Pre-processing (`preprocess_bitewing.py`)

Prepares raw dental X-ray images for YOLOv8 training in four stages:

### 1. Mode normalisation
Ensures every image is RGB regardless of the source format.

| Input mode | Action |
|---|---|
| `L` (greyscale) | Channel replicated → RGB |
| `RGBA` | Alpha composited over white → RGB |
| `P`, `CMYK`, etc. | Converted → RGB |

### 2. Letterbox resize
Images are scaled to the target size (default `640 × 640`) while preserving the aspect ratio. The remainder is padded with grey (`114, 114, 114`) — matching YOLOv8's internal preprocessing exactly, so YOLO normalised coordinates in label files stay valid.

### 3. Brightness & contrast standardisation (CLAHE)
Contrast Limited Adaptive Histogram Equalisation is applied to the **L** (luminance) channel in LAB colour space. This corrects for different X-ray machine exposures and scanner settings without washing out structural detail.

### 4. Colour standardisation
Per-channel z-score normalisation (`(x − mean) / std`) removes residual colour bias between different acquisition hardware. The result is rescaled back to `uint8` for disk storage.

### Usage

```bash
python preprocess_bitewing.py \
    --input_dir   /path/to/raw/images \
    --output_dir  /path/to/processed/images \
    --labels_dir  /path/to/yolo/labels \   # optional – copies .txt files unchanged
    --size 640
```

Or call it programmatically:

```python
from preprocess_bitewing import preprocess_dataset

preprocess_dataset(
    input_dir   = "dataset/images/train",
    output_dir  = "dataset_processed/images/train",
    labels_dir  = "dataset/labels/train",
    target_size = 640,
)
```

---

## Post-processing (`yolo_processing.py`)

Refines raw YOLOv8 output into structured, clinically meaningful annotations.

### Core functions

#### `process_single_model(args)`
Runs a single YOLOv8 model on an image and returns a list of annotation dictionaries. Each annotation contains:

```json
{
  "label": "tooth_class",
  "bounding_box": [{"x": ..., "y": ...}, ...],
  "segmentation": [{"x": ..., "y": ...}, ...],
  "confidence": 0.94,
  "created_by": "Model v1.0.0",
  "created_on": "2024-01-01T00:00:00+00:00"
}
```

Optionally applies the pre-processing pipeline before inference (`requires_preprocessing=True`).

---

#### `process_parallel_models(img, model_group)`
Runs multiple YOLOv8 models on the same image in parallel (up to 4 workers) and merges their outputs. Duplicate detections across models are automatically removed.

```python
model_group = [
    (model_a, False),   # (model, requires_preprocessing)
    (model_b, True),
]
annotations = process_parallel_models(img, model_group)
```

---

#### `remove_duplicates(annotations, iou_threshold=50, edge_threshold_percent=0.1)`
Removes redundant detections using polygon overlap percentage (Shapely). When two annotations of the same class overlap by more than `iou_threshold`%, only the one with the highest confidence (and most detailed segmentation) is kept. Tooth-number annotations too close to the image edges are also discarded.

---

#### `postprocess_teeth_numbers_bitewing(annotations, img_array)`
Corrects tooth numbering for **bitewing** X-rays.

- Separates teeth into upper and lower jaw by vertical position.
- Determines the correct quadrant (FDI notation) from the spatial ordering of tooth positions.
- Converts FDI numbers to Universal Numbering System (1–32).
- Stores the original model label in `original_label` for traceability.

---

#### `postprocess_teeth_numbers_pariapical(annotations, img_array, pariapical_model_orientation)`
Corrects tooth numbering for **periapical** X-rays.

- Uses a separate YOLOv8 classification model (`pariapical_model_orientation`) to determine whether the image shows the upper or lower jaw.
- Applies the same FDI → Universal conversion as the bitewing postprocessor.

---

#### `process_pano_postprocessing(annotations, img_array)`
Full postprocessing pipeline for **panoramic** X-rays. Includes:

- **Tooth cap** — limits detections to a maximum of 32 teeth (16 per jaw), keeping the highest-confidence ones.
- **Jaw assignment** — uses segmentation overlap with jaw annotations (`lower jaw` / `mandible`) to assign each tooth to upper or lower jaw; falls back to vertical position if jaw annotations are absent.
- **3-molar correction** — when all three molars in a quadrant are detected, their spatial arrangement is used to assign definitive positions (e.g. teeth 1, 2, 3 or 14, 15, 16).
- **Gap detection** — recursively propagates tooth numbers outward from anchor molars, estimating missing teeth by measuring the physical gap between adjacent detections relative to per-type average tooth widths.

---

### Tooth numbering systems

The pipeline uses **FDI World Dental Federation** notation internally and converts to the **Universal Numbering System** in output labels.

| FDI quadrant | Jaw | Side |
|---|---|---|
| 1x | Upper | Right |
| 2x | Upper | Left |
| 3x | Lower | Left |
| 4x | Lower | Right |

Universal numbers run 1–16 (upper) and 17–32 (lower).

---

## Annotation output format

Every annotation returned by the post-processing functions follows this schema:

```json
{
  "label": "9",
  "bounding_box": [
    {"x": 120.0, "y": 45.0},
    {"x": 200.0, "y": 45.0},
    {"x": 200.0, "y": 130.0},
    {"x": 120.0, "y": 130.0},
    {"x": 120.0, "y": 45.0}
  ],
  "segmentation": [
    {"x": 125.3, "y": 50.1},
    ...
  ],
  "confidence": 0.91,
  "original_label": "21",
  "original_label_after_correction": "21",
  "model_label": "21",
  "created_by": "Model v1.0.0 with Auto Labelling",
  "created_on": "2024-01-01T12:00:00+00:00"
}
```

Fields `original_label`, `original_label_after_correction`, and `model_label` are only present on annotations that went through a numbering correction step.

---

## Pipeline diagram

```
Raw X-ray image
      │
      ▼
preprocess_bitewing.py
  ├── Mode normalisation  (L/RGBA → RGB)
  ├── Letterbox resize    (640 × 640)
  ├── CLAHE               (brightness/contrast)
  └── Colour z-score      (per-channel normalisation)
      │
      ▼
YOLOv8 segmentation model (trained weights)
      │
      ▼
yolo_processing.py
  ├── process_single_model / process_parallel_models
  ├── remove_duplicates
  └── postprocess_teeth_numbers_bitewing
      / postprocess_teeth_numbers_pariapical
      / process_pano_postprocessing
      │
      ▼
Structured annotation list (JSON)
```

---

## File structure

```
.
├── preprocess_bitewing.py   # Pre-processing pipeline
├── yolo_processing.py       # Post-processing pipeline
├── SETUP.md                 # Installation and usage guide
└── README.md                # This file
```

---
