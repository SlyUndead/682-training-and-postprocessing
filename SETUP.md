# SETUP

## Requirements

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training and inference)
- Google Colab or a local environment with at least 8 GB RAM

---

## 1. Install Python dependencies

```bash
pip install ultralytics opencv-python-headless pillow numpy shapely
```

If you are running locally with a display (not headless):

```bash
pip install ultralytics opencv-python pillow numpy shapely
```

For GPU support, make sure your PyTorch installation matches your CUDA version before installing `ultralytics`:

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python-headless pillow numpy shapely
```

---

## 2. Dataset setup

The pipeline expects images and YOLO-format `.txt` label files arranged like this:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/          # optional
└── labels/
    ├── train/
    ├── val/
    └── test/          # optional
```

A matching `data.yaml` is required by YOLOv8:

```yaml
path: /path/to/dataset
train: images/train
val:   images/val
test:  images/test    # optional

nc: <number_of_classes>
names:
  - tooth
  - cavity
  # ... add all class names here
```

---

## 3. Preprocess the dataset

Run `preprocess_bitewing.py` on the raw images **before** training. The script standardises brightness, contrast, colour, and dimensions, and handles greyscale / RGBA inputs.

```bash
python preprocess_bitewing.py \
    --input_dir   dataset/images/train \
    --output_dir  dataset_processed/images/train \
    --labels_dir  dataset/labels/train \
    --size 640
```

Repeat for the `val` (and optionally `test`) split:

```bash
python preprocess_bitewing.py \
    --input_dir   dataset/images/val \
    --output_dir  dataset_processed/images/val \
    --labels_dir  dataset/labels/val \
    --size 640
```

Update `data.yaml` to point at `dataset_processed/` before training.

---

## 4. Train the model

```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")   # swap for yolov8s/m/l/x-seg as needed
model.train(
    data="dataset_processed/data.yaml",
    epochs=100,
    imgsz=640,
    device="cuda",   # or "cpu"
)
```

The best weights will be saved to `runs/segment/train/weights/best.pt`.

---

## 5. Run inference with postprocessing

```python
from ultralytics import YOLO
from PIL import Image
from yolo_processing import process_single_model, postprocess_teeth_numbers_bitewing
import numpy as np

model = YOLO("runs/segment/train/weights/best.pt")

img = Image.open("path/to/image.jpg")
img_array = np.array(img)

# Run model and collect raw annotations
annotations = process_single_model((img, model, False))

# Apply bitewing-specific tooth numbering corrections
annotations = postprocess_teeth_numbers_bitewing(annotations, img_array)

print(annotations)
```

For panoramic images use `process_pano_postprocessing`; for periapical images use `postprocess_teeth_numbers_pariapical` (which also requires an orientation classification model).

---

## 6. Google Colab quick start

```python
# Mount Drive and unzip dataset
from google.colab import drive
drive.mount("/content/drive")
!unzip /content/drive/MyDrive/right.zip -d /content/

# Upload the two pipeline scripts
# (preprocess_bitewing.py and yolo_processing.py)

# Install dependencies
!pip install ultralytics shapely

# Preprocess
!python preprocess_bitewing.py \
    --input_dir  /content/right/images/train \
    --output_dir /content/right_processed/images/train \
    --labels_dir /content/right/labels/train

# Train
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
model.train(data="/content/right_processed/data.yaml", epochs=100, imgsz=640, device="cuda")
```

---

## Dependency versions (tested)

| Package | Version |
|---|---|
| Python | 3.9 – 3.12 |
| ultralytics | ≥ 8.0 |
| opencv-python-headless | ≥ 4.8 |
| Pillow | ≥ 9.0 |
| numpy | ≥ 1.24 |
| shapely | ≥ 2.0 |
| torch | ≥ 2.0 |
