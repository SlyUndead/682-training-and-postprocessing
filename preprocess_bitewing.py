"""
preprocess_bitewing.py
----------------------
Preprocessing pipeline for bitewing X-ray images before YOLOv8 training.

Performs:
  - Mode normalisation  : L (greyscale) → RGB, RGBA → RGB
  - Dimension standardisation : resize to TARGET_SIZE with letterboxing
  - Brightness / contrast standardisation via CLAHE on the luminance channel
  - Colour standardisation : per-channel z-score normalisation
  - Output: saves processed images (and optionally labels) to an output directory

Usage
-----
    python preprocess_bitewing.py \
        --input_dir  /path/to/raw/images \
        --output_dir /path/to/processed/images \
        --labels_dir /path/to/yolo/labels          # optional – copies labels unchanged
        --size 640                                  # target square size (default 640)
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# Constants (override via CLI flags if desired)
# ─────────────────────────────────────────────
TARGET_SIZE = 640          # pixels – YOLOv8 default
CLAHE_CLIP  = 2.0          # CLAHE clip limit (higher = stronger local contrast)
CLAHE_GRID  = (8, 8)       # CLAHE tile grid size
NORM_MEAN   = (0.485, 0.456, 0.406)   # ImageNet mean (R, G, B) – used for viz only
NORM_STD    = (0.229, 0.224, 0.225)   # ImageNet std  (R, G, B) – used for viz only
LETTERBOX_COLOR = (114, 114, 114)      # grey padding used by YOLOv8


# ──────────────────────────────────
# Step 1 – Mode normalisation
# ──────────────────────────────────
def normalise_mode(img: Image.Image) -> Image.Image:
    """
    Convert any PIL image to RGB.
      - L  (greyscale, 1-channel) → RGB  by replicating the single channel
      - RGBA (4-channel)          → RGB  by compositing over a white background
      - P  (palette)              → RGB
      - RGB                       → unchanged
    """
    if img.mode == "RGB":
        return img

    if img.mode == "L":
        return img.convert("RGB")          # replicates L into R, G, B

    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])   # composite alpha over white
        return background

    # Catch-all: P, LA, CMYK, etc.
    return img.convert("RGB")


# ──────────────────────────────────────────────────────────────────────
# Step 2 – Letterbox resize (preserves aspect ratio, pads with grey)
# ──────────────────────────────────────────────────────────────────────
def letterbox_resize(img_np: np.ndarray, target: int) -> np.ndarray:
    """
    Resize a NumPy HxWx3 uint8 image to (target x target) with letterboxing.
    Matches the default YOLOv8 preprocessing behaviour so label coordinates
    remain valid after the same letterbox is applied at training time.
    """
    h, w = img_np.shape[:2]
    scale = min(target / h, target / w)

    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding amounts
    pad_top    = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left   = (target - new_w) // 2
    pad_right  = target - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=LETTERBOX_COLOR,
    )
    return padded   # shape: (target, target, 3)


# ──────────────────────────────────────────────────────────
# Step 3 – Brightness / contrast standardisation via CLAHE
# ──────────────────────────────────────────────────────────
def apply_clahe(img_np: np.ndarray,
                clip_limit: float = CLAHE_CLIP,
                tile_grid: tuple  = CLAHE_GRID) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    luminance channel in LAB colour space.  This standardises local contrast
    and brightness while leaving hue and saturation intact.

    Input / output: HxWx3 uint8 BGR (OpenCV convention).
    """
    lab  = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ────────────────────────────────────────────────────────
# Step 4 – Colour standardisation (per-channel z-score)
# ────────────────────────────────────────────────────────
def standardise_colour(img_np: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score normalisation across the entire image.
    Removes channel-level colour bias introduced by different X-ray machines
    or acquisition settings.

    Input:  HxWx3 uint8 (0-255)
    Output: HxWx3 float32 (approx mean 0, std 1 per channel)

    NOTE: YOLOv8 performs its own [0,1] pixel scaling internally, so
    this function returns float32 and the caller should decide whether to:
      (a) save back as uint8 (clipped / rescaled) – recommended for on-disk storage
      (b) keep float32 for in-memory pipelines
    Here we rescale back to uint8 for on-disk storage.
    """
    img_f = img_np.astype(np.float32)

    for c in range(3):
        channel = img_f[:, :, c]
        mean, std = channel.mean(), channel.std()
        if std > 0:
            img_f[:, :, c] = (channel - mean) / std

    # Rescale float to [0, 255] uint8 for saving
    img_f = img_f - img_f.min()
    max_val = img_f.max()
    if max_val > 0:
        img_f = img_f / max_val * 255.0

    return img_f.astype(np.uint8)


# ─────────────────────────────
# Full pipeline (single image)
# ─────────────────────────────
def preprocess_image(input_path: Path,
                     output_path: Path,
                     target_size: int = TARGET_SIZE) -> None:
    """
    Run the full preprocessing pipeline on one image and save the result.
    """
    # --- Load ---
    pil_img = Image.open(input_path)

    # --- Step 1: mode ---
    pil_img = normalise_mode(pil_img)

    # Convert to NumPy BGR (OpenCV convention)
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- Step 2: letterbox resize ---
    img_np = letterbox_resize(img_np, target_size)

    # --- Step 3: brightness / contrast (CLAHE) ---
    img_np = apply_clahe(img_np)

    # --- Step 4: colour standardisation ---
    img_np = standardise_colour(img_np)

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_np)


# ─────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def preprocess_dataset(input_dir:  str,
                        output_dir: str,
                        labels_dir: str | None = None,
                        target_size: int = TARGET_SIZE) -> None:
    """
    Preprocess every image in input_dir and write results to output_dir.
    If labels_dir is provided, copies the matching .txt label files
    (YOLO format) into output_dir unchanged – the letterbox keeps
    normalised YOLO coordinates valid.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in sorted(input_dir.rglob("*"))
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        print(f"[WARNING] No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images.  Target size: {target_size}px")

    for i, img_path in enumerate(image_paths, 1):
        rel       = img_path.relative_to(input_dir)
        out_path  = output_dir / rel.with_suffix(".jpg")   # always save as JPEG

        try:
            preprocess_image(img_path, out_path, target_size)
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            continue

        # Copy matching label if labels_dir supplied
        if labels_dir:
            label_src = Path(labels_dir) / rel.with_suffix(".txt")
            if label_src.exists():
                label_dst = output_dir / rel.with_suffix(".txt")
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_src, label_dst)

        if i % 50 == 0 or i == len(image_paths):
            print(f"  [{i}/{len(image_paths)}] done")

    print(f"\nPreprocessing complete.  Results saved to: {output_dir}")


# ──────────────────────────────────
# CLI entry point
# ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess bitewing X-ray images for YOLOv8 training."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing raw images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write preprocessed images")
    parser.add_argument("--labels_dir", default=None,
                        help="Optional: directory with YOLO .txt label files to copy")
    parser.add_argument("--size", type=int, default=TARGET_SIZE,
                        help=f"Target image size in pixels (default {TARGET_SIZE})")
    args = parser.parse_args()

    preprocess_dataset(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        labels_dir  = args.labels_dir,
        target_size = args.size,
    )
