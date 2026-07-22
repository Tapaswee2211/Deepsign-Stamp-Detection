# Deepsign-Stamp-Detection

Automated detection, extraction, and separation of **signatures** and **stamps** from scanned documents — combining a fine-tuned YOLOv8 detector with custom OpenCV/scikit-image pipelines for clean, isolated outputs. Built during an AI Hackathon for forgery-analysis / document-verification use cases.

![Python](https://img.shields.io/badge/Python-3.x-yellow)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-blueviolet)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-green)

## Overview
--
![Overall Architecture](.media/achitecture.jpg.png)
Scanned legal/financial documents often need signatures and stamps isolated for forensic or verification analysis, but the two overlap, vary in ink color, and sit on noisy paper backgrounds — making naive cropping or thresholding unreliable. This project combines:

1. **Detection** — a YOLOv8 model (pre-trained, fine-tuned on a custom-annotated dataset of scanned documents) localizes signature and stamp regions.
2. **Extraction & separation** — two dedicated OpenCV/scikit-image pipelines then clean each region type independently, removing the *other* artifact (e.g., stripping a blue stamp out of a signature crop) and isolating the object on a clean background.

## Pipeline

```
Scanned Document
      │
      ▼
YOLOv8 Detection (fine-tuned) ── classifies + localizes: "signature" / "stamp"
      │
      ├── crop → cropped_signTest/
      └── crop → cropped_stamps/
      │
      ▼
┌─────────────────────────┐        ┌──────────────────────────┐
│ Signature Extraction     │        │ Stamp Extraction          │
├─────────────────────────┤        ├──────────────────────────┤
│ RGB → HSV                │        │ Grayscale conversion       │
│ Blue-stamp inpainting     │        │ Adaptive thresholding      │
│ (remove stamp bleed)      │        │ Contour detection +        │
│ CLAHE contrast enhance    │        │   inpainting                │
│ Adaptive thresholding     │        │ Contrast −50% /             │
│ Morphological refine      │        │   Brightness +15%           │
│ Otsu's thresholding        │        │ Multi-format export         │
│ Region-size filtering      │        │   (PNG/JPG)                 │
│ (skimage regionprops)     │        │                              │
└─────────────────────────┘        └──────────────────────────┘
      │                                       │
      ▼                                       ▼
   Clean signature on white bg          Clean isolated stamp
```

## Demo

*(Add your input/output comparison images here — you already have great before/after captures from the hackathon writeup.)*

| Stage | Input | Output |
|---|---|---|
| YOLOv8 detection | `./media/detection_input.jpg` | `./media/detection_output.jpg` |
| Signature extraction | `./media/signature_input.jpg` | `./media/signature_output.jpg` |
| Stamp extraction | `./media/stamp_input.jpg` | `./media/stamp_output.jpg` |

## Key Technical Details

**Detection**
- Started with an attempt to train a custom YOLOv8 model from scratch; annotation quality/coverage made this unreliable, so the approach was switched to fine-tuning a pre-trained YOLOv8 checkpoint on the custom dataset instead — a pragmatic tradeoff worth mentioning in interviews as a real design decision, not just a default choice.
- Detected classes: `signature`, `stamp`. Crops for each are saved separately for downstream processing.

**Signature extraction** (`process_image_combined`)
- Removes blue stamp ink first via HSV color-range masking + morphological closing + inpainting, so stamp bleed doesn't contaminate the signature mask.
- Extracts the (typically dark) signature strokes via CLAHE-enhanced adaptive thresholding, then cleans small noise using connected-component analysis (`skimage.measure.regionprops`) with size-based outlier filtering — components too small (noise) or implausibly large relative to the average are dropped.
- Final signature is composited onto a clean white background and exposure/contrast-adjusted for consistency.

**Stamp extraction** (`process_stamps` / `remove_signature`)
- Converts to grayscale, thresholds to mask out the signature, and inpaints over it so only the stamp remains.
- Applies a contrast reduction (−50%) and brightness boost to normalize appearance across scans of varying quality.

## Tech Stack

- **YOLOv8** (Ultralytics) — object detection
- **OpenCV** — color-space conversion, thresholding, morphology, inpainting
- **scikit-image** — connected-component / region analysis for noise filtering
- **PyTorch**, **Pillow**, **Matplotlib**

## Repository Structure

```
Deepsign-Stamp-Detection/
├── best.pt                  # fine-tuned YOLOv8 weights
├── detect_and_crop.py        # runs YOLOv8 inference, crops signature/stamp regions
├── extract_signature.py      # signature extraction pipeline
├── extract_stamp.py          # stamp extraction pipeline
├── docs/                     # demo images (add your before/after captures here)
└── README.md
```
*(Rename the scripts above to match your actual filenames — this structure is inferred from the shared code; update as needed.)*

## Setup

```bash
pip install ultralytics opencv-python numpy pillow torch torchvision scikit-image matplotlib
```

## Usage

```bash
# 1. Run detection + cropping (edit the dataset path in the script first)
python detect_and_crop.py

# 2. Extract clean stamps
python extract_stamp.py

# 3. Extract clean signatures
python extract_signature.py
```

Debug intermediate images (blue-mask, CLAHE output, thresholded masks, etc.) are saved automatically to a `debug/` subfolder for inspecting each pipeline stage.

## Known Limitations / Future Work

- Custom YOLOv8 training from scratch struggled with annotation quality — a larger, more carefully annotated dataset (or active-learning-based annotation) would likely improve detection recall over the fine-tuned baseline.
- Extraction thresholds (HSV ranges, CLAHE clip limits, region-size constants) are currently hand-tuned constants — could be made adaptive per-document for scans with very different lighting/ink conditions.
- [ADD: any evaluation numbers you have — detection mAP, extraction accuracy on a held-out set, processing time per document]

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV, scikit-image

