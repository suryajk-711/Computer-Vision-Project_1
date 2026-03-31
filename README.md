# Classical CV for Traffic Sign Detection

A classical computer vision pipeline for detecting and classifying traffic signs using HSV color segmentation and SIFT-based template matching.

---

## Overview

This system identifies traffic signs from real-world images by:
1. Isolating sign-colored regions via HSV color masking
2. Matching SIFT descriptors against hand-crafted templates
3. Scoring each candidate class and returning the best match

Supported classes: `keepRight`, `merge`, `pedestrianCrossing`, `signalAhead`, `speedLimit25`, `speedLimit35`, `stop`, `yield`, `yieldAhead`

---

## Setup

```bash
git clone git@github.com:suryajk-711/Computer-Vision-Project_1.git
cd Computer-Vision-Project_1
pip install -r requirements.txt
```

---

## Usage

**Custom dataset (camera-captured images)**
```bash
python3 src/main.py dataset --annotations dataset/annotations.csv
```

**Tiny LISA dataset**
```bash
python3 src/main.py db_lisa_tiny --annotations db_lisa_tiny/annotations.csv
```

**Web frontend**
```bash
cd src
python3 app.py
```
Then open `http://127.0.0.1:5000` and upload an image.

---

## Pipeline

```
Query Image: BGR → Color Mask → Upscale (if < 64×64) → Grayscale → CLAHE → Sharpening
Templates:   BGR → Color Mask → Upscale → Grayscale
```

- **HSV Color Segmentation** — masks pixels to only the colors relevant to each sign class, reducing background noise before SIFT runs
- **SIFT Matching** — descriptors matched against 5 templates per class using BFMatcher; top 3 template scores are averaged per class
- **ROI Cropping** — when annotations are provided, the image is cropped to the bounding box and resized to 128×128 before matching
- **Template Caching** — SIFT descriptors are precomputed and saved to `templates_cache.pkl` to avoid recomputation on subsequent runs

---

## Datasets

| Dataset | Images | Classes |
|---|---|---|
| Self-captured (phone camera, real streets) | ~70 | 3 (stop, pedestrianCrossing, signalAhead) |
| [Tiny LISA (Kaggle)](https://www.kaggle.com/datasets/mmontiel/tiny-lisa-traffic-sign-detection-dataset) | 900 | 9 |

---

## Results

**Tiny LISA Dataset**

| Metric | Value |
|---|---|
| Total images | 900 |
| Correct predictions | 79 |
| Mispredicted | 157 |
| No match (skipped) | 664 |

**Custom Dataset**

| Metric | Value |
|---|---|
| Total images | 70 |
| Correct predictions | 12 |
| Mispredicted | 57 |
| No match (skipped) | 1 |

---

## Limitations

- HSV thresholds are sensitive to lighting and do not generalize across all conditions
- Without bounding box annotations, SIFT extracts keypoints from irrelevant background regions
- Partially occluded or distant signs produce weak keypoints and are often misclassified
