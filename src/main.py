import os
import sys
import cv2
import argparse
import csv
import numpy as np
from pathlib import Path
from templates import get_template_descriptors, summarize_descriptor_store
from pipeline import run_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

TEMPLATES_DIR   = "templates"
OUTPUT_DIR      = "outputs"
CACHE_FILE      = "templates_cache.pkl"
ANNOTATIONS_CSV = "db_lisa_tiny/annotations.csv"


def save_annotated_image(result, output_dir):
    """
    Draw predicted class + confidence on the image and save to output_dir.
    """
    img = result["image"]

    predicted = result["predicted"] or "no_match"
    src_name  = Path(result["image_path"]).stem

    annotated = img.copy()
    label     = f"{predicted}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)

    color = (0, 255, 0) if result["status"] == "ok" else (0, 0, 255)
    cv2.putText(annotated, label, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    out_path = os.path.join(output_dir, f"{src_name}_result.jpg")
    cv2.imwrite(out_path, annotated)
    return out_path


def print_summary(results):
    total    = len(results)
    matched  = sum(1 for r in results if r["status"] == "ok")
    no_match = sum(1 for r in results if r["status"] == "no_match")
    errors   = sum(1 for r in results if r["status"] == "load_error")

    print(f"\n{'='*50}")
    print(f"  SUMMARY")
    print(f"{'='*50}")
    print(f"  Total images   : {total}")
    print(f"  Matched        : {matched}")
    print(f"  No match       : {no_match}")
    print(f"  Load errors    : {errors}")
    print(f"{'='*50}\n")


def load_annotations(csv_path):
    """
    Returns:
        Dict mapping filename (stem, no extension) -> class label string.
    """
    annotations = {}
    csv_path = Path(csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = Path(row["filename"].strip()).stem
            label = row["class"].strip()
            if fname not in annotations:
                annotations[fname] = label

    print(f"  Loaded {len(annotations)} annotations from '{csv_path}'")
    return annotations


def build_confusion_matrix(results, annotations, save_path="confusion_matrix.png"):
    """
    Build a confusion matrix comparing predictions to ground truth and save as an image.

    Args:
        results: List of result dicts from run_pipeline().
        annotations: Dict mapping image stem -> true class label.
        save_path: Path to save the confusion matrix image.
    """
    pairs = []
    skipped = 0

    for r in results:
        stem = Path(r["image_path"]).stem
        true_label = annotations.get(stem)

        if true_label is None:
            skipped += 1
            continue

        pred_label = r["predicted"] if r["predicted"] is not None else "no_match"
        pairs.append((true_label, pred_label))

    if not pairs:
        print("\n  [INFO] No annotated images found — skipping confusion matrix.")
        return

    true_labels = [p[0] for p in pairs]
    pred_labels = [p[1] for p in pairs]
    all_labels = sorted(set(true_labels) | set(pred_labels))

    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    n = len(all_labels)
    matrix = np.zeros((n, n), dtype=int)

    for (true, pred) in pairs:
        i = label_to_idx[true]
        j = label_to_idx[pred]
        matrix[i][j] += 1

    plt.figure(figsize=(max(8, n), max(6, n * 0.5)))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                     xticklabels=all_labels, yticklabels=all_labels,
                     cbar=True, linewidths=0.5, linecolor="gray")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({len(pairs)} annotated images, {skipped} skipped)")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n  Confusion matrix saved as '{save_path}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Detection — Classical CV Pipeline"
    )
    parser.add_argument(
        "input",
        help="Path to a single image or a folder of images."
    )
    parser.add_argument(
        "--templates", default=TEMPLATES_DIR,
        help=f"Path to templates folder (default: '{TEMPLATES_DIR}')."
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Folder to save annotated images (default: '{OUTPUT_DIR}')."
    )
    parser.add_argument(
        "--annotations", default=ANNOTATIONS_CSV,
        help=f"Path to annotations CSV (default: '{ANNOTATIONS_CSV}')."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load / compute template descriptors
    print("\n[1/4] Loading template descriptors...")
    descriptor_store = get_template_descriptors(
        templates_dir=args.templates,
        cache_path=CACHE_FILE
    )
    summarize_descriptor_store(descriptor_store)

    # Load annotations
    print("\n[2/4] Loading annotations...")
    annotations = load_annotations(args.annotations)

    # Run pipeline
    print("\n[3/4] Running pipeline...")
    results = run_pipeline(args.input, descriptor_store)

    # Print + save results
    print("\n[4/4] Results:")
    for result in results:
        saved_path = save_annotated_image(result, args.output)
        if saved_path:
            print(f"  Saved      : {saved_path}")

    print_summary(results)
    build_confusion_matrix(results, annotations)


if __name__ == "__main__":
    main()