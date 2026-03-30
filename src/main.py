import os
import cv2
import argparse
import csv
import shutil
from pathlib import Path
from templates import get_template_descriptors, summarize_descriptor_store
from pipeline import run_pipeline

TEMPLATES_DIR   = "templates"
OUTPUT_DIR      = "outputs"
CACHE_FILE      = "templates_cache.pkl"


def save_annotated_image(result, output_dir):
    """
    Draw predicted class + confidence on the image and save to output_dir.
    """
    img = result["image"]

    predicted = result["predicted"] or "no_match"
    src_name  = Path(result["image_path"]).stem

    class_dir = os.path.join(output_dir, predicted)
    os.makedirs(class_dir, exist_ok=True)

    annotated = img.copy()
    label     = f"{predicted}"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated, (8, 8), (18 + tw, 18 + th + 8), (0, 0, 0), -1)

    color = (0, 255, 0) if result["status"] == "ok" else (0, 0, 255)
    cv2.putText(annotated, label, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    out_path = os.path.join(class_dir, f"{src_name}_result.jpg")
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
    annotations = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = Path(row["filename"].strip()).stem
            if fname not in annotations:
                annotations[fname] = {
                    "class": row["class"].strip()
                }
            
                if all(k in row for k in ("x1", "y1", "x2", "y2")):
                    annotations[fname].update({
                        "x1": int(row["x1"]),
                        "y1": int(row["y1"]),
                        "x2": int(row["x2"]),
                        "y2": int(row["y2"]),
                    })

    print(f"  Loaded {len(annotations)} annotations from '{csv_path}'")
    return annotations


def print_prediction_report(results, annotations):
    """
    Print per-image accuracy vs ground-truth annotations.

    Only runs when the pipeline was executed in template mode (db_lisa_tiny).
    If results were produced without annotations (custom input), nothing is done
    """
    # Check mode via the flag attached by run_pipeline
    if not annotations:
        print(f"\n{'='*50}")
        print(f"  PREDICTION REPORT")
        print(f"{'='*50}")
        print(f"  Skipped — no annotations available.")
        print(f"  Pass --annotations <path> to enable evaluation.")
        print(f"{'='*50}\n")
        return

    correct      = 0
    mispredicted = 0
    no_pred      = 0

    for r in results:
        stem       = Path(r["image_path"]).stem
        annotation = annotations.get(stem)

        if annotation is None:
            continue

        true_label = annotation["class"]
        pred_label = r["predicted"]

        if pred_label is None:
            no_pred += 1
        elif pred_label == true_label:
            correct += 1
        else:
            mispredicted += 1

    total = correct + mispredicted + no_pred

    print(f"\n{'='*50}")
    print(f"  PREDICTION REPORT")
    print(f"{'='*50}")
    print(f"  Total annotated  : {total}")
    print(f"  Correct          : {correct}")
    print(f"  Mispredicted     : {mispredicted}")
    print(f"  Couldn't predict : {no_pred}")
    print(f"{'='*50}\n")


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
        "--annotations", default=None,
        help=f"Path to annotations CSV."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Removes previous run results
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    # Load / compute template descriptors
    print("\n[1/4] Loading template descriptors...")
    descriptor_store = get_template_descriptors(
        templates_dir=args.templates,
        cache_path=CACHE_FILE
    )
    summarize_descriptor_store(descriptor_store)

    annotations_path = args.annotations
    annotations   = {}
    if annotations_path:
        print("\n[2/4] Loading annotations...")
        annotations = load_annotations(annotations_path)
    else:
        print("\n[2/4] Skipping annotations(no --annotations provided).")

    # Run pipeline
    print("\n[3/4] Running pipeline...")
    results = run_pipeline(args.input, descriptor_store, annotations)

    # Print + save results
    print("\n[4/4] Results:")
    for result in results:
        _ = save_annotated_image(result, args.output)

    print_summary(results)
    print_prediction_report(results, annotations)


def run_on_image(image_path, templates_dir=TEMPLATES_DIR, output_dir=OUTPUT_DIR):
    """
        Callable entry point for app.py — skips argparse, returns result dict.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    descriptor_store = get_template_descriptors(
        templates_dir=templates_dir,
        cache_path=CACHE_FILE
    )

    results = run_pipeline(image_path, descriptor_store, annotations={})

    for result in results:
        save_annotated_image(result, output_dir)

    return results[0] if results else None


if __name__ == "__main__":
    main()