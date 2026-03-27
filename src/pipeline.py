import os
import cv2
from pathlib import Path
from matching import score_all_classes, predict_class

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def process_single_image(img_path, descriptor_store):
    """
    Run the full detection pipeline on one image.
    """
    img_path = str(img_path)
    img = cv2.imread(img_path)

    if img is None:
        return {
            "image_path": img_path,
            "image":      None,
            "predicted":  None,
            "scores":     [],
            "status":     "load_error",
        }

    class_scores = score_all_classes(img, descriptor_store)
    predicted, ranked = predict_class(class_scores)

    status = "ok" if predicted is not None else "no_match"

    return {
        "image_path": img_path,
        "image":      img,
        "predicted":  predicted,
        "scores":     ranked,
        "status":     status,
    }


def process_folder(folder_path, descriptor_store):
    """
    Run the pipeline on every image in a folder.
    """
    folder = Path(folder_path)
    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print(f"  Found {len(image_files)} images in '{folder_path}'")

    results = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"  [{idx:>3}/{len(image_files)}] Processing: {img_file.name}")
        result = process_single_image(img_file, descriptor_store)
        results.append(result)

    return results


def run_pipeline(input_path, descriptor_store):
    """
    Main entry point. Auto-detects whether input is a single image or folder.

    Args:
        input_path:       Path to an image file OR a folder of images.
        descriptor_store: Output of get_template_descriptors().

    Returns:
        List of result dicts.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_dir():
        return process_folder(path, descriptor_store)

    elif path.is_file():
        result = process_single_image(path, descriptor_store)
        return [result]