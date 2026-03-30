import cv2
from pathlib import Path
from matching import score_all_classes, predict_class

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

WINDOW_SIZE  = 64

def _generate_windows(img_h, img_w):
    """
    Yield (x1, y1, x2, y2) for every window position.
    Clamps last window to image edge so borders aren't missed.
    """
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + WINDOW_SIZE, img_w)
            y2 = min(y + WINDOW_SIZE, img_h)
            x1 = max(0, x2 - WINDOW_SIZE)
            y1 = max(0, y2 - WINDOW_SIZE)
            yield (x1, y1, x2, y2)
            if x + WINDOW_SIZE >= img_w:
                break
            x += WINDOW_SIZE
        if y + WINDOW_SIZE >= img_h:
            break
        y += WINDOW_SIZE


def _sliding_window(img, descriptor_store):
    """
    Slide WINDOW_SIZES over img, score each crop.

    Returns best detection dict or None.
        {"box": (x1,y1,x2,y2), "class": str, "score": float}
    """
    img_h, img_w = img.shape[:2]
    detections = []

    windows = list(_generate_windows(img_h, img_w))

    for (x1, y1, x2, y2) in windows:
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        class_scores = score_all_classes(crop, descriptor_store)
        predicted, ranked = predict_class(class_scores)

        if predicted is None:
            continue

        best_score = ranked[0][1]

        detections.append({
            "box":   (x1, y1, x2, y2),
            "class": predicted,
            "score": best_score,
        })

    if not detections:
        return None

    detections.sort(key=lambda d: d["score"])
    return detections[0]


def process_single_image(img_path, descriptor_store, annotation):
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

    has_bbox = (
        annotation is not None and
        all(k in annotation for k in ("x1", "y1", "x2", "y2"))
    )
    box = None

    if has_bbox:
        x1, y1, x2, y2 = annotation["x1"], annotation["y1"], annotation["x2"], annotation["y2"]
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        box  = (x1, y1, x2, y2)
    else:
        crop = img

    class_scores = score_all_classes(crop, descriptor_store)
    predicted, ranked = predict_class(class_scores)

    if predicted is None:
        return {
            "image_path": img_path,
            "image":      img,
            "predicted":  None,
            "scores":     [],
            "status":     "no_match",
        }

    print(f"  {Path(img_path).name} => {predicted}")
    return {
        "image_path": img_path,
        "image":      img,
        "predicted":  predicted,
        "scores":     ranked,
        "status":     "ok",
        "box":        box,
    }


def process_folder(folder_path, descriptor_store, annotations):
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
        annotation = annotations.get(img_file.stem)
        result = process_single_image(img_file, descriptor_store, annotation)
        results.append(result)

    return results


def run_pipeline(input_path, descriptor_store, annotations):
    """
    Main entry point. Auto-detects whether input is a single image or folder.

    Args:
        input_path:       Path to an image file OR a folder of images.
        descriptor_store: Output of get_template_descriptors().
        annotations:      Dict of annotation

    Returns:
        List of result dicts.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if path.is_dir():
        results = process_folder(path, descriptor_store, annotations)
    else:
        annotation = annotations.get(path.stem)
        results    = [process_single_image(path, descriptor_store, annotation)]

    return results