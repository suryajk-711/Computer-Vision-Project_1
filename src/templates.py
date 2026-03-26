import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from preprocessing import preprocess_template_for_class, CLASS_COLORS

SIFT_NFEATURES   = 500
MIN_KEYPOINTS    = 8       # skip templates with keypoints fewer than this
CACHE_FILE       = "templates_cache.pkl"


def _get_sift():
    """Returns a shared SIFT detector instance."""
    return cv2.SIFT_create(nfeatures=SIFT_NFEATURES)


def load_templates(templates_dir):
    """
    Args:
        templates_dir: Path to the root templates folder.

    Returns:
        Dict mapping class_name => list of BGR images.
        Only loads classes defined in CLASS_COLORS.
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    templates = {}
    templates_path = Path(templates_dir)

    for class_name in CLASS_COLORS.keys():
        class_dir = templates_path / class_name

        images = []
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(str(img_file))
            images.append(img)

        templates[class_name] = images
        print(f"  Loaded {len(images):>2} templates for '{class_name}'")

    return templates


# SIFT descriptor
def compute_descriptors_for_template(img, class_name, sift):
    """
    Apply class-specific color segmentation + preprocessing on a single template,
    then compute SIFT keypoints and descriptors.

    Args:
        img:        BGR template image.
        class_name: Sign class this template belongs to.
        sift:       Shared SIFT detector instance.

    Returns:
        (keypoints, descriptors)
        None ==> if too few keypoints found.
    """
    gray = preprocess_template_for_class(img, class_name)
    kp, des = sift.detectAndCompute(gray, None)

    if des is None or len(kp) < MIN_KEYPOINTS:
        return None

    return (kp, des)


def compute_template_descriptors(templates):
    """
    Args:
        templates: Dict mapping class_name => list of BGR images.

    Returns:
        Dict mapping class_name => list of (keypoints, descriptors) tuples.

    Return example:
        {
            "stop":  [(kp1, des1), (kp2, des2), ...],
            "yield": [(kp1, des1), ...],
            ...
        }
    """
    sift = _get_sift()
    descriptor_store = {}

    print("\nComputing SIFT descriptors for templates...")
    for class_name, images in templates.items():
        valid_descriptors = []

        for idx, img in enumerate(images):
            result = compute_descriptors_for_template(img, class_name, sift)

            if result is None:
                print(f"  [SKIP] {class_name} template {idx} - too few keypoints")
                continue

            valid_descriptors.append(result)

        descriptor_store[class_name] = valid_descriptors
        print(f"  {class_name}: {len(valid_descriptors)}/{len(images)} templates kept")

    return descriptor_store


# Caching templates SIFT descriptors
def save_descriptor_cache(descriptor_store, cache_path = CACHE_FILE):
    serializable = {}
    for class_name, desc_list in descriptor_store.items():
        serializable[class_name] = []
        for (kp_list, des) in desc_list:
            kp_dicts = [
                {
                    "pt":       kp.pt,
                    "size":     kp.size,
                    "angle":    kp.angle,
                    "response": kp.response,
                    "octave":   kp.octave,
                    "class_id": kp.class_id,
                }
                for kp in kp_list
            ]
            serializable[class_name].append((kp_dicts, des))

    with open(cache_path, "wb") as f:
        pickle.dump(serializable, f)

    print(f"\nDescriptor cache saved => {cache_path}")


def load_descriptor_cache(cache_path = CACHE_FILE):
    """
    Load descriptor store from pkl file.

    Returns:
        Descriptor store dict
        None ==> if cache doesn't exist.
    """
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "rb") as f:
        serializable = pickle.load(f)

    descriptor_store = {}
    for class_name, desc_list in serializable.items():
        descriptor_store[class_name] = []
        for (kp_dicts, des) in desc_list:
            kp_list = [
                cv2.KeyPoint(
                    x=kp["pt"][0],
                    y=kp["pt"][1],
                    size=kp["size"],
                    angle=kp["angle"],
                    response=kp["response"],
                    octave=kp["octave"],
                    class_id=kp["class_id"],
                )
                for kp in kp_dicts
            ]
            descriptor_store[class_name].append((kp_list, des))

    print(f"Descriptor cache loaded => {cache_path}")
    return descriptor_store


def get_template_descriptors(templates_dir, cache_path = CACHE_FILE):
    """
    Main entry point for template setup.
    Loads from cache if available, otherwise computes and caches the descriptor.

    Args:
        templates_dir:   Path to templates folder.
        cache_path:      Where to save/load the pickle cache.

    Returns:
        Descriptor store: class_name => list of (keypoints, descriptors).
    """
    cached = load_descriptor_cache(cache_path)
    if cached is not None:
        return cached

    # cache miss
    templates = load_templates(templates_dir)
    descriptor_store = compute_template_descriptors(templates)
    save_descriptor_cache(descriptor_store, cache_path)

    return descriptor_store


def summarize_descriptor_store(descriptor_store):
    """Print a summary table of loaded template descriptors."""
    print("\n   Template Descriptor Summary")
    total = 0
    for class_name, desc_list in descriptor_store.items():
        n = len(desc_list)
        total += n
        kp_counts = [len(kp) for kp, _ in desc_list]
        avg_kp = int(np.mean(kp_counts)) if kp_counts else 0
        print(f"  {class_name:<22} {n:>2} templates   avg keypoints: {avg_kp}")
    print(f"  {'TOTAL':<22} {total:>2} templates")
    print("="*70)