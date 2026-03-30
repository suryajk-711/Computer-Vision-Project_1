import cv2
import numpy as np

# HSV color ranges per color name
HSV_COLOR_RANGES = {
    "red": [
        (np.array([0, 70, 50]), np.array([180, 255, 255])),
    ],
    "white": [
        (np.array([0,   0,   180]), np.array([180, 40,  255])),   # low saturation, high value
    ],
    "yellow": [
        (np.array([18,  80,  80]),  np.array([35,  255, 255])),
    ],
    "black": [
        (np.array([0,   0,   0]),   np.array([180, 255, 60])),
    ],
    "green": [
        (np.array([40,  50,  50]),  np.array([90,  255, 255])),
    ],
}

# Colors from each class
CLASS_COLORS = {
    "keepRight":          ["white", "black"],
    "merge":              ["black", "yellow"],
    "pedestrianCrossing": ["black", "yellow"],
    "signalAhead":        ["yellow", "black", "red", "green"],
    "speedLimit25":       ["white", "black"],
    "speedLimit35":       ["white", "black"],
    "stop":               ["white", "red"],
    "yield":              ["white", "red"],
    "yieldAhead":         ["white", "red", "black", "yellow"],
}


def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray, clip_limit = 2.0, tile_size= (8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization to  handles shadows and glare well.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def apply_sharpening(gray):
    """
    To enhances edges for better SIFT keypoints.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return sharpened


def upscale_if_small(img, min_size = 64):
    """
    Upscale small images before preprocessing.
    """
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        scale = min_size / min(h, w)
        # Inter cubic for interpolation
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def _build_color_mask(hsv, color):
    """
    Build a binary mask for a single color using its HSV ranges.
    """
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in HSV_COLOR_RANGES[color]:
        mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_or(combined, mask)
    return combined


def build_class_mask(img, class_name):
    """
    Build a combined binary mask for all colors associated with a given class.
    E.g. 'stop' => combines red + white masks.

    Args:
        img:        BGR image (query or template).
        class_name: One of the 9 known sign classes.

    Returns:
        Binary mask.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined = np.zeros(img.shape[:2], dtype=np.uint8)

    for color in CLASS_COLORS[class_name]:
        color_mask = _build_color_mask(hsv, color)
        combined = cv2.bitwise_or(combined, color_mask)

    return combined


def apply_mask_to_image(img, mask):
    """
    Zero out pixels outside the mask.
    To focus only on certain regions
    """
    return cv2.bitwise_and(img, img, mask=mask)


# Preprocessing pipeline
def preprocess_query(img):
    """
    grayscale => CLAHE => sharpening

    Returns a preprocessed grayscale image ready for SIFT.
    """
    img = upscale_if_small(img)
    gray = to_grayscale(img)
    gray = apply_clahe(gray)
    gray = apply_sharpening(gray)
    return gray


# Preprocessing pipeline for a template image.
def preprocess_template(img):
    """
    Templates are clean so no CLAHE and sharpening.

    Returns a preprocessed grayscale image ready for SIFT.
    """
    img = upscale_if_small(img)
    gray = to_grayscale(img)
    return gray


def preprocess_query_for_class(img, class_name):
    """
    Full query preprocessing with class-specific color masking applied.
    Returns a masked + preprocessed grayscale image.

    Pipeline:
        BGR => color mask (class-specific) => upscale => grayscale => CLAHE => sharpen
    """
    mask = build_class_mask(img, class_name)
    masked_img = apply_mask_to_image(img, mask)
    gray = preprocess_query(masked_img)
    # gray = preprocess_query(img)
    return gray


def preprocess_template_for_class(img, class_name):
    """
    Template preprocessing with class-specific color masking applied.
    Returns a masked + preprocessed grayscale image.

    Pipeline:
        BGR => color mask (class-specific) => upscale => grayscale
    """
    mask = build_class_mask(img, class_name)
    masked_img = apply_mask_to_image(img, mask)
    gray = preprocess_template(masked_img)
    # gray = preprocess_query(img)
    return gray