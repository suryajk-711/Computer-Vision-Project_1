import cv2
import numpy as np
from preprocessing import preprocess_query_for_class, CLASS_COLORS

MIN_GOOD_MATCHES   = 6
MIN_MATCH_COUNT = 6
SIFT_NFEATURES  = 500
TOP_K_TEMPLATES    = 3


def _get_sift():
    return cv2.SIFT_create(nfeatures=SIFT_NFEATURES)


def match_descriptors(des_query, des_template):
    """
    BFMatcher between one query descriptor and one template descriptor.
    Returns normalized average distance, or None if too few matches.

    Normalization: divide avg distance by match count so larger templates
    don't unfairly produce lower raw distances.

    Args:
        des_query:    SIFT descriptors from query region (np.ndarray).
        des_template: SIFT descriptors from one template (np.ndarray).

    Returns:
        float: normalized score (lower = better match).
        None:  if not enough matches.
    """
    if des_query is None or des_template is None:
        return None

    bf      = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_query, des_template)

    # absolute distance threshold — drop junk matches
    good = [m for m in matches if m.distance < 250]

    if len(good) < MIN_GOOD_MATCHES:
        return None

    avg_dist = float(np.mean([m.distance for m in good]))
    coverage = len(good) / len(des_query)   # rewards high explanation of query

    return avg_dist / (coverage + 1e-6)


def score_single_class(des_query, class_name, descriptor_store):
    """
    Match query descriptors against all templates of one class.
    Returns the average normalized score across all valid templates.

    Args:
        des_query:        SIFT descriptors from query image.
        class_name:       Class to score against.
        descriptor_store: Output of get_template_descriptors().

    Returns:
        float: average score across templates (lower = better).
        None:  if no valid template matches found.
    """
    templates = descriptor_store.get(class_name, [])
    if not templates:
        return None

    scores = []
    for (kp_t, des_t) in templates:
        score = match_descriptors(des_query, des_t)
        if score is not None:
            scores.append(score)

    if not scores:
        return None

    return float(np.mean(sorted(scores)[:TOP_K_TEMPLATES]))


def score_all_classes(img, descriptor_store):
    """
    Same as above function but tries all 9 classes.
    For each class: apply class-specific color segmentation on the query,
    compute SIFT, then match against all templates of that class.

    Args:
        img:              BGR query image.
        descriptor_store: Output of get_template_descriptors().

    Returns:
        Dict mapping class_name -> score (float).
        Classes with no valid matches are excluded from the dict.
    """
    sift = _get_sift()
    class_scores = {}

    for class_name in CLASS_COLORS.keys():
        gray = preprocess_query_for_class(img, class_name)
        kp, des = sift.detectAndCompute(gray, None)

        # not enough keypoints => skip
        if des is None or len(kp) < MIN_MATCH_COUNT:
            continue

        score = score_single_class(des, class_name, descriptor_store)
        if score is not None:
            class_scores[class_name] = score

    return class_scores


def predict_class(class_scores):
    """
    Pick the class with the lowest average distance score.
    """
    # Sort ascending
    if not class_scores:
        return None, []
    ranked = sorted(class_scores.items(), key=lambda x: x[1])
    predicted_class, best_score = ranked[0]
    return predicted_class, ranked