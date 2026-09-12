import cv2
import numpy as np


def detect_payasam_region(image):
    """
    Try to find the region that most likely contains payasam.

    This is a simple computer-vision heuristic.
    It is not a trained food-recognition model.
    """

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Look for warm colors commonly found in payasam
    lower_warm = np.array([5, 30, 30])
    upper_warm = np.array([40, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_warm,
        upper_warm
    )

    # Remove small noise
    kernel = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Find connected regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # If no warm region is found,
    # use the complete image
    if not contours:
        return image, mask

    # Find the largest detected region
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(largest_contour)

    image_area = image.shape[0] * image.shape[1]

    # Ignore extremely small regions
    if area < image_area * 0.02:
        return image, mask

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # Crop the detected region
    region = image[
        y:y + h,
        x:x + w
    ]

    return region, mask


def analyze_visual_features(image):
    """
    Extract visual features from the likely payasam region.
    """

    # Detect likely payasam region
    payasam_region, mask = detect_payasam_region(image)

    # Convert to HSV
    hsv = cv2.cvtColor(
        payasam_region,
        cv2.COLOR_BGR2HSV
    )

    # Calculate brightness
    brightness = np.mean(
        hsv[:, :, 2]
    )

    # Calculate saturation
    saturation = np.mean(
        hsv[:, :, 1]
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        payasam_region,
        cv2.COLOR_BGR2GRAY
    )

    # Calculate texture
    texture = np.std(gray)

    # Detect edges
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Calculate edge density
    edge_density = np.mean(
        edges > 0
    )

    # Calculate smoothness
    smoothness = 1 - min(
        edge_density / 0.25,
        1
    )

    return {
        "brightness": float(brightness),
        "saturation": float(saturation),
        "texture": float(texture),
        "edge_density": float(edge_density),
        "smoothness": float(smoothness)
    }


def is_likely_payasam(features):
    """
    Perform a basic suitability check.

    This is a heuristic for our prototype.
    It is NOT a trained food-recognition model.
    """

    texture = features["texture"]
    edge_density = features["edge_density"]
    saturation = features["saturation"]

    # Very high edge density suggests a highly
    # detailed image rather than a relatively
    # smooth payasam surface.
    if edge_density > 0.18:
        return False

    # Very high saturation combined with strong
    # texture is suspicious.
    if saturation > 180 and texture > 60:
        return False

    return True