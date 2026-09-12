import cv2


def load_image(image_path):
    """
    Load an image using OpenCV.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the image.")

    return image


def resize_image(image, width=800):
    """
    Resize the image while keeping its aspect ratio.
    """

    height, original_width = image.shape[:2]

    if original_width <= width:
        return image

    ratio = width / original_width
    new_height = int(height * ratio)

    resized = cv2.resize(
        image,
        (width, new_height)
    )

    return resized