import cv2
import numpy as np

from PIL import Image
from transformers import pipeline


# =========================================================
# LOAD AI MODEL
# =========================================================

print("🤖 Loading photo visual AI model...")

photo_classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)

print("🤖 Photo visual AI model loaded!")


# =========================================================
# ANALYZE BASIC VISUAL FEATURES
# =========================================================

def analyze_visual_features(image):

    # Resize image
    resized = cv2.resize(
        image,
        (640, 480)
    )

    # -----------------------------------------------------
    # HSV
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2HSV
    )

    # -----------------------------------------------------
    # Brightness
    # -----------------------------------------------------

    brightness = float(
        np.mean(
            hsv[:, :, 2]
        )
    )

    # -----------------------------------------------------
    # Saturation
    # -----------------------------------------------------

    saturation = float(
        np.mean(
            hsv[:, :, 1]
        )
    )

    # -----------------------------------------------------
    # Texture
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    texture = float(
        np.std(gray)
    )

    # -----------------------------------------------------
    # Edge Density
    # -----------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    return {

        "brightness":
            round(
                brightness,
                2
            ),

        "saturation":
            round(
                saturation,
                2
            ),

        "texture":
            round(
                texture,
                2
            ),

        "edge_density":
            round(
                edge_density,
                4
            )
    }


# =========================================================
# CHECK WHETHER IMAGE IS PAYASAM
# =========================================================

def is_likely_payasam(
    features,
    image=None
):

    # =====================================================
    # AI CHECK
    # =====================================================

    if image is not None:

        # -------------------------------------------------
        # Convert BGR → RGB
        # -------------------------------------------------

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        # -------------------------------------------------
        # AI CLASSIFICATION
        # -------------------------------------------------

        try:

            results = photo_classifier(
                pil_image,
                candidate_labels=[
                    "a bowl of payasam or kheer",
                    "biriyani or savory Indian food"
                ]
            )

        except Exception as error:

            print(
                "❌ Photo AI error:",
                error
            )

            return False

        # -------------------------------------------------
        # SCORES
        # -------------------------------------------------

        payasam_score = 0.0

        other_food_score = 0.0

        for result in results:

            label = result["label"]

            score = float(
                result["score"]
            )

            if label == (
                "a bowl of payasam or kheer"
            ):

                payasam_score = score

            elif label == (
                "biriyani or savory Indian food"
            ):

                other_food_score = score

        # -------------------------------------------------
        # PRINT RESULT
        # -------------------------------------------------

        print("")
        print("========================================")
        print("📷 PHOTO AI VALIDATION")
        print("========================================")

        print(
            f"🥣 Payasam score: "
            f"{payasam_score:.3f}"
        )

        print(
            f"🍛 Other food score: "
            f"{other_food_score:.3f}"
        )

        # -------------------------------------------------
        # DECISION
        # -------------------------------------------------

        suitable = (

            payasam_score >= 0.40

            and payasam_score
                > other_food_score

        )

        if suitable:

            print(
                "✅ PHOTO ACCEPTED AS PAYASAM"
            )

        else:

            print(
                "❌ PHOTO REJECTED AS NON-PAYASAM"
            )

        print("========================================")

        return suitable


    # =====================================================
    # FALLBACK
    # =====================================================

    # If no image was supplied, use basic
    # visual features.

    brightness = features.get(
        "brightness",
        0
    )

    saturation = features.get(
        "saturation",
        0
    )

    texture = features.get(
        "texture",
        0
    )

    score = 0

    if brightness > 80:

        score += 30

    if saturation < 150:

        score += 20

    if texture < 80:

        score += 20

    return score >= 50