import cv2
import numpy as np

from PIL import Image
from transformers import pipeline


# =========================================================
# LOAD AI MODEL
# =========================================================

print("🤖 Loading visual AI model...")

classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)

print("🤖 Visual AI model loaded!")


# =========================================================
# CHECK PAYASAM
# =========================================================

def check_payasam(image):

    results = classifier(
        image,
        candidate_labels=[
            "a bowl of payasam or kheer",
            "a bowl of biriyani or savory Indian food"
        ]
    )

    payasam_score = 0.0
    other_food_score = 0.0

    for result in results:

        if result["label"] == "a bowl of payasam or kheer":
            payasam_score = float(result["score"])

        elif result["label"] == "a bowl of biriyani or savory Indian food":
            other_food_score = float(result["score"])

    return payasam_score, other_food_score


# =========================================================
# VALIDATE VIDEO
# =========================================================

def validate_video(video_path):

    print("")
    print("========================================")
    print("🔍 PAYASAM VIDEO VALIDATION")
    print("========================================")

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():

        return {
            "suitable": False,
            "confidence": 0,
            "message": "Could not open the video.",
            "frames_checked": 0,
            "frame_results": []
        }


    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    if total_frames <= 0:

        video.release()

        return {
            "suitable": False,
            "confidence": 0,
            "message": "No readable frames found.",
            "frames_checked": 0,
            "frame_results": []
        }


    # =====================================================
    # SELECT 5 FRAMES
    # =====================================================

    sample_count = min(
        5,
        total_frames
    )

    positions = np.linspace(
        0,
        total_frames - 1,
        sample_count,
        dtype=int
    )


    payasam_scores = []
    frame_results = []


    # =====================================================
    # ANALYZE FRAMES
    # =====================================================

    for i, position in enumerate(positions):

        print("")
        print(
            f"🧠 Analyzing frame "
            f"{i + 1}/{sample_count}"
        )


        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(position)
        )


        success, frame = video.read()


        if not success:

            print(
                "⚠️ Could not read frame."
            )

            continue


        # -------------------------------------------------
        # RESIZE
        # -------------------------------------------------

        frame = cv2.resize(
            frame,
            (640, 360)
        )


        # -------------------------------------------------
        # BGR → RGB
        # -------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        image = Image.fromarray(
            frame_rgb
        )


        # -------------------------------------------------
        # AI
        # -------------------------------------------------

        try:

            payasam_score, other_score = (
                check_payasam(image)
            )

        except Exception as error:

            print(
                "❌ AI ERROR:",
                error
            )

            continue


        payasam_scores.append(
            payasam_score
        )


        print(
            f"🥣 Payasam score: "
            f"{payasam_score:.3f}"
        )

        print(
            f"🍛 Other food score: "
            f"{other_score:.3f}"
        )


        frame_results.append({

            "frame":
                i + 1,

            "payasam_score":
                round(
                    payasam_score,
                    4
                ),

            "other_food_score":
                round(
                    other_score,
                    4
                )
        })


    video.release()


    # =====================================================
    # NO FRAMES
    # =====================================================

    if len(payasam_scores) == 0:

        return {

            "suitable": False,

            "confidence": 0,

            "message":
                "The AI could not analyze the video.",

            "frames_checked": 0,

            "frame_results": []
        }


    # =====================================================
    # CALCULATE SCORE
    # =====================================================

    average_score = float(
        np.mean(
            payasam_scores
        )
    )


    maximum_score = float(
        np.max(
            payasam_scores
        )
    )


    # =====================================================
    # PAYASAM THRESHOLD
    #
    # 25/100 or higher = ACCEPT
    # Below 25/100 = REJECT
    # =====================================================

    suitable = (
        average_score >= 0.25
    )


    # =====================================================
    # MESSAGE
    # =====================================================

    if suitable:

        message = (
            "The video appears to contain "
            "payasam or kheer and is suitable "
            "for flow analysis."
        )

        print("")
        print("✅ VIDEO ACCEPTED")
        print(
            f"🥣 Average payasam score: "
            f"{average_score:.3f}"
        )
        print("➡️ Starting PAI analysis...")


    else:

        message = (
            "The video does not appear to "
            "contain payasam or kheer."
        )

        print("")
        print("❌ VIDEO REJECTED")
        print(
            f"🥣 Average payasam score: "
            f"{average_score:.3f}"
        )
        print("➡️ PAI will NOT be generated.")


    # =====================================================
    # RETURN
    # =====================================================

    return {

        "suitable":
            suitable,

        "confidence":
            round(
                average_score * 100,
                2
            ),

        "maximum_confidence":
            round(
                maximum_score * 100,
                2
            ),

        "message":
            message,

        "frames_checked":
            len(payasam_scores),

        "frame_results":
            frame_results
    }