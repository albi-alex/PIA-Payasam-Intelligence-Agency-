import os

from analysis.optical_flow import calculate_optical_flow


VIDEO_FOLDER = "uploads/videos"


# Find video files
videos = [
    file
    for file in os.listdir(VIDEO_FOLDER)
    if file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    )
]


if not videos:

    print("❌ No video found.")

else:

    video_path = os.path.join(
        VIDEO_FOLDER,
        videos[0]
    )

    print("Testing video:")
    print(video_path)

    print()
    print("🌊 Calculating optical flow...")

    result = calculate_optical_flow(
        video_path
    )

    print()
    print("========== OPTICAL FLOW RESULT ==========")

    print(
        "Frames analyzed:",
        result["frames_analyzed"]
    )

    print(
        "Average movement:",
        round(
            result["average_movement"],
            4
        )
    )

    print(
        "Maximum movement:",
        round(
            result["maximum_movement"],
            4
        )
    )

    print(
        "Minimum movement:",
        round(
            result["minimum_movement"],
            4
        )
    )

    print(
        "Movement consistency:",
        round(
            result["movement_consistency"],
            4
        )
    )

    print("=========================================")