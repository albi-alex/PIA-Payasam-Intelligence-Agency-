import cv2


def get_video_info(video_path):
    """
    Read basic information from a video using OpenCV.
    """

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError("Could not open the video.")

    # Number of frames
    frame_count = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Frames per second
    fps = video.get(
        cv2.CAP_PROP_FPS
    )

    # Video width
    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    # Video height
    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Calculate duration
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0

    video.release()

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration
    }