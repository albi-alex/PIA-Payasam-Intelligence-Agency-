import cv2
import numpy as np


def calculate_optical_flow(video_path):

    print("")
    print("========================================")
    print("🌊 STARTING OPTICAL FLOW ANALYSIS")
    print("========================================")

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():

        print("❌ Could not open video.")

        return {
            "average_movement": 0,
            "maximum_movement": 0,
            "minimum_movement": 0,
            "movement_consistency": 0,
            "frames_analyzed": 0,
            "movement_values": []
        }

    # -----------------------------------------------------
    # READ FIRST FRAME
    # -----------------------------------------------------

    success, previous_frame = video.read()

    if not success:

        video.release()

        return {
            "average_movement": 0,
            "maximum_movement": 0,
            "minimum_movement": 0,
            "movement_consistency": 0,
            "frames_analyzed": 0,
            "movement_values": []
        }

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    # Resize for faster processing
    previous_gray = cv2.resize(
        previous_gray,
        (640, 360)
    )

    # -----------------------------------------------------
    # PARAMETERS
    # -----------------------------------------------------

    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03
        )
    )

    # -----------------------------------------------------
    # FIND INITIAL FEATURES
    # -----------------------------------------------------

    previous_points = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=None,
        **feature_params
    )

    movement_values = []

    frame_count = 0

    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_count += 1

        # Analyze every second frame
        if frame_count % 2 != 0:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            (640, 360)
        )

        # -------------------------------------------------
        # IF FEATURES ARE LOST, FIND NEW ONES
        # -------------------------------------------------

        if (
            previous_points is None
            or len(previous_points) < 10
        ):

            previous_points = cv2.goodFeaturesToTrack(
                previous_gray,
                mask=None,
                **feature_params
            )

        if previous_points is None:

            previous_gray = gray.copy()

            continue

        # -------------------------------------------------
        # CALCULATE OPTICAL FLOW
        # -------------------------------------------------

        current_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                previous_points,
                None,
                **lk_params
            )
        )

        if current_points is None:

            previous_gray = gray.copy()

            previous_points = None

            continue

        # -------------------------------------------------
        # SELECT GOOD POINTS
        # -------------------------------------------------

        good_previous = previous_points[
            status == 1
        ]

        good_current = current_points[
            status == 1
        ]

        if len(good_previous) > 0:

            # -------------------------------------------------
            # CALCULATE MOVEMENT
            # -------------------------------------------------

            displacement = (
                good_current
                - good_previous
            )

            distances = np.sqrt(
                np.sum(
                    displacement ** 2,
                    axis=1
                )
            )

            # Remove extreme outliers
            if len(distances) > 5:

                upper_limit = np.percentile(
                    distances,
                    95
                )

                distances = distances[
                    distances <= upper_limit
                ]

            if len(distances) > 0:

                movement = float(
                    np.mean(distances)
                )

                movement_values.append(
                    movement
                )

        # -------------------------------------------------
        # UPDATE FRAME
        # -------------------------------------------------

        previous_gray = gray.copy()

        previous_points = (
            good_current.reshape(-1, 1, 2)
            if len(good_current) > 0
            else None
        )

    video.release()

    # =====================================================
    # NO MOVEMENT DATA
    # =====================================================

    if len(movement_values) == 0:

        print("⚠️ No movement detected.")

        return {
            "average_movement": 0,
            "maximum_movement": 0,
            "minimum_movement": 0,
            "movement_consistency": 0,
            "frames_analyzed": 0,
            "movement_values": []
        }

    # =====================================================
    # CONVERT TO NUMPY
    # =====================================================

    movements = np.array(
        movement_values,
        dtype=float
    )

    # =====================================================
    # REMOVE EXTREME OUTLIERS
    # =====================================================

    if len(movements) > 10:

        lower = np.percentile(
            movements,
            5
        )

        upper = np.percentile(
            movements,
            95
        )

        movements = movements[
            (movements >= lower)
            & (movements <= upper)
        ]

    # =====================================================
    # BASIC STATISTICS
    # =====================================================

    average_movement = float(
        np.mean(movements)
    )

    maximum_movement = float(
        np.max(movements)
    )

    minimum_movement = float(
        np.min(movements)
    )

    # =====================================================
    # MOVEMENT CONSISTENCY
    #
    # Lower variation = more consistent flow
    # =====================================================

    movement_std = float(
        np.std(movements)
    )

    if average_movement > 0:

        coefficient_variation = (
            movement_std
            / average_movement
        )

        consistency = 1 / (
            1 + coefficient_variation
        )

    else:

        consistency = 0

    consistency = float(
        max(
            0,
            min(
                consistency,
                1
            )
        )
    )

    # =====================================================
    # FLOW TREND
    #
    # Compare beginning and ending movement.
    # =====================================================

    if len(movements) >= 10:

        split = len(movements) // 2

        first_half = float(
            np.mean(
                movements[:split]
            )
        )

        second_half = float(
            np.mean(
                movements[split:]
            )
        )

        if first_half > 0:

            flow_change = (
                second_half
                / first_half
            )

        else:

            flow_change = 1.0

    else:

        flow_change = 1.0

    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("")
    print("========== OPTICAL FLOW RESULT ==========")

    print(
        f"Frames analyzed: "
        f"{len(movements)}"
    )

    print(
        f"Average movement: "
        f"{average_movement:.4f}"
    )

    print(
        f"Maximum movement: "
        f"{maximum_movement:.4f}"
    )

    print(
        f"Minimum movement: "
        f"{minimum_movement:.4f}"
    )

    print(
        f"Movement consistency: "
        f"{consistency:.4f}"
    )

    print(
        f"Flow change: "
        f"{flow_change:.4f}"
    )

    print("========================================")

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "average_movement":
            round(
                average_movement,
                4
            ),

        "maximum_movement":
            round(
                maximum_movement,
                4
            ),

        "minimum_movement":
            round(
                minimum_movement,
                4
            ),

        "movement_consistency":
            round(
                consistency,
                4
            ),

        "flow_change":
            round(
                flow_change,
                4
            ),

        "frames_analyzed":
            len(movements),

        "movement_values":
            [
                round(
                    float(value),
                    4
                )
                for value in movements
            ]
    }