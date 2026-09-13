from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename

from analysis.photo_processor import load_image, resize_image
from analysis.visual_analysis import analyze_visual_features, is_likely_payasam
from analysis.viscosity import calculate_vpai, classify_vpai
from analysis.video_processor import get_video_info
from analysis.optical_flow import calculate_optical_flow
from analysis.video_validator import validate_video


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = "pia-hackathon-secret-key"


# =========================================================
# FOLDERS
# =========================================================

IMAGE_FOLDER = "uploads/images"
VIDEO_FOLDER = "uploads/videos"

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

os.makedirs(
    VIDEO_FOLDER,
    exist_ok=True
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    latest_pai = session.get(
        "latest_pai"
    )

    latest_type = session.get(
        "latest_type"
    )

    latest_classification = session.get(
        "latest_classification"
    )

    latest_message = session.get(
        "latest_message"
    )

    return render_template(
        "index.html",
        latest_pai=latest_pai,
        latest_type=latest_type,
        latest_classification=latest_classification,
        latest_message=latest_message
    )


# =========================================================
# PHOTO ANALYSIS
# =========================================================

@app.route(
    "/analyze-photo",
    methods=["POST"]
)
def analyze_photo():

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":

        return "No image selected."


    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    file.save(filepath)


    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    image = load_image(
        filepath
    )

    if image is None:

        return "Could not read image."


    # -----------------------------------------------------
    # RESIZE IMAGE
    # -----------------------------------------------------

    image = resize_image(
        image
    )


    # -----------------------------------------------------
    # ANALYZE VISUAL FEATURES
    # -----------------------------------------------------

    features = analyze_visual_features(
        image
    )


    # -----------------------------------------------------
    # CHECK WHETHER IMAGE IS PAYASAM
    # -----------------------------------------------------

    if not is_likely_payasam(
        features,
        image
    ):

        return """
        <!DOCTYPE html>

        <html>

        <head>

            <title>
                PIA - Photo Analysis
            </title>

            <style>

                body {

                    font-family:
                        Arial,
                        sans-serif;

                    background:
                        #f5f5f5;

                    text-align:
                        center;

                    padding:
                        80px;

                }

                .box {

                    background:
                        white;

                    max-width:
                        650px;

                    margin:
                        auto;

                    padding:
                        45px;

                    border-radius:
                        20px;

                    box-shadow:
                        0 5px 20px
                        rgba(0,0,0,0.1);

                }

                h1 {

                    color:
                        #d9534f;

                }

                .message {

                    font-size:
                        20px;

                    margin-top:
                        20px;

                }

                .back {

                    display:
                        inline-block;

                    margin-top:
                        30px;

                    padding:
                        12px 25px;

                    background:
                        #333;

                    color:
                        white;

                    text-decoration:
                        none;

                    border-radius:
                        10px;

                }

            </style>

        </head>

        <body>

            <div class="box">

                <h1>
                    ❌ PHOTO NOT SUITABLE
                </h1>

                <div class="message">

                    This image does not appear
                    to contain payasam.

                </div>

                <p>

                    No V-PAI score was generated.

                </p>

                <a
                    class="back"
                    href="/"
                >

                    ← Try Again

                </a>

            </div>

        </body>

        </html>
        """


    # =====================================================
    # CALCULATE V-PAI
    # =====================================================

    raw_vpai = calculate_vpai(
        features
    )

    raw_vpai = float(
        raw_vpai
    )


    # -----------------------------------------------------
    # NORMALIZE V-PAI
    #
    # If old function gives:
    #
    # 431 → 43.1
    # 430 → 43.0
    #
    # keep it on a 0-100 scale.
    # -----------------------------------------------------

    if raw_vpai > 100:

        vpai = raw_vpai / 10

    else:

        vpai = raw_vpai


    vpai = max(
        0,
        min(
            vpai,
            100
        )
    )


    vpai = round(
        vpai,
        1
    )


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    classification = classify_vpai(
        vpai
    )


    # =====================================================
    # SAVE LATEST PHOTO RESULT
    # =====================================================

    session["latest_pai"] = vpai

    session["latest_type"] = "PHOTO"

    session["latest_classification"] = (
        classification
    )

    session["latest_message"] = (
        "Static visual analysis completed. "
        "The sample has been assigned a "
        "V-PAI score."
    )


    # =====================================================
    # PHOTO RESULT PAGE
    # =====================================================

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            PIA - Photo Analysis
        </title>

        <style>

            body {{

                font-family:
                    Arial,
                    sans-serif;

                background:
                    #f5f5f5;

                text-align:
                    center;

                padding:
                    50px;

            }}


            .container {{

                background:
                    white;

                max-width:
                    700px;

                margin:
                    auto;

                padding:
                    40px;

                border-radius:
                    20px;

                box-shadow:
                    0 5px 20px
                    rgba(0,0,0,0.1);

            }}


            h1 {{

                color:
                    #333;

            }}


            .score {{

                font-size:
                    52px;

                font-weight:
                    bold;

                margin:
                    30px 0;

            }}


            .classification {{

                font-size:
                    25px;

                margin-bottom:
                    30px;

            }}


            .features {{

                text-align:
                    left;

                background:
                    #f8f8f8;

                padding:
                    20px;

                border-radius:
                    12px;

            }}


            .features p {{

                font-size:
                    18px;

                margin:
                    12px 0;

            }}


            .back {{

                display:
                    inline-block;

                margin-top:
                    30px;

                padding:
                    12px 25px;

                background:
                    #333;

                color:
                    white;

                text-decoration:
                    none;

                border-radius:
                    10px;

            }}

        </style>

    </head>


    <body>


        <div class="container">


            <h1>
                PAYASAM ANALYSIS
            </h1>


            <h2>
                PHOTO ANALYSIS
            </h2>


            <div class="score">

                V-PAI:
                {vpai:.1f}/100

            </div>


            <div class="classification">

                {classification}

            </div>


            <div class="features">

                <h3>
                    Visual Features
                </h3>


                <p>

                    Brightness:
                    {features["brightness"]}

                </p>


                <p>

                    Saturation:
                    {features["saturation"]}

                </p>


                <p>

                    Texture:
                    {features["texture"]}

                </p>


                <p>

                    Edge Density:
                    {features["edge_density"]}

                </p>

            </div>


            <a
                class="back"
                href="/"
            >

                ← Analyze Another

            </a>


        </div>


    </body>

    </html>
    """


# =========================================================
# VIDEO ANALYSIS
# =========================================================

@app.route(
    "/analyze-video",
    methods=["POST"]
)
def analyze_video():

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "video" not in request.files:

        return "No video uploaded."

    file = request.files["video"]

    if file.filename == "":

        return "No video selected."


    # -----------------------------------------------------
    # SAVE VIDEO
    # -----------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        VIDEO_FOLDER,
        filename
    )

    file.save(filepath)


    # -----------------------------------------------------
    # GET VIDEO INFORMATION
    # -----------------------------------------------------

    video_info = get_video_info(
        filepath
    )


    # -----------------------------------------------------
    # AI VIDEO VALIDATION
    # -----------------------------------------------------

    validation = validate_video(
        filepath
    )


    # -----------------------------------------------------
    # REJECT UNSUITABLE VIDEO
    # -----------------------------------------------------

    if not validation["suitable"]:

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>
                PIA - Video Analysis
            </title>

            <style>

                body {{

                    font-family:
                        Arial,
                        sans-serif;

                    background:
                        #f5f5f5;

                    text-align:
                        center;

                    padding:
                        60px;

                }}


                .container {{

                    background:
                        white;

                    max-width:
                        700px;

                    margin:
                        auto;

                    padding:
                        40px;

                    border-radius:
                        20px;

                    box-shadow:
                        0 5px 20px
                        rgba(0,0,0,0.1);

                }}


                h1 {{

                    color:
                        #d9534f;

                }}


                .message {{

                    font-size:
                        20px;

                    margin:
                        25px;

                }}


                .info {{

                    background:
                        #f8f8f8;

                    padding:
                        20px;

                    border-radius:
                        12px;

                    text-align:
                        left;

                }}


                .info p {{

                    margin:
                        10px 0;

                }}


                .back {{

                    display:
                        inline-block;

                    margin-top:
                        30px;

                    padding:
                        12px 25px;

                    background:
                        #333;

                    color:
                        white;

                    text-decoration:
                        none;

                    border-radius:
                        10px;

                }}

            </style>

        </head>


        <body>


            <div class="container">


                <h1>
                    ❌ VIDEO NOT SUITABLE
                </h1>


                <div class="message">

                    {validation["message"]}

                </div>


                <div class="info">


                    <p>

                        <strong>
                            Visual Suitability:
                        </strong>

                        {validation["confidence"]}%

                    </p>


                    <p>

                        <strong>
                            Frames Checked:
                        </strong>

                        {validation["frames_checked"]}

                    </p>


                    <p>

                        <strong>
                            PAI:
                        </strong>

                        Not generated

                    </p>


                </div>


                <a
                    class="back"
                    href="/"
                >

                    ← Try Another Video

                </a>


            </div>


        </body>

        </html>
        """


    # =====================================================
    # OPTICAL FLOW
    # =====================================================

    flow = calculate_optical_flow(
        filepath
    )


    # =====================================================
    # FLOW VALUES
    # =====================================================

    average_movement = float(
        flow.get(
            "average_movement",
            0
        )
    )


    maximum_movement = float(
        flow.get(
            "maximum_movement",
            0
        )
    )


    consistency = float(
        flow.get(
            "movement_consistency",
            0
        )
    )


    flow_change = float(
        flow.get(
            "flow_change",
            1.0
        )
    )


    # =====================================================
    # SPEED SCORE
    # =====================================================

    speed_score = min(
        average_movement * 20,
        100
    )


    # =====================================================
    # MAXIMUM FLOW SCORE
    # =====================================================

    max_flow_score = min(
        maximum_movement * 10,
        100
    )


    # =====================================================
    # CONSISTENCY SCORE
    # =====================================================

    consistency_score = (
        consistency * 100
    )


    # =====================================================
    # FLOW TREND SCORE
    # =====================================================

    trend_difference = abs(
        flow_change - 1.0
    )


    trend_score = max(
        0,
        100 - (
            trend_difference * 100
        )
    )


    # =====================================================
    # PAI CALCULATION
    # =====================================================

    pai = (

        speed_score * 0.45

        + consistency_score * 0.25

        + max_flow_score * 0.15

        + trend_score * 0.15

    )


    pai = max(
        0,
        min(
            pai,
            100
        )
    )


    pai = round(
        pai,
        1
    )


    # =====================================================
    # PAI CLASSIFICATION
    # =====================================================

    if pai >= 75:

        classification = (
            "🌊 VERY FLUID FLOW"
        )

    elif pai >= 55:

        classification = (
            "🥣 MODERATELY FLUID FLOW"
        )

    elif pai >= 35:

        classification = (
            "🍮 THICK FLOW"
        )

    else:

        classification = (
            "🧱 VERY THICK FLOW"
        )


    # =====================================================
    # SAVE LATEST VIDEO RESULT
    # =====================================================

    session["latest_pai"] = pai

    session["latest_type"] = "VIDEO"

    session["latest_classification"] = (
        classification
    )

    session["latest_message"] = (
        "Motion and flow analysis completed. "
        "The sample has been assigned a "
        "PAI score."
    )


    # =====================================================
    # VIDEO RESULT PAGE
    #
    # IMPORTANT:
    # There is NO movement graph here.
    # =====================================================

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>
            PIA - Video Analysis
        </title>


        <style>

            body {{

                font-family:
                    Arial,
                    sans-serif;

                background:
                    #f5f5f5;

                text-align:
                    center;

                padding:
                    40px;

            }}


            .container {{

                background:
                    white;

                max-width:
                    850px;

                margin:
                    auto;

                padding:
                    40px;

                border-radius:
                    20px;

                box-shadow:
                    0 5px 20px
                    rgba(0,0,0,0.1);

            }}


            h1 {{

                color:
                    #333;

                margin-bottom:
                    10px;

            }}


            h2 {{

                color:
                    #555;

                margin-bottom:
                    25px;

            }}


            /* =================================================
               PAI
            ================================================= */

            .pai {{

                font-size:
                    58px;

                font-weight:
                    bold;

                margin:
                    25px 0;

            }}


            /* =================================================
               CLASSIFICATION
            ================================================= */

            .classification {{

                font-size:
                    26px;

                margin-bottom:
                    35px;

            }}


            /* =================================================
               STATISTICS
            ================================================= */

            .stats {{

                display:
                    grid;

                grid-template-columns:
                    repeat(2, 1fr);

                gap:
                    18px;

                margin:
                    30px 0;

            }}


            .stat {{

                background:
                    #f8f8f8;

                padding:
                    24px;

                border-radius:
                    12px;

                border:
                    1px solid #e5e5e5;

            }}


            .stat h3 {{

                margin:
                    0 0 10px 0;

                font-size:
                    20px;

            }}


            .stat p {{

                font-size:
                    22px;

                font-weight:
                    bold;

                margin:
                    0;

            }}


            /* =================================================
               EXPLANATION
            ================================================= */

            .info {{

                margin-top:
                    30px;

                padding:
                    20px;

                background:
                    #fafafa;

                border-radius:
                    12px;

                text-align:
                    left;

            }}


            .info p {{

                margin:
                    10px 0;

                line-height:
                    1.6;

            }}


            /* =================================================
               BACK BUTTON
            ================================================= */

            .back {{

                display:
                    inline-block;

                margin-top:
                    30px;

                padding:
                    13px 28px;

                background:
                    #333;

                color:
                    white;

                text-decoration:
                    none;

                border-radius:
                    10px;

                font-size:
                    16px;

            }}


            .back:hover {{

                background:
                    #555;

            }}


            /* =================================================
               MOBILE
            ================================================= */

            @media (max-width: 600px) {{

                .stats {{

                    grid-template-columns:
                        1fr;

                }}


                .pai {{

                    font-size:
                        45px;

                }}


                .classification {{

                    font-size:
                        21px;

                }}

            }}

        </style>

    </head>


    <body>


        <div class="container">


            <!-- =================================================
                 TITLE
            ================================================= -->

            <h1>

                🥣 PAYASAM VIDEO ANALYSIS

            </h1>


            <h2>

                FLOW ANALYSIS COMPLETE

            </h2>


            <!-- =================================================
                 PAI
            ================================================= -->

            <div class="pai">

                PAI:
                {pai:.1f}/100

            </div>


            <!-- =================================================
                 CLASSIFICATION
            ================================================= -->

            <div class="classification">

                {classification}

            </div>


            <!-- =================================================
                 FLOW STATISTICS
            ================================================= -->

            <div class="stats">


                <div class="stat">

                    <h3>
                        🌊 Average Flow
                    </h3>

                    <p>

                        {average_movement}

                    </p>

                </div>


                <div class="stat">

                    <h3>
                        📈 Maximum Flow
                    </h3>

                    <p>

                        {maximum_movement}

                    </p>

                </div>


                <div class="stat">

                    <h3>
                        🔄 Consistency
                    </h3>

                    <p>

                        {round(consistency * 100, 2)}%

                    </p>

                </div>


                <div class="stat">

                    <h3>
                        📊 Flow Trend
                    </h3>

                    <p>

                        {round(flow_change, 3)}

                    </p>

                </div>


            </div>


            <!-- =================================================
                 EXPLANATION
            ================================================= -->

            <div class="info">


                <p>

                    <strong>
                        How PAI is calculated:
                    </strong>

                    PAI combines flow speed,
                    movement consistency,
                    maximum movement,
                    and flow trend.

                </p>


                <p>

                    <strong>
                        Interpretation:
                    </strong>

                    Higher PAI indicates more fluid
                    movement, while lower PAI indicates
                    thicker and slower movement.

                </p>


                <p>

                    <strong>
                        Current range:
                    </strong>

                    75-100 = Very Fluid,
                    55-74 = Moderately Fluid,
                    35-54 = Thick,
                    0-34 = Very Thick.

                </p>


            </div>


            <!-- =================================================
                 BACK
            ================================================= -->

            <a
                class="back"
                href="/"
            >

                ← Analyze Another Video

            </a>


        </div>


    </body>

    </html>
    """


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )