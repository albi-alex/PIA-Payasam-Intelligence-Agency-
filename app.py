from flask import Flask, render_template, request
import os

from werkzeug.utils import secure_filename

from analysis.photo_processor import load_image, resize_image

from analysis.visual_analysis import (
    analyze_visual_features,
    is_likely_payasam
)

from analysis.viscosity import (
    calculate_vpai,
    classify_vpai
)


app = Flask(__name__)


# ==========================================
# UPLOAD FOLDER
# ==========================================

UPLOAD_FOLDER = "uploads/images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# PHOTO ANALYSIS
# ==========================================

@app.route("/analyze-photo", methods=["POST"])
def analyze_photo():

    # Get the uploaded photo
    photo = request.files.get("photo")


    # Check whether a photo was selected
    if photo is None or photo.filename == "":

        return """
        <h1>❌ No photo selected</h1>

        <p>
            Please choose a payasam photo and try again.
        </p>
        """


    # Make the filename safe
    filename = secure_filename(photo.filename)


    # Create the complete file path
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # Save the uploaded photo
    photo.save(filepath)

    print("Photo saved:", filepath)


    try:

        # ==========================================
        # LOAD IMAGE
        # ==========================================

        image = load_image(filepath)


        # ==========================================
        # RESIZE IMAGE
        # ==========================================

        image = resize_image(image)


        # ==========================================
        # VISUAL ANALYSIS
        # ==========================================

        features = analyze_visual_features(image)


        print("Visual Features:", features)


        # ==========================================
        # PAYASAM CHECK
        # ==========================================

        if not is_likely_payasam(features):

            return """
            <!DOCTYPE html>

            <html>

            <head>

                <title>Payasam Not Detected</title>

            </head>


            <body>

                <h1>❌ PAYASAM NOT DETECTED</h1>


                <h2>
                    🍛 Something suspicious was detected.
                </h2>


                <p>
                    The image contains visual characteristics
                    that do not look suitable for our
                    payasam analysis.
                </p>


                <p>
                    Please upload a clearer photo of your
                    payasam.
                </p>


                <hr>


                <h3>
                    🔬 Computer Vision Check
                </h3>


                <p>
                    The image failed our basic
                    payasam suitability check.
                </p>


                <p>
                    Try taking a photo with the payasam
                    clearly visible and with good lighting.
                </p>


            </body>

            </html>
            """


        # ==========================================
        # CALCULATE V-PAI
        # ==========================================

        vpai = calculate_vpai(features)


        # ==========================================
        # CLASSIFICATION
        # ==========================================

        classification = classify_vpai(vpai)


        print("V-PAI:", vpai)

        print("Classification:", classification)


        # ==========================================
        # RESULT PAGE
        # ==========================================

        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <meta charset="UTF-8">

            <meta name="viewport"
                  content="width=device-width, initial-scale=1.0">

            <title>Payasam Analysis</title>

        </head>


        <body>


            <h1>
                🍮 PAYASAM ANALYSIS
            </h1>


            <h2>
                📸 PHOTO ANALYSIS
            </h2>


            <hr>


            <h1>
                V-PAI: {vpai}
            </h1>


            <h2>
                {classification}
            </h2>


            <hr>


            <h3>
                🔬 Visual Features
            </h3>


            <p>
                Brightness:
                {features["brightness"]:.2f}
            </p>


            <p>
                Saturation:
                {features["saturation"]:.2f}
            </p>


            <p>
                Texture:
                {features["texture"]:.2f}
            </p>


            <p>
                Edge Density:
                {features["edge_density"]:.2%}
            </p>


        </body>

        </html>
        """


    # ==========================================
    # ERROR HANDLING
    # ==========================================

    except Exception as error:

        print("Analysis error:", error)


        return f"""
        <!DOCTYPE html>

        <html>

        <head>

            <title>Analysis Error</title>

        </head>


        <body>

            <h1>
                ❌ COULD NOT ANALYZE THE IMAGE
            </h1>


            <p>
                Something went wrong while analyzing
                the uploaded image.
            </p>


            <p>
                Technical error:
                {error}
            </p>


        </body>

        </html>
        """


# ==========================================
# START FLASK
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)