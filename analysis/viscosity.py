def calculate_vpai(features):
    """
    Calculate the Visual Payasam Arbitrary Index.

    This is a fictional entertainment score,
    not a scientific viscosity measurement.
    """

    brightness = features["brightness"]
    saturation = features["saturation"]
    texture = features["texture"]
    edge_density = features["edge_density"]

    # Normalize values approximately to 0-100
    brightness_score = min(brightness / 255 * 100, 100)

    saturation_score = min(saturation / 255 * 100, 100)

    texture_score = min(texture / 128 * 100, 100)

    edge_score = min(edge_density * 100, 100)

    # Weighted fictional score
    vpai = (
        brightness_score * 0.20
        + saturation_score * 0.20
        + texture_score * 0.35
        + edge_score * 0.25
    )

    # Convert to 0-999
    vpai = int(vpai * 9.99)

    vpai = max(0, min(vpai, 999))

    return vpai


def classify_vpai(vpai):
    """
    Convert the V-PAI score into a funny classification.
    """

    if vpai < 200:
        return "🚨 VISUALLY LIQUID"

    elif vpai < 400:
        return "💧 VISUALLY WATERY"

    elif vpai < 600:
        return "😐 VISUALLY ACCEPTABLE"

    elif vpai < 750:
        return "🍮 VISUALLY THICK"

    elif vpai < 900:
        return "😎 VISUALLY RESPECTFULLY THICK"

    else:
        return "🏆 VISUALLY LEGENDARY"