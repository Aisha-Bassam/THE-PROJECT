"""
explainability/thresholder.py
------------------------------
Maps a single predicted value to its category label.

Rule-based, deterministic, stateless. No model, no confidence, no SHAP.
One column at a time — caller decides what to threshold.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Thresholds are grounded in Met Office, WMO, NWS, and Beaufort citations.
# Snow is out of scope — documented as a known limitation.
# NWS cloud cover thresholds are a US source applied pragmatically to
# UK context — acknowledged.
"""

from rules import CATEGORIES

def threshold(short_name, value):
    """
    Maps a predicted value to its category label.

    Input:  short_name (str) — one of: rain, cloud, wind, humidity,
                                        temp_min, temp_max, wind_dir
            value (float)    — raw predicted value
    Output: category string  — e.g. "moderate", "sunny", "cold"

    Raises ValueError for unrecognised short_name.
    """

    # ── Rain ──────────────────────────────────────────────────────────────────
    # Source: Met Office / WMO
    if short_name == "rain":
        if value < 0.1:
            return "none"
        elif value < 1.0:
            return "light"
        elif value < 10.0:
            return "moderate"
        else:
            return "heavy"

    # ── Cloud cover ───────────────────────────────────────────────────────────
    # Source: Met Office (sunny/overcast), NWS (middle bands)
    elif short_name == "cloud":
        if value <= 25:
            return "sunny"
        elif value <= 50:
            return "mostly_sunny"
        elif value <= 85:
            return "mostly_cloudy"
        else:
            return "overcast"

    # ── Wind speed ────────────────────────────────────────────────────────────
    # Source: Met Office Beaufort scale
    elif short_name == "wind":
        if value < 20:
            return "light"
        elif value < 40:
            return "moderate"
        else:
            return "strong"

    # ── Humidity ──────────────────────────────────────────────────────────────
    # Source: Met Office comfort guidance
    elif short_name == "humidity":
        if value < 60:
            return "dry"
        elif value <= 80:
            return "normal"
        else:
            return "humid"

    # ── Min temperature ───────────────────────────────────────────────────────
    # Source: Met Office Cold Weather Alerts
    elif short_name == "temp_min":
        if value <= 10:
            return "cold"
        elif value < 25:
            return "normal"
        else:
            return "hot"

    # ── Max temperature ───────────────────────────────────────────────────────
    # Source: Met Office Heat Health Alert
    elif short_name == "temp_max":
        if value <= 10:
            return "cold"
        elif value < 25:
            return "normal"
        else:
            return "hot"

    # ── Wind direction ────────────────────────────────────────────────────────
    # 360° divided into 8 standard compass sectors
    # Used for display and text generation only — not outfit logic
    elif short_name == "wind_dir":
        if value >= 337.5 or value < 22.5:
            return "N"
        elif value < 67.5:
            return "NE"
        elif value < 112.5:
            return "E"
        elif value < 157.5:
            return "SE"
        elif value < 202.5:
            return "S"
        elif value < 247.5:
            return "SW"
        elif value < 292.5:
            return "W"
        else:
            return "NW"

    else:
        raise ValueError(f"Unrecognised short_name: '{short_name}'. "
                         f"Expected one of: {list(CATEGORIES.keys())}")