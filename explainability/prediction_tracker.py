"""
explainability/prediction_tracker.py
--------------------------------------
Compares today's actual prediction against yesterday's prediction for today,
and flags columns that changed significantly.

TODAY_<date>.json  — today's prediction (ran with today's real input)
TOMOR_<date>.json  — yesterday's prediction for today (ran with yesterday's input)

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Prediction changes are a key XAI signal — when the forecast changes,
# users feel betrayed (survey finding: "betrayal paradox"). This component
# surfaces which columns changed and by how much, feeding the emotion popup
# explanation so users understand why the forecast shifted.
"""

from rules import SHORT_TO_COLUMN

# ── Change thresholds (per column) ────────────────────────────────────────────

THRESHOLDS = {
    "rain":     1.0,
    "temp_min": 2.0,
    "temp_max": 2.0,
    "wind":     10.0,
    "humidity": 10.0,
    "cloud":    15.0,
    "wind_dir": 45.0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def circular_distance(a, b):
    """
    Computes the shortest angular distance between two wind direction values.
    Handles wrap-around at 360°.
    """
    return min(abs(a - b), 360 - abs(a - b))

def extract_predictions(prediction_json):
    """
    Extracts raw predicted values from a loaded JSON.
    Returns dict of short_name → predicted value.
    """
    result = {}
    for short_name, full_name in SHORT_TO_COLUMN.items():
        result[short_name] = prediction_json["variables"][full_name]["prediction"]
    return result

# ── Core function ─────────────────────────────────────────────────────────────

def prediction_tracker(today_json, tomorrow_json):
    """
    Compares today's prediction against yesterday's prediction for today.
    Flags columns that changed significantly.

    Input:  today_json    (dict) — loaded TODAY_<date>.json
            tomorrow_json (dict) — loaded TOMOR_<date>.json
    Output: dict of changed columns only
            e.g. {"rain": {"old": 0.5, "new": 2.3, "direction": "increase"}}
            empty dict if nothing changed significantly.
    """
    old_preds = extract_predictions(tomorrow_json)
    new_preds = extract_predictions(today_json)

    changes = {}
    for short_name in SHORT_TO_COLUMN:
        old_val   = old_preds[short_name]
        new_val   = new_preds[short_name]
        threshold = THRESHOLDS[short_name]

        if short_name == "wind_dir":
            diff = circular_distance(old_val, new_val)
        else:
            diff = abs(new_val - old_val)

        if diff >= threshold:
            changes[short_name] = {
                "old":       round(old_val, 4),
                "new":       round(new_val, 4),
                "direction": "increase" if new_val > old_val else "decrease",
            }

    return changes