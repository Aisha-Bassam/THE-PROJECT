"""
explainability/emotion_generator.py
-------------------------------------
Generates the fox's emotional state based on overall confidence
and whether the forecast has changed since the previous prediction.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# The fox's expression is the primary XAI signal in WeatherFox.
# It reflects composite confidence across all 7 predicted variables.
# If the forecast changed significantly since yesterday's prediction,
# the expression overrides to "apologetic" regardless of confidence —
# a deliberate design choice grounded in the survey finding that
# unexpected forecast changes feel like betrayal to users.
"""

from rules import SHORT_TO_COLUMN
from confidence_tier import confidence_tier
from prediction_tracker import prediction_tracker

# ── Core function ─────────────────────────────────────────────────────────────

def emotion_generator(today_json):
    """
    Generates the fox's emotional state for a given prediction.

    Input:  today_json (dict) — loaded TODAY_<date>.json
    Output: dict with expression, label placeholder, change flag, changed columns

    # DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
    # Label is a placeholder — will be replaced by Labeller output once
    # Weather Mapper and Labeller are implemented.
    """
    # Overall confidence — always all 7 columns for emotion
    all_columns = list(SHORT_TO_COLUMN.keys())
    tier_result = confidence_tier(all_columns, today_json)
    expression  = tier_result["expression"]

    # Prediction change detection
    changes     = prediction_tracker(today_json)
    change      = len(changes) > 0
    changed     = list(changes.keys())

    # Override expression if forecast changed significantly
    if change:
        expression = "apologetic"

    return {
        "expression": expression,
        "label":      "PLACEHOLDER",
        "change":     change,
        "changed":    changed,
    }