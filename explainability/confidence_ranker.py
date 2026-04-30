"""
explainability/confidence_ranker.py
-------------------------------------
Identifies which columns deviate significantly from the composite
confidence baseline, and flags them as higher or lower than expected.

Used by: Text Generator (emotion popup) to add nuance to confidence text.
e.g. "I'm fairly confident, though less certain about rain."

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# The baseline is the geometric mean confidence across the input columns.
# A column is flagged if it deviates by 15+ points from that baseline.
# This threshold is a pragmatic choice to surface only meaningful variation.
"""

from explainability.confidence_tier import confidence_tier
from explainability.utils import get_column_confidences

# ── Config ────────────────────────────────────────────────────────────────────

DEVIATION_THRESHOLD = 15

# ── Core function ─────────────────────────────────────────────────────────────

def confidence_ranker(columns, prediction_json):
    """
    Compares per-column confidence scores against the composite baseline
    and flags columns that deviate significantly.

    Input:  columns (list)         — short names e.g. ["rain", "cloud"]
            prediction_json (dict) — loaded from load_prediction()
    Output: dict of deviating columns → "high" or "low"
            e.g. {"rain": "low", "wind": "high"}
            empty dict if no columns deviate significantly.
    """
    baseline      = confidence_tier(columns, prediction_json)["score"]
    column_scores = get_column_confidences(columns, prediction_json)

    deviations = {}
    for short_name, score in column_scores.items():
        diff = score - baseline
        if diff >= DEVIATION_THRESHOLD:
            deviations[short_name] = "higher"
        elif diff <= -DEVIATION_THRESHOLD:
            deviations[short_name] = "lower"

    return deviations