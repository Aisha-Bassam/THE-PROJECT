"""
explainability/confidence_tier.py
----------------------------------
Takes one or more column names and a loaded prediction JSON,
and returns a combined confidence score and tier label.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Confidence Tier is used by three popups:
#   - Weather popup: confidence of dominant label columns only
#   - Outfit popup: confidence of clothing-relevant columns
#   - Emotion popup: overall composite confidence (all 7 columns)
# Tier thresholds are pragmatic boundaries chosen to produce meaningful
# variation across scenarios and map directly to the fox's 4 expressions.
"""

from common import geometric_mean
from explainability.utils import get_column_confidences

# ── Tier thresholds ───────────────────────────────────────────────────────────

TIERS = [
    (70, "high"),
    (40, "medium"),
    (20, "low"),
    (0,  "very_low"),
]

# ── Fox expression mapping ────────────────────────────────────────────────────

TIER_TO_EXPRESSION = {
    "high":     "confident",
    "medium":   "neutral",
    "low":      "hesitant",
    "very_low": "apologetic",
}

# ── Core function ─────────────────────────────────────────────────────────────

def confidence_tier(columns, prediction_json):
    """
    Combines confidence scores for given columns and returns
    a score, tier, expression, and the columns used.

    Input:  columns (list)         — short names e.g. ["rain", "cloud"]
            prediction_json (dict) — loaded from load_prediction()
    Output: dict with score, tier, expression, and columns

    Raises ValueError if columns is empty.
    """
    if not columns:
        raise ValueError("columns must not be empty.")

    column_scores = get_column_confidences(columns, prediction_json)
    scores        = list(column_scores.values())
    score         = geometric_mean(scores)

    tier = "very_low"
    for threshold, label in TIERS:
        if score >= threshold:
            tier = label
            break

    return {
        "score":      score,
        "tier":       tier,
        "expression": TIER_TO_EXPRESSION[tier],
        "columns":    list(column_scores.keys()),
    }