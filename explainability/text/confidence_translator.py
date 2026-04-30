"""
explainability/text/confidence_translator.py
---------------------------------------------
Translates confidence tier and ranker output into a human-readable
confidence summary string.

Used by: clothes text, weather text, emotion text generators.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Communicating model uncertainty in plain language is a core XAI goal.
# Rather than exposing raw confidence scores, the system combines a tier
# label, a percentage, and per-column deviation flags into a single
# readable sentence. Templated output is a known limitation of rule-based
# NLG — noted as a direction for future work.
"""

from explainability.confidence_tier import confidence_tier
from explainability.confidence_ranker import confidence_ranker
from rules import SHORT_TO_DISPLAY

# ── Confidence language mapping ───────────────────────────────────────────────
# Maps tier labels to user-facing expressions.
# Kept here — only confidence_translator uses this mapping.

CONFIDENCE_LANGUAGE = {
    "high":     "Confident",
    "medium":   "Fairly confident",
    "low":      "Uncertain",
    "very_low": "Very uncertain",
}


def confidence_translator(columns, prediction_json):
    """
    Produces a plain-language confidence summary for the given columns.

    Input:  columns (list[str])    — short names e.g. ["rain", "wind"]
            prediction_json (dict) — loaded from load_prediction()

    Output: str — single confidence summary string
            e.g. "Uncertain: 38% certainty on rainfall and wind speed
                  prediction. Certainty of rainfall prediction is high,
                  certainty of wind speed prediction is low."

    If no columns deviate, second sentence is omitted.
    """
    # Step 1 — overall confidence
    tier_output = confidence_tier(columns, prediction_json)
    score       = round(tier_output["score"])
    tier        = tier_output["tier"]
    label       = CONFIDENCE_LANGUAGE[tier]

    # Step 2 — resolve display names
    display_names = [SHORT_TO_DISPLAY.get(col, col) for col in columns]

    if len(display_names) == 1:
        col_list = display_names[0]
    elif len(display_names) == 2:
        col_list = " and ".join(display_names)
    else:
        col_list = ", ".join(display_names[:-1]) + " and " + display_names[-1]

    # Step 3 — first sentence
    first = f"{label}: {score}% certainty on {col_list} prediction."

    # Step 4 — ranker deviations
    deviations = confidence_ranker(columns, prediction_json)

    if not deviations:
        return first

    # Step 5 — second sentence, one clause per deviating column
    clauses = []
    for col, level in deviations.items():
        display = SHORT_TO_DISPLAY.get(col, col)
        clauses.append(f"certainty of {display} prediction is {level}")

    second = ", ".join(clauses).capitalize() + "."

    return f"{first} {second}"


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from explainability.utils import load_prediction

    # Load TODAY JSON for July 2 scenario (YESTERDAY = 01072023, location = london)
    prediction_json = load_prediction("02072023", "london", prefix="TODAY")

    print("=== Umbrella ===")
    print(confidence_translator(["rain", "wind"], prediction_json))

    print("\n=== Scarf ===")
    print(confidence_translator(["temp_min", "wind"], prediction_json))