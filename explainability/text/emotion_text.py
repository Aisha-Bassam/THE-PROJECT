"""
explainability/text/emotion_text.py
-------------------------------------
Generates plain-language confidence explanation for the emotion popup.
Combines fox expression, overall confidence, and prediction change explanation.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# The emotion popup communicates model uncertainty in first-person fox voice —
# grounded in survey finding that users ignored numerical confidence indicators.
# When the forecast changes, the explanation shifts to address the betrayal
# paradox directly, telling the user what changed and why.
"""

from explainability.text.confidence_translator import confidence_translator
from explainability.text.prediction_explainer import prediction_explainer
from rules import SHORT_TO_COLUMN

# ── Expression to text mapping ────────────────────────────────────────────────
# Kept here — only emotion_text uses this mapping.

EXPRESSION_TO_TEXT = {
    "confident":  "Fox is feeling confident about today's forecast",
    "neutral":    "Fox is fairly sure about today's forecast",
    "hesitant":   "Fox is a little uncertain about today's forecast",
    "apologetic": {
        False: "Fox is not very confident about today's forecast",
        True:  "Fox is sorry for the forecast change",
    }
}


def emotion_text(today_json, emotion_output, label):
    """
    Generates a plain-language confidence explanation for the emotion popup.

    Input:  today_json (dict)      — loaded TODAY JSON
            emotion_output (dict)  — from emotion_generator:
                                     {expression, change, changed, changes}
            label (str)            — dominant weather label e.g. "Light Rain"

    Output: single string — full emotion popup text
    """
    expression = emotion_output["expression"]
    change     = emotion_output["change"]
    changes    = emotion_output.get("changes", {})

    # Step 1 — expression sentence
    if expression == "apologetic":
        expression_sentence = EXPRESSION_TO_TEXT["apologetic"][change]
    else:
        expression_sentence = EXPRESSION_TO_TEXT[expression]

    # Step 2 — label
    label_sentence = f"Today's forecast: {label}."

    # Step 3 — overall confidence (all 7 columns)
    all_columns = list(SHORT_TO_COLUMN.keys())
    confidence_sentence = confidence_translator(all_columns, today_json)

    # Step 4 — prediction change explanation (only if changed)
    if change and changes:
        change_sentence = prediction_explainer(today_json, changes)
    else:
        change_sentence = ""

    # Step 5 — assemble
    parts = [
        f"{expression_sentence}; {label_sentence}",
        confidence_sentence,
    ]
    if change_sentence:
        parts.append(change_sentence)

    return {expression: " ".join(parts)}


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import json
    from explainability.utils import load_prediction
    from explainability.emotion_generator import emotion_generator
    from explainability.weather_mapper import weather_mapper, labeller
    from explainability.thresholder import threshold
    from rules import SCENARIO_COLUMN_TO_SHORT

    # Load TODAY JSON
    today_json = load_prediction("02072023", "london", prefix="TODAY")

    # Build categories from SCENARIO TODAY predictions
    scenario_path = os.path.join("outputs", "scenarios", "SCENARIO_01072023_london.json")
    with open(scenario_path) as f:
        raw = json.load(f)

    today_raw  = raw["TODAY"]
    categories = {}
    for col, val in today_raw.items():
        short = SCENARIO_COLUMN_TO_SHORT.get(col)
        if short:
            categories[short] = threshold(short, val)

    # Get label
    mapper_output  = weather_mapper(categories)
    labeller_output = labeller(mapper_output)
    label          = labeller_output["label"]

    # Get emotion
    emotion_output = emotion_generator(today_json)

    print(f"Expression: {emotion_output['expression']}")
    print(f"Change: {emotion_output['change']}")
    print(f"Changed: {emotion_output['changed']}")
    print(f"Label: {label}")
    print()

    result = emotion_text(today_json, emotion_output, label)
    print("=== Emotion Text ===")
    print(result)