"""
explainability/text/weather_text.py
-------------------------------------
Generates plain-language weather explanation for the weather insight popup.
Covers dominant label, always-present columns (rain, temp_min, temp_max),
and secondary labels — each with SHAP and confidence for their driving columns.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Weather text grounds the forecast label in the model's reasoning.
# Dominant label driving columns are explained first, followed by
# universally relevant signals (rain, temperature), then secondary labels.
# This mirrors the priority order in WEATHER_LABEL_PRIORITY, which is
# grounded in survey findings about what users most want to know.
"""

from explainability.text.shap_translator import shap_translator
from explainability.text.confidence_translator import confidence_translator
from rules import WEATHER_LABELS, SHORT_TO_DISPLAY


def _explain_columns(columns, categories, today_json):
    """
    Builds a text block for a set of driving columns:
    - "because" sentence from category values
    - SHAP snippets
    - confidence string

    Input:  columns (list)    — short names e.g. ["rain"]
            categories (dict) — thresholded e.g. {"rain": "light", ...}
            today_json (dict) — loaded TODAY JSON
    Output: str — assembled block, empty string if no columns
    """
    if not columns:
        return ""

    # Because sentence
    parts = []
    for col in columns:
        display  = SHORT_TO_DISPLAY.get(col, col)
        category = categories.get(col, "")
        parts.append(f"{category} {display}")

    because = ", ".join(parts) + " expected today."
    because = because.capitalize()

    # SHAP
    shap_output = shap_translator(today_json, columns)
    shap_sentences = " ".join(s for s in shap_output.values() if s)

    # Confidence
    confidence_sentence = confidence_translator(columns, today_json)

    block = " ".join(p for p in [because, shap_sentences, confidence_sentence] if p)
    return block


def weather_text(today_json, mapper_output, categories):
    """
    Generates a plain-language weather explanation for the insight popup.

    Input:  today_json (dict)     — loaded TODAY JSON
            mapper_output (dict)  — from weather_mapper: {dominant, secondary}
            categories (dict)     — thresholded e.g. {"rain": "light", ...}

    Output: single string — full weather explanation paragraph
    """
    dominant  = mapper_output["dominant"]
    secondary = mapper_output["secondary"]

    blocks = []
    covered = set()

    # Step 1 — dominant label
    dominant_columns = list(WEATHER_LABELS[dominant][0].keys())
    covered.update(dominant_columns)
    block = _explain_columns(dominant_columns, categories, today_json)
    if block:
        blocks.append(f"{dominant}: {block}")

    # Step 2 — always-present columns (rain, temp_min, temp_max)
    always = ["rain", "temp_min", "temp_max"]
    remaining_always = [col for col in always if col not in covered]
    covered.update(remaining_always)
    if remaining_always:
        block = _explain_columns(remaining_always, categories, today_json)
        if block:
            blocks.append(block)

    # Step 3 — secondary labels
    for label in secondary:
        label_columns = list(WEATHER_LABELS[label][0].keys())
        new_columns = [col for col in label_columns if col not in covered]
        covered.update(new_columns)
        if new_columns:
            block = _explain_columns(new_columns, categories, today_json)
            if block:
                blocks.append(f"{label}: {block}")

    return " ".join(blocks)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import json
    from explainability.utils import load_prediction
    from explainability.thresholder import threshold
    from explainability.weather_mapper import weather_mapper
    from rules import SCENARIO_COLUMN_TO_SHORT

    # Load TODAY JSON
    today_json = load_prediction("02072023", "london", prefix="TODAY")

    # Build categories from SCENARIO TODAY predictions
    scenario_path = os.path.join("outputs", "scenarios", "SCENARIO_01072023_london.json")
    with open(scenario_path) as f:
        raw = json.load(f)

    today_raw = raw["TODAY"]
    categories = {}
    for col, val in today_raw.items():
        short = SCENARIO_COLUMN_TO_SHORT.get(col)
        if short:
            categories[short] = threshold(short, val)

    mapper_output = weather_mapper(categories)
    print(f"Dominant: {mapper_output['dominant']}")
    print(f"Secondary: {mapper_output['secondary']}")
    print()

    result = weather_text(today_json, mapper_output, categories)
    print("=== Weather Text ===")
    print(result)