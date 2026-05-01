"""
explainability/text/clothes_text.py
-------------------------------------
Generates plain-language clothing explanations for each triggered item.
One string per item, returned as a dict keyed by item name.

Called by: text_pipeline (which passes output to Flask → UI popup per item)

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Clothing explanations are the primary XAI interface in WeatherFox.
# Each item's text combines: what conditions triggered it, which yesterday
# inputs most influenced those predictions (SHAP), and how confident the
# model is. This grounds the recommendation in the model's reasoning
# without exposing raw outputs to the user.
"""

from explainability.text.shap_translator import shap_translator
from explainability.text.confidence_translator import confidence_translator
from rules import SHORT_TO_DISPLAY, CLOTHING_KEY_COLUMNS, SCENARIO_COLUMN_TO_SHORT


def clothes_text(outfit, today_json):
    """
    Generates a plain-language explanation for each clothing item.

    Input:  outfit (dict)     — from clothes_mapper
                                e.g. {"umbrella": {"rain": "light", "wind": "light"},
                                       "scarf":   {"rain": "light", "temp_min": "normal"}}
            today_json (dict) — loaded TODAY JSON

    Output: dict {item: text} — one explanation string per item
            Empty outfit → {"none": "Nothing special to wear today."}
    """
    if not outfit:
        return {"fox_base": "Nothing special to wear today."}

    result = {}

    for item, driving_categories in outfit.items():

        # Step 1 — build "because" sentence from driving category values
        parts = []
        for col, category in driving_categories.items():
            display = SHORT_TO_DISPLAY.get(col, col)
            parts.append(f"{category} {display}")

        because_sentence = f"\n{item.capitalize()}: {', '.join(parts)} expected today.\n"

        # Step 2 — Option C filter: key columns ∩ driving categories
        key_columns = CLOTHING_KEY_COLUMNS.get(item, [])
        filtered_columns = [col for col in key_columns if col in driving_categories]

        # Step 3 — SHAP snippets (one per filtered column)
        if filtered_columns:
            shap_output = shap_translator(today_json, filtered_columns)
            shap_sentences = " ".join(
                snippet for snippet in shap_output.values() if snippet
            )
        else:
            shap_sentences = ""

        # Step 4 — confidence string
        if filtered_columns:
            confidence_sentence = confidence_translator(filtered_columns, today_json)
        else:
            confidence_sentence = ""

        # Step 5 — assemble
        text_parts = [because_sentence]
        if shap_sentences:
            text_parts.append(shap_sentences)
        if confidence_sentence:
            text_parts.append(confidence_sentence)

        result[item] = " ".join(text_parts)

    for item, text in result.items():
        result[item] = text.replace("_", " ")

    result["fox_base"] = outfit_text(result)    
    return result

def first_name(text: str) -> str:
    if not text:
        return ""
    first = text.split()[0]
    return first.rstrip(':')


def outfit_text(result):
    outfit = []
    for item, text in result.items():
        if item == "fox_base":
            return text
        outfit.append(first_name(text))

    if len(outfit) == 0:
        return ""
    
    if len(outfit) == 1:
        return f"{outfit[0]}\n"
    
    if len(outfit) == 2:
        return f"{outfit[0]} and {outfit[1]}\n"
    
    # n >= 3
    return f"{', '.join(outfit[:-1])}, and {outfit[-1]}\n"



if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from explainability.utils import load_prediction
    from explainability.thresholder import threshold
    from explainability.clothes_mapper import clothes_mapper

    # Load TODAY JSON (YESTERDAY = 01072023, location = london)
    today_json = load_prediction("02072023", "london", prefix="TODAY")

    # Build categories from TODAY input features
    # Load SCENARIO to get TODAY's predicted values
    import json
    scenario_path = os.path.join("outputs", "scenarios", "SCENARIO_01072023_london.json")
    with open(scenario_path) as f:
        raw = json.load(f)

    today_raw = raw["TODAY"]
    categories = {}
    for col, val in today_raw.items():
        short = SCENARIO_COLUMN_TO_SHORT.get(col)
        if short:
            categories[short] = threshold(short, val)

    outfit = clothes_mapper(categories)

    print(f"Outfit triggered: {list(outfit.keys())}")
    print()

    result = clothes_text(outfit, today_json)

    # print(outfit_text(result))

    for item, text in result.items():
        print(f"=== {item.upper()} ===")
        print(text)
        print()