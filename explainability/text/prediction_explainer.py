"""
explainability/text/prediction_explainer.py
--------------------------------------------
Explains significant forecast changes using real prediction values
and SHAP attribution for the new prediction.

Called by emotion text generator when change == True.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# When the forecast changes significantly, users experience the "betrayal
# paradox" (survey finding). This component surfaces what changed, by how
# much, and what is driving the new prediction — addressing the gap between
# user expectation and model output directly.
"""

from explainability.text.shap_translator import shap_translator
from rules import SHORT_TO_DISPLAY


def prediction_explainer(today_json, changes):
    """
    Explains all significantly changed columns in plain language.

    Input:  today_json (dict) — loaded TODAY JSON
            changes (dict)    — from emotion_generator["changes"]
                                e.g. {"rain": {"old": 3.67, "new": 0.60,
                                               "direction": "decrease"}}

    Output: str — one paragraph, one sentence block per changed column.
            Empty string if changes is empty.
    """
    if not changes:
        return ""

    blocks = []

    for column, detail in changes.items():
        old_val   = round(detail["old"], 2)
        new_val   = round(detail["new"], 2)
        direction = detail["direction"]
        display   = SHORT_TO_DISPLAY.get(column, column)

        # Change sentence
        change_sentence = (
            f"{display.capitalize()} was expected at {old_val} "
            f"but is now predicted at {new_val} ({direction})."
        )

        # SHAP sentence for new prediction only
        shap_output = shap_translator(today_json, [column])
        shap_sentence = shap_output.get(column, "")

        blocks.append(f"{change_sentence} {shap_sentence}".strip())

    return " ".join(blocks)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from explainability.utils import load_prediction
    from explainability.prediction_tracker import prediction_tracker

    # TODAY JSON from July 3 scenario (YESTERDAY = 02072023)
    today_json = load_prediction("03072023", "london", prefix="TODAY")

    changes = prediction_tracker(today_json)

    print(f"Changed columns: {list(changes.keys())}")
    print()

    result = prediction_explainer(today_json, changes)

    if result:
        print("=== Prediction Explainer Output ===")
        print(result)
    else:
        print("No significant changes detected.")