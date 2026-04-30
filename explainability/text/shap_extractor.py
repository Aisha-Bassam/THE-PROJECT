"""
explainability/text/shap_extractor.py
--------------------------------------
Extracts and structures SHAP attribution data for a single target variable
from a loaded prediction JSON.

Foundation of the text layer — all other text components depend on this.
Returns a self-describing dict so callers never need to track context separately.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# SHAP (SHapley Additive exPlanations) assigns each input feature a signed
# contribution to the model's prediction relative to a baseline (base_value).
# Positive = pushed prediction up; negative = pushed it down.
# Sorting by magnitude surfaces the most influential features first,
# which grounds the plain-language explanations shown to users.
"""

from rules import SHORT_TO_COLUMN, SCENARIO_COLUMN_TO_SHORT, DAY_TO_SEASON


def shap_extractor(prediction_json, column):
    """
    Extracts SHAP contributors for one target variable.

    Input:  prediction_json (dict) — loaded from load_prediction() in utils.py
            column (str)           — short name e.g. "rain", "temp_max"
    Output: dict — {column: sorted list of contributor dicts}

    Each contributor dict:
        feature    (str)   — human-ready short name or resolved season string
        shap_value (float) — raw signed SHAP value (positive = increases prediction)
        direction  (str)   — "increases" or "decreases"

    Sorted by abs(shap_value) descending (most influential first).
    location_code is filtered silently.
    day_of_year is resolved to the current season via DAY_TO_SEASON.
    """
    full_name  = SHORT_TO_COLUMN[column]
    shap_block = prediction_json["variables"][full_name]["shap"]

    # Read once — used only if day_of_year is present in the feature list
    day_of_year = prediction_json["input_features"]["day_of_year"]

    contributors = []
    for feature_name, shap_value in zip(shap_block["features"], shap_block["values"]):

        if feature_name == "location_code":
            continue

        if feature_name == "day_of_year":
            feature_short = next(name for end, name in DAY_TO_SEASON if day_of_year <= end)
        else:
            feature_short = SCENARIO_COLUMN_TO_SHORT[feature_name]

        contributors.append({
            "feature":    feature_short,
            "shap_value": shap_value,
            "direction":  "increases" if shap_value > 0 else "decreases",
        })

    contributors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {column: contributors}
