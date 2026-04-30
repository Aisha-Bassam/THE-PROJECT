"""
explainability/text/shap_translator.py
----------------------------------------
Translates SHAP attribution data into human-readable text snippets.
One snippet per column, returned as a dict.

Calls shap_extractor internally — callers never touch shap_extractor directly.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Translating raw SHAP values into plain language is the core XAI mechanism
# in WeatherFox. Rather than exposing numbers, the system communicates which
# of yesterday's conditions most influenced today's prediction, and in which
# direction — making the model's reasoning interpretable to non-expert users.
"""

from explainability.text.shap_extractor import shap_extractor
from rules import SHORT_TO_COLUMN, SHAP_TOP_N

# Features that refer to YESTERDAY's observed values — prepend "yesterday's"
_YESTERDAY_FEATURES = set(SHORT_TO_COLUMN.keys())  # all 7 short names


def _resolve_feature_name(feature):
    """
    Converts a raw feature short name into a display-ready string.
    - day_of_year → "time of year" (calendar position, not a yesterday value)
    - known input features → prepend "yesterday's"
    - unknown → passed through as-is (guard fallback)
    """
    if feature == "day_of_year":
        return "time of year"
    if feature in _YESTERDAY_FEATURES:
        return f"yesterday's {feature}"
    return feature


def _select_contributors(contributors):
    """
    Selects top SHAP_TOP_N contributors by magnitude.
    If fewer than SHAP_TOP_N exist, returns all of them.
    If empty, returns empty list — caller handles this case.

    # DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
    # Limiting to top N prevents information overload in the popup —
    # grounded in survey finding that users want concise explanations.
    """
    return contributors[:SHAP_TOP_N]


def shap_translator(prediction_json, columns):
    """
    Translates SHAP data for a list of columns into plain-language snippets.

    Input:  prediction_json (dict) — loaded from load_prediction() in utils.py
            columns (list[str])    — short names e.g. ["rain", "wind"]

    Output: dict — {short_name: snippet}
            e.g. {
                "rain": "Rain is predicted mainly due to yesterday's rain
                         (increases) and humidity (increases).",
                "wind": "Wind is predicted mainly due to yesterday's wind
                         (decreases) and time of year (decreases)."
            }

    Returns empty string as snippet if no contributors selected (edge case).
    """
    result = {}

    for column in columns:
        # Extract all contributors for this column, sorted by magnitude
        extracted = shap_extractor(prediction_json, column)
        contributors = extracted[column]

        # Select top N
        selected = _select_contributors(contributors)

        if not selected:
            result[column] = ""
            continue

        # Resolve display names
        parts = []
        for c in selected:
            display_name = _resolve_feature_name(c["feature"])
            parts.append(f"{display_name} ({c['direction']})")

        # Assemble snippet
        if len(parts) == 1:
            snippet = f"{column.capitalize()} is predicted mainly due to {parts[0]}."
        else:
            snippet = f"{column.capitalize()} is predicted mainly due to {parts[0]} and {', '.join(parts[1:])}."

        result[column] = snippet

    return result


if __name__ == "__main__":
    from explainability.utils import load_prediction

    # Load TODAY JSON for July 2 scenario (YESTERDAY = 01072023, location = london)
    # TODAY JSON is saved as TODAY_02072023_london.json
    prediction_json = load_prediction("02072023", "london", prefix="TODAY")

    # Umbrella: driving categories {"rain": "light", "cloud": "mostly_cloudy", "wind": "light"}
    # Option C: CLOTHING_KEY_COLUMNS ["rain", "wind"] ∩ driving categories → ["rain", "wind"]
    umbrella_columns = ["rain", "wind"]

    # Scarf: driving categories {"rain": "light", "cloud": "mostly_cloudy", "wind": "light", "temp_min": "normal"}
    # Option C: CLOTHING_KEY_COLUMNS ["temp_min", "wind"] ∩ driving categories → ["temp_min", "wind"]
    scarf_columns = ["temp_min", "wind"]

    print("=== Umbrella ===")
    umbrella_snippets = shap_translator(prediction_json, umbrella_columns)
    for col, snippet in umbrella_snippets.items():
        print(f"  {col}: {snippet}")

    print("\n=== Scarf ===")
    scarf_snippets = shap_translator(prediction_json, scarf_columns)
    for col, snippet in scarf_snippets.items():
        print(f"  {col}: {snippet}")