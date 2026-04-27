"""
explainability/clothes_mapper.py
---------------------------------
Maps today's predicted weather to a list of recommended clothing items
and their driving categories.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Clothes Mapper is the primary XAI interface in WeatherFox — the clothing
# recommendation IS the explanation. Rather than exposing raw model outputs,
# the system communicates predictions through familiar, actionable items.
# Driving categories per item feed directly into Text Generator, allowing
# confidence and SHAP explanations to be grounded in the specific weather
# signals that triggered each recommendation.
"""

from rules import CLOTHING_RULES


def clothes_mapper(categories):
    """
    Maps today's categorised weather to recommended clothing items
    and their driving categories.

    Input:  categories (dict) — {short_name: category_string}
            e.g. {"rain": "light", "cloud": "overcast", "temp_min": "cold", ...}
            Already thresholded — produced by day_pipeline.

    Output: dict of triggered items → driving categories
            e.g. {
                "umbrella": {"rain": "light", "cloud": "overcast"},
                "jacket":   {"temp_min": "cold"}
            }
            Empty dict if no items trigger.
    """
    outfit = {}

    for item, rules in CLOTHING_RULES.items():
        for rule in rules:
            if all(categories.get(col) in allowed for col, allowed in rule.items()):
                # First matching rule wins
                outfit[item] = {col: categories[col] for col in rule}
                break

    return outfit