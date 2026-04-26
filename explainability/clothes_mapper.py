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
from utils import extract_predictions
from thresholder import threshold

def clothes_mapper(today_json):
    """
    Maps today's predicted weather to recommended clothing items
    and their driving categories.

    Input:  today_json (dict) — loaded TODAY_<date>.json
    Output: dict of triggered items → driving categories
            e.g. {
                "umbrella": {"rain": "light", "cloud": "overcast", "wind": "light"},
                "jacket":   {"temp_min": "cold", "wind": "strong"}
            }
            empty dict if no items trigger.
    """
    # Step 1 — extract raw predictions and threshold all columns
    predictions = extract_predictions(today_json)
    categories  = {short: threshold(short, val) for short, val in predictions.items()}

    # Step 2 — check each item's rules against categories
    outfit = {}
    for item, rules in CLOTHING_RULES.items():
        for rule in rules:
            if all(categories.get(col) in allowed for col, allowed in rule.items()):
                # First matching rule wins — extract actual category values for driving cols
                outfit[item] = {col: categories[col] for col in rule}
                break

    return outfit