"""
explainability/utils.py
-----------------------
Shared utilities for all WeatherFox reasoning components.
"""

import json
import os
from rules import SHORT_TO_COLUMN

SCENARIOS_DIR = "outputs/scenarios"

def load_prediction(date, location, prefix):
    """
    Loads a saved prediction JSON for a given date, location, and prefix.

    Input:  date (YYYY-MM-DD), location string, prefix ("TODAY" or "TOMORROW")
    Output: parsed JSON dict — contains meta, input_features, variables

    Example filename: TODAY_02072023_london.json
    """
    formatted_date = "".join(reversed(date.split("-")))  # YYYY-MM-DD → DDMMYYYY
    filename = f"{prefix}_{formatted_date}_{location.lower()}.json"
    filepath = os.path.join(SCENARIOS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prediction file not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)
    

def get_column_confidences(columns, prediction_json):
    """
    Extracts confidence scores for given short column names from a loaded JSON.

    Input:  columns (list)        — short names e.g. ["rain", "cloud"]
            prediction_json (dict) — loaded from load_prediction()
    Output: dict — short name → confidence score (0-100)
            e.g. {"rain": 64.2, "cloud": 71.0}
    """
    result = {}
    for short_name in columns:
        full_name = SHORT_TO_COLUMN[short_name]
        result[short_name] = prediction_json["variables"][full_name]["confidence"]
    return result

def extract_predictions(prediction_json):
    """
    Extracts raw predicted values from a loaded JSON.
    Returns dict of short_name → predicted value.
    """
    result = {}
    for short_name, full_name in SHORT_TO_COLUMN.items():
        result[short_name] = prediction_json["variables"][full_name]["prediction"]
    return result