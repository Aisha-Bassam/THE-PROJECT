"""
explainability/utils.py
-----------------------
Shared utilities for all WeatherFox reasoning components.
"""

import json
import os

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