"""
ml/generate_scenario.py
-----------------------
Generates 7-day scenario data for WeatherFox prototype demonstrations.

Given a YESTERDAY date and location, produces:
  - TODAY and TOMORROW as full JSON files (predictions + confidence + SHAP)
  - FOUR through SEVEN as X-format DataFrames (predictions only, for table)

JSON files are saved to outputs/scenarios/ and consumed by all reasoning
components. The prediction dictionary is returned for the Weather Wrapper.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Predictions are chained — each day's output becomes the next day's input.
# Only TODAY and TOMORROW receive full treatment (confidence + SHAP) because:
#   - TODAY is the main forecast card and XAI explanation source
#   - TOMORROW is needed for the prediction tracker comparison
#   - FOUR through SEVEN are table-only, no explanation is shown
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
import os
import shap

from utils import (
    FEATURE_COLS, TARGET_COLS,
    load_model, load_bounds, tree_std, normalise,
    row_locator, input_to_model, output_to_model
)
from common import geometric_mean

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE = "data/uk_weather_data.csv"
OUTPUT_DIR = "outputs/scenarios"

# ── Utilities ─────────────────────────────────────────────────────────────────

def format_date(date):
    """Converts YYYY-MM-DD to DDMMYYYY for use in filenames."""
    return pd.to_datetime(date).strftime("%d%m%Y")

def increment_date(date, days):
    """Returns a new date string (YYYY-MM-DD) incremented by given number of days."""
    return (pd.to_datetime(date) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")

# ── Core functions ────────────────────────────────────────────────────────────

def predict_only(X_row):
    """
    Lightweight prediction — no confidence, no SHAP.
    Used for days FOUR through SEVEN in the 7-day chain.

    Input:  X-format DataFrame (1 row, 9 columns)
    Output: X-format DataFrame for the next day, ready for model.predict()
    """
    preds = {
        target: float(load_model(target).predict(X_row)[0])
        for target in TARGET_COLS
    }

    day_of_year   = int(X_row["day_of_year"].values[0])
    location_code = int(X_row["location_code"].values[0])

    return output_to_model(preds, day_of_year, location_code)


def generate_day_json(X_row, date, location, bounds, prefix):
    """
    Full treatment — predictions + confidence + SHAP — saved as JSON.
    Used for TODAY and TOMORROW in the 7-day chain.

    Input:  X-format DataFrame (1 row, 9 columns), date (YYYY-MM-DD),
            location string, calibration bounds, file prefix (TODAY/TOMORROW)
    Output: X-format DataFrame for the next day, ready for model.predict()
    Saves:  outputs/scenarios/<PREFIX>_<DDMMYYYY>_<location>.json

    # DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
    # JSON structure is the contract between the ML layer and all reasoning
    # components. Nothing is recomputed downstream — Flask reads these files
    # directly at request time.
    """
    results     = {}
    conf_scores = []

    for target in TARGET_COLS:
        model = load_model(target)

        # Prediction
        pred = float(model.predict(X_row)[0])

        # Confidence — normalised std dev across trees
        std    = float(tree_std(model, X_row)[0])
        lo, hi = bounds[target]
        conf   = normalise(std, lo, hi)
        conf_scores.append(conf)

        # SHAP — one value per feature, sums to prediction - base_value
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        shap_list   = [round(float(v), 6) for v in shap_values[0]]

        results[target] = {
            "prediction": round(pred, 4),
            "confidence": conf,
            "shap": {
                "base_value": round(float(explainer.expected_value), 6),
                "values":     shap_list,   # aligned to FEATURE_COLS
                "features":   FEATURE_COLS,
            }
        }

    # Composite confidence across all 7 variables
    composite = geometric_mean(conf_scores)

    # Assemble output JSON
    output = {
        "meta": {
            "date":                 date,
            "location":             location,
            "composite_confidence": composite,
        },
        "input_features": {
            col: round(float(X_row[col].values[0]), 4)
            for col in FEATURE_COLS
        },
        "variables": results,
    }

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{prefix}_{format_date(date)}_{location.lower()}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {filepath}")

    # Build and return next day's X-format input
    preds         = {t: results[t]["prediction"] for t in TARGET_COLS}
    day_of_year   = int(X_row["day_of_year"].values[0])
    location_code = int(X_row["location_code"].values[0])

    return output_to_model(preds, day_of_year, location_code)


def save_scenario(scenario, location):
    """
    Serialises the scenario dict to disk as JSON.
    DataFrames are converted to flat dicts (one row each).
    Saved to outputs/scenarios/scenario_latest.json.
    Flask reads this at request time — no models or CSV needed.
    """
    serialisable = {"Date": scenario["Date"]}

    for key in ["YESTERDAY", "TODAY", "TOMORROW", "FOUR", "FIVE", "SIX", "SEVEN"]:
        serialisable[key] = scenario[key].to_dict(orient="records")[0]

    filename = f"SCENARIO_{scenario['Date']}_{location.lower()}.json"
    # filename = f"SCENARIO_{serialisable['Date']}_{location.lower()}.json"

    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved: {filepath}")

# ── Main ──────────────────────────────────────────────────────────────────────

def generate_seven_predictions(date, location):
    """
    Generates 7 days of weather predictions starting from a given YESTERDAY date.

    Input:  date (YYYY-MM-DD) — YESTERDAY's real date
            location string — e.g. "London"
    Output: dict with keys YESTERDAY, TODAY, TOMORROW, FOUR, FIVE, SIX, SEVEN
            each holding an X-format DataFrame, plus Date for reference.

    Side effects: saves TODAY and TOMORROW JSON files to outputs/scenarios/
    """
    # Load data and bounds inside function — avoids loading on import
    df         = pd.read_csv(INPUT_FILE)
    holdout_df = df[df["year"] == 2023].copy()
    bounds     = load_bounds()

    # YESTERDAY — real data from CSV
    raw_row     = row_locator(holdout_df, date, location)
    yesterday_X = input_to_model(raw_row)

    # TODAY — predicted from YESTERDAY, full JSON saved
    today_date = increment_date(date, 1)
    today_X    = generate_day_json(yesterday_X, today_date, location, bounds, "TODAY")

    # TOMORROW — predicted from TODAY, full JSON saved (for prediction tracker)
    tomorrow_date = increment_date(date, 2)
    tomorrow_X    = generate_day_json(today_X, tomorrow_date, location, bounds, "TOMORROW")

    # FOUR through SEVEN — chained predictions only, no JSON saved
    four_X  = predict_only(tomorrow_X)
    five_X  = predict_only(four_X)
    six_X   = predict_only(five_X)
    seven_X = predict_only(six_X)

    save_scenario({"Date": format_date(date), "YESTERDAY": yesterday_X, "TODAY": today_X,
               "TOMORROW": tomorrow_X, "FOUR": four_X, "FIVE": five_X,
               "SIX": six_X, "SEVEN": seven_X}, location)

    return {
        "Date"      : format_date(date),
        "YESTERDAY" : yesterday_X,
        "TODAY"     : today_X,
        "TOMORROW"  : tomorrow_X,
        "FOUR"      : four_X,
        "FIVE"      : five_X,
        "SIX"       : six_X,
        "SEVEN"     : seven_X,
    }


# ── scenarios ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    location = "London"

    scenarios = [
        # Winter — high confidence drops next day
        "2023-12-16",
        "2023-12-17",

        # Spring — interesting outfit combos, sunglasses appear
        "2023-05-25",
        "2023-05-26",

        # Summer — dramatic outfit flip, consistently high confidence
        "2023-07-30",
        "2023-07-31",

        # Existing summer — prediction change already confirmed
        "2023-07-01",
        "2023-07-02",

        # Autumn — consecutive low confidence, extreme outfit swing
        "2023-10-09",
        "2023-10-10",
        "2023-10-11",

        # Autumn — low confidence, outfit shift across days
        "2023-09-08",
        "2023-09-10",
        "2023-09-11",

        # Winter — low confidence
        "2023-01-01",
        "2023-01-02",
    ]

    for date in scenarios:
        print(f"\nGenerating 7-day scenario for {location}, starting {date}...")
        generate_seven_predictions(date, location)
        print()