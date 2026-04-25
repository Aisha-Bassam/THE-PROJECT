"""
ml/generate_scenario.py
-----------------------
Generates 7-day scenario data for WeatherFox prototype demonstrations.
Uses pre-trained models from outputs/models/ and pre-computed confidence
bounds from outputs/models/confidence_bounds.pkl.
"""

import pandas as pd
import json
import os
import shap

from utils import (
    FEATURE_COLS, TARGET_COLS, MODELS_DIR,
    load_model, load_bounds, tree_std, normalise, geometric_mean,
    row_locator, input_to_model, output_to_model
)

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE   = "data/uk_weather_data.csv"
OUTPUT_DIR   = "outputs/scenarios"

# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_only(X_row):
    """
    Lightweight prediction only — no confidence, no SHAP.
    Used for days FOUR through SEVEN in the 7-day chain.

    Takes X format (9-column DataFrame), runs all 7 models,
    returns next day's X format ready for the next prediction.
    """
    preds = {target: float(load_model(target).predict(X_row)[0]) for target in TARGET_COLS}

    day_of_year  = int(X_row["day_of_year"].values[0])
    location_code = int(X_row["location_code"].values[0])

    return output_to_model(preds, day_of_year, location_code)


def generate_day_json(X_row, date, location, bounds):
    """
    Full treatment — predictions + confidence + SHAP — saved as JSON.
    Used for TODAY and TOMORROW in the 7-day chain.

    Takes X format (9-column DataFrame), runs all 7 models,
    saves outputs/scenarios/<date>_<location>.json,
    returns next day's X format ready for the next prediction.
    """
    results     = {}
    conf_scores = []

    for target in TARGET_COLS:
        model = load_model(target)

        pred = float(model.predict(X_row)[0])

        std    = float(tree_std(model, X_row)[0])
        lo, hi = bounds[target]
        conf   = normalise(std, lo, hi)
        conf_scores.append(conf)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        shap_list   = [round(float(v), 6) for v in shap_values[0]]

        results[target] = {
            "prediction": round(pred, 4),
            "confidence": conf,
            "shap": {
                "base_value": round(float(explainer.expected_value), 6),
                "values":     shap_list,
                "features":   FEATURE_COLS,
            }
        }

    composite = geometric_mean(conf_scores)

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{date}_{location.lower()}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {filepath}")

    preds        = {t: results[t]["prediction"] for t in TARGET_COLS}
    day_of_year  = int(X_row["day_of_year"].values[0]) 
    location_code = int(X_row["location_code"].values[0])

    return output_to_model(preds, day_of_year, location_code)


# ── DEMO: How it is used ──────────────────────────────────────────────────────

if __name__ == "__main__":

    df         = pd.read_csv(INPUT_FILE)
    holdout_df = df[df["year"] == 2023].copy()
    bounds     = load_bounds()

    date     = "2023-07-01"
    location = "London"

    # Get real YESTERDAY data
    raw_row     = row_locator(holdout_df, date, location)
    yesterday_X = input_to_model(raw_row)

    # TODAY and TOMORROW — full JSON
    today_X    = generate_day_json(yesterday_X, "2023-07-02", location, bounds)
    tomorrow_X = generate_day_json(today_X,     "2023-07-03", location, bounds)

    # FOUR through SEVEN — predictions only
    four_X  = predict_only(tomorrow_X)
    five_X  = predict_only(four_X)
    six_X   = predict_only(five_X)
    seven_X = predict_only(six_X)

    print("\nDone. JSONs saved to outputs/scenarios/")