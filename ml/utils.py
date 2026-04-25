"""
ml/utils.py
-----------
Shared constants and helper functions for the WeatherFox ML pipeline.
Used by: shap_generate.py, and any script that loads models or computes confidence.
"""

import pandas as pd
import numpy as np
import pickle
import os

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS_DIR  = "outputs/models"
BOUNDS_FILE = "outputs/models/confidence_bounds.pkl"

FEATURE_COLS = [
    "min_temp °c", "max_temp °c", "rain mm", "humidity %",
    "cloud_cover %", "wind_speed km/h", "wind_direction_numerical",
    "day_of_year", "location_code"
]

TARGET_COLS = [
    "next_min_temp °c", "next_max_temp °c", "next_rain mm",
    "next_humidity %", "next_cloud_cover %", "next_wind_speed km/h",
    "next_wind_direction_numerical"
]

MODEL_FILES = {
    "next_min_temp °c":              "rf_next_min_temp_degc.pkl",
    "next_max_temp °c":              "rf_next_max_temp_degc.pkl",
    "next_rain mm":                  "rf_next_rain_mm.pkl",
    "next_humidity %":               "rf_next_humidity_%.pkl",
    "next_cloud_cover %":            "rf_next_cloud_cover_%.pkl",
    "next_wind_speed km/h":          "rf_next_wind_speed_km_h.pkl",
    "next_wind_direction_numerical": "rf_next_wind_direction_numerical.pkl",
}

# Mapping from model output names to model input names
PREDICTION_TO_FEATURE = {
    "next_min_temp °c":              "min_temp °c",
    "next_max_temp °c":              "max_temp °c",
    "next_rain mm":                  "rain mm",
    "next_humidity %":               "humidity %",
    "next_cloud_cover %":            "cloud_cover %",
    "next_wind_speed km/h":          "wind_speed km/h",
    "next_wind_direction_numerical": "wind_direction_numerical",
}

# ── Functions ─────────────────────────────────────────────────────────────────

def load_model(target):
    """Load a trained Random Forest model by target column name."""
    path = os.path.join(MODELS_DIR, MODEL_FILES[target])
    with open(path, "rb") as f:
        return pickle.load(f)

def load_bounds():
    """Load calibration bounds (lo, hi) per target variable from disk."""
    with open(BOUNDS_FILE, "rb") as f:
        return pickle.load(f)

def tree_std(model, X):
    """
    Compute std dev of predictions across all trees in a Random Forest.
    Returns an array of std devs, one per row.
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    return tree_preds.std(axis=0)

def normalise(std_val, lo, hi):
    """
    Normalise a std dev value to a confidence score (0-100).
    Clips to [lo, hi] bounds, then inverts so high std = low confidence.
    """
    clipped = np.clip(std_val, lo, hi)
    normalised_uncertainty = (clipped - lo) / (hi - lo)
    return round(float((1 - normalised_uncertainty) * 100), 2)

def geometric_mean(values):
    """
    Geometric mean of a list of confidence scores (0-100).
    Clips to 0.01 to avoid log(0) if a score hits exactly 0.

    # DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
    # Used to combine per-variable confidence scores into a single composite.
    # Empirically justified: correlation with actual prediction error was
    # 0.2878 (geometric) vs 0.2712 (minimum) vs 0.2644 (average).
    """
    values = np.clip(values, 0.01, 100)
    return round(float(np.exp(np.mean(np.log(values)))), 2)

def row_locator(df, date, location):
    """
    Finds a single row from the dataset by date and location.
    Returns the full raw row as a DataFrame (1 row, all columns).
    Falls back to first available row for that date if location not found.
    """
    date_rows     = df[df["date"] == date]
    location_rows = date_rows[date_rows["location"].str.lower() == location.lower()]

    if not location_rows.empty:
        return location_rows.iloc[[0]]
    else:
        print(f"  Warning: {location} not found for {date} — using {date_rows.iloc[0]['location']}")
        return date_rows.iloc[[0]]

def input_to_model(raw_row):
    """
    Takes a full raw row (DataFrame, 1 row, all columns) and returns
    only the 9 feature columns in the correct order, ready for model.predict().
    """
    return raw_row[FEATURE_COLS].copy()


def output_to_model(predictions, day_of_year, location_code):
    """
    Takes a predictions dict (7 next_X values), current day_of_year,
    and location_code — returns a 9-column DataFrame ready for the model.
    Increments day_of_year internally.

    predictions: dict e.g. {"next_min_temp °c": 13.2, "next_max_temp °c": 20.1, ...}
    day_of_year: int — current day, will be incremented by 1
    location_code: int — stays constant across the scenario
    """
    row = {PREDICTION_TO_FEATURE[k]: v for k, v in predictions.items()}
    row["day_of_year"]   = day_of_year + 1
    row["location_code"] = location_code

    return pd.DataFrame([row])[FEATURE_COLS]