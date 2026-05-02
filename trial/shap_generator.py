"""
ml/shap_generate.py
-------------------
Step 5 of the WeatherFox ML pipeline.

Generates SHAP values for a single demo row (2023-07-01, London)
using TreeExplainer, one explainer per model. Combines predictions,
confidence scores, and SHAP values into a single JSON file that
Flask reads and serves directly — no computation at request time.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# SHAP values are pre-generated at build time rather than computed
# live per request. This is an architectural decision: TreeExplainer
# is fast for a single row, but pre-generation guarantees consistent
# outputs for the prototype demonstration and removes any latency
# from the user-facing interface.
#
# TreeExplainer is used because all 7 models are tree-based (Random
# Forest). It computes exact Shapley values rather than approximations,
# which is possible efficiently for tree models. Each model gets its
# own explainer — SHAP values are therefore per-variable, and each
# set of 9 values (one per feature) sums to the difference between
# the model's prediction and its base value (expected output).
#
# The output JSON is the contract between the ML layer and the UI
# layer. Structure is fixed here and must be respected by Flask.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import shap

import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE   = "data/uk_weather_data.csv"
MODELS_DIR   = "outputs/models"
BOUNDS_FILE  = "outputs/models/confidence_bounds.pkl"
OUTPUT_FILE  = "outputs/shap_output.json"

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

DEMO_DATE     = "2023-07-01"
DEMO_LOCATION = "London"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(target):
    path = os.path.join(MODELS_DIR, MODEL_FILES[target])
    with open(path, "rb") as f:
        return pickle.load(f)

def tree_std(model, X):
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    return tree_preds.std(axis=0)

def normalise(std_val, lo, hi):
    clipped = np.clip(std_val, lo, hi)
    normalised_uncertainty = (clipped - lo) / (hi - lo)
    return round(float((1 - normalised_uncertainty) * 100), 2)

def geometric_mean(values):
    values = np.clip(values, 0.01, 100)
    return round(float(np.exp(np.mean(np.log(values)))), 2)

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
holdout_df = df[df["year"] == 2023].copy()

# ── Load bounds ───────────────────────────────────────────────────────────────

with open(BOUNDS_FILE, "rb") as f:
    bounds = pickle.load(f)
print(f"Bounds loaded from: {BOUNDS_FILE}\n")

# ── Get demo row ──────────────────────────────────────────────────────────────

date_rows   = holdout_df[holdout_df["date"] == DEMO_DATE]
london_rows = date_rows[date_rows["location"].str.lower() == DEMO_LOCATION.lower()]
demo_row    = london_rows.iloc[[0]] if not london_rows.empty else date_rows.iloc[[0]]

location_used = demo_row.iloc[0]["location"]
print(f"Demo row: {DEMO_DATE} — {location_used}")
print(f"Input features:\n{demo_row[FEATURE_COLS].to_string(index=False)}\n")

X_demo = demo_row[FEATURE_COLS]

# ── Per-variable: predict, confidence, SHAP ───────────────────────────────────

print("── Generating predictions, confidence, and SHAP values ──────────────\n")

results     = {}
conf_scores = []

for target in TARGET_COLS:
    print(f"  {target}")

    model = load_model(target)

    # Prediction
    pred = float(model.predict(X_demo)[0])

    # Confidence
    std     = float(tree_std(model, X_demo)[0])
    lo, hi  = bounds[target]
    conf    = normalise(std, lo, hi)
    conf_scores.append(conf)

    # SHAP
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_demo)
    # shap_values shape: (1, 9) — one value per feature for the single row
    shap_list   = [round(float(v), 6) for v in shap_values[0]]
    base_value  = round(float(explainer.expected_value), 6)

    print(f"    Prediction:  {pred:.4f}")
    print(f"    Confidence:  {conf}/100")
    print(f"    Base value:  {base_value}")
    print(f"    SHAP values: {shap_list}\n")

    results[target] = {
        "prediction": round(pred, 4),
        "confidence": conf,
        "shap": {
            "base_value":  base_value,
            "values":      shap_list,   # 9 values, aligned to FEATURE_COLS
            "features":    FEATURE_COLS,
        }
    }

composite = geometric_mean(conf_scores)
print(f"Composite confidence (geometric mean): {composite}/100\n")

# ── Assemble output JSON ───────────────────────────────────────────────────────
#
# This is the contract between the ML layer and Flask.
# Flask reads this file directly — nothing is recomputed at request time.
#
# Structure:
#   meta.date           — date of the demo prediction
#   meta.location       — location used
#   meta.composite_confidence — single score (0-100) for fox expression
#   input_features      — dict of feature name -> value (for display)
#   variables           — dict of target -> { prediction, confidence, shap }
#     shap.base_value   — expected model output (intercept)
#     shap.values       — list of 9 SHAP values, one per feature
#     shap.features     — feature names, aligned to shap.values

output = {
    "meta": {
        "date":                 DEMO_DATE,
        "location":             location_used,
        "composite_confidence": composite,
    },
    "input_features": {
        col: round(float(demo_row[col].values[0]), 4)
        for col in FEATURE_COLS
    },
    "variables": results,
}

os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved: {OUTPUT_FILE}")
print("\nStep 5 complete.")