"""
ml/confidence.py
----------------
Step 4 of the WeatherFox ML pipeline.

Computes normalisation bounds for the confidence mechanism using std dev
across RF trees, calibrated to the 1st/99th percentile of the 2022 test set.

Output: outputs/models/confidence_bounds.pkl — loaded at inference time by ml/utils.py.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Confidence is derived from the std dev of predictions across all
# decision trees in each Random Forest. A high std dev means the trees
# disagree — the model is uncertain. A low std dev means the trees
# converge — the model is confident.
#
# Normalisation bounds use the 1st and 99th percentile of std devs
# observed across the full test set (2022), rather than min/max.
# This choice makes the 0-100 scale robust to outliers at the extremes.
# The trade-off is that a small number of predictions will be clipped
# to 0 or 100 — this is acceptable and documented as a known limitation.
"""

import pandas as pd
import numpy as np
import pickle
import os

import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "data/uk_weather_data.csv"
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

# Filename mapping — mirrors what train.py saved
MODEL_FILES = {
    "next_min_temp °c":              "rf_next_min_temp_degc.pkl",
    "next_max_temp °c":              "rf_next_max_temp_degc.pkl",
    "next_rain mm":                  "rf_next_rain_mm.pkl",
    "next_humidity %":               "rf_next_humidity_%.pkl",
    "next_cloud_cover %":            "rf_next_cloud_cover_%.pkl",
    "next_wind_speed km/h":          "rf_next_wind_speed_km_h.pkl",
    "next_wind_direction_numerical": "rf_next_wind_direction_numerical.pkl",
}

LO_PERCENTILE = 1    # lower bound for normalisation
HI_PERCENTILE = 99   # upper bound for normalisation

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(target):
    path = os.path.join(MODELS_DIR, MODEL_FILES[target])
    with open(path, "rb") as f:
        return pickle.load(f)

def tree_std(model, X):
    """
    Compute std dev of predictions across all trees in a Random Forest.
    X can be a single row or a full DataFrame — returns an array of std devs,
    one per row.
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # tree_preds shape: (n_estimators, n_samples)
    return tree_preds.std(axis=0)

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"  Total rows: {len(df):,}\n")

test_df = df[df["year"] == 2022].copy()
X_test  = test_df[FEATURE_COLS]

# ── Calibration ───────────────────────────────────────────────────────────────

print("── Calibration (2022 test set) ───────────────────────────────────────")
print(f"  Computing std dev distributions across {len(test_df):,} test rows...")
print(f"  Using {LO_PERCENTILE}th / {HI_PERCENTILE}th percentile as normalisation bounds.\n")

bounds = {}

for target in TARGET_COLS:
    model = load_model(target)
    stds  = tree_std(model, X_test)

    lo = np.percentile(stds, LO_PERCENTILE)
    hi = np.percentile(stds, HI_PERCENTILE)
    bounds[target] = (lo, hi)

    print(f"  {target}")
    print(f"    Std dev range (full): {stds.min():.4f} – {stds.max():.4f}")
    print(f"    Bounds ({LO_PERCENTILE}th / {HI_PERCENTILE}th): {lo:.4f} / {hi:.4f}\n")

# Save bounds for reuse at inference time
with open(BOUNDS_FILE, "wb") as f:
    pickle.dump(bounds, f)
print(f"  Bounds saved to: {BOUNDS_FILE}")

print("\nStep 4 complete.")
