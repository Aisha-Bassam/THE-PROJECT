"""
ml/confidence.py
----------------
Step 4 of the WeatherFox ML pipeline.

Implements the confidence mechanism using std dev across RF trees,
normalised to 0-100 using 1st/99th percentile bounds from the 2022
test set, then combined via geometric mean across all 7 variables.

Two phases:
  1. Calibration  — compute normalisation bounds from test set std devs
  2. Inference    — apply to a single holdout row (2023-07-01)

Output is a structured dict shaped for direct SHAP consumption.

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
#
# Geometric mean is used to combine per-variable confidence scores
# into a single composite score. Empirically justified: correlation
# with actual prediction error was 0.2878 (geometric) vs 0.2712
# (minimum) vs 0.2644 (average) on the test set.
#
# July 1st 2023 was chosen as the demonstration row. It sits in the
# middle of the holdout year, making it a representative summer
# prediction. The choice is a demonstration decision, not a
# methodological one — any 2023 row would be equally valid as
# held-out data.
"""

import pandas as pd
import numpy as np
import pickle
import os

import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE    = "data/uk_weather_data.csv"
MODELS_DIR    = "outputs/models"
BOUNDS_FILE   = "outputs/models/confidence_bounds.pkl"

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

DEMO_DATE     = "2023-07-01"
DEMO_LOCATION = "London"     # fallback if London not found: first available 2023-07-01 row

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

def normalise(std_vals, lo, hi):
    """
    Normalise std dev values to 0-100 using precomputed percentile bounds.
    Values outside [lo, hi] are clipped.
    Returns confidence (inverted): high std = low confidence.
    """
    clipped = np.clip(std_vals, lo, hi)
    normalised_uncertainty = (clipped - lo) / (hi - lo)   # 0 = certain, 1 = uncertain
    confidence = (1 - normalised_uncertainty) * 100        # 0 = uncertain, 100 = certain
    return confidence

def geometric_mean(values):
    """
    Geometric mean of a list of values.
    All values must be > 0 (confidence scores are 0-100; clipped to 0.01
    to avoid log(0) if a score hits exactly 0).
    """
    values = np.clip(values, 0.01, 100)
    return np.exp(np.mean(np.log(values)))

# ── Load data ─────────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"  Total rows: {len(df):,}\n")

test_df    = df[df["year"] == 2022].copy()
holdout_df = df[df["year"] == 2023].copy()

X_test = test_df[FEATURE_COLS]

# ── Phase 1: Calibration ──────────────────────────────────────────────────────

print("── Phase 1: Calibration (2022 test set) ─────────────────────────────")
print(f"  Computing std dev distributions across {len(test_df):,} test rows...")
print(f"  Using {LO_PERCENTILE}th / {HI_PERCENTILE}th percentile as normalisation bounds.\n")

bounds = {}

for target in TARGET_COLS:
    model = load_model(target)
    stds = tree_std(model, X_test)

    lo = np.percentile(stds, LO_PERCENTILE)
    hi = np.percentile(stds, HI_PERCENTILE)
    bounds[target] = (lo, hi)

    print(f"  {target}")
    print(f"    Std dev range (full): {stds.min():.4f} – {stds.max():.4f}")
    print(f"    Bounds ({LO_PERCENTILE}th / {HI_PERCENTILE}th): {lo:.4f} / {hi:.4f}\n")

# Save bounds for reuse at inference time
with open(BOUNDS_FILE, "wb") as f:
    pickle.dump(bounds, f)
print(f"  Bounds saved to: {BOUNDS_FILE}\n")

# ── Phase 2: Inference on 2023-07-01 ─────────────────────────────────────────

print(f"── Phase 2: Inference on holdout row ({DEMO_DATE}) ──────────────────")

# Find the demo row — prefer London, fall back to first available
if "date" in holdout_df.columns:
    date_rows = holdout_df[holdout_df["date"] == DEMO_DATE]
else:
    # If date column missing, filter by day_of_year (July 1 = day 182)
    date_rows = holdout_df[holdout_df["day_of_year"] == 182]

if date_rows.empty:
    raise ValueError(f"No rows found for {DEMO_DATE} in holdout set.")

london_rows = date_rows[date_rows["location"].str.lower() == DEMO_LOCATION.lower()]
if not london_rows.empty:
    demo_row = london_rows.iloc[[0]]
    print(f"  Location: {DEMO_LOCATION}")
else:
    demo_row = date_rows.iloc[[0]]
    print(f"  London not found for this date — using: {date_rows.iloc[0]['location']}")

print(f"  Date: {DEMO_DATE}")
print(f"  Input features:\n{demo_row[FEATURE_COLS].to_string(index=False)}\n")

X_demo = demo_row[FEATURE_COLS]

# Run through all 7 models
per_variable = {}
confidence_scores = []

for target in TARGET_COLS:
    model   = load_model(target)
    pred    = model.predict(X_demo)[0]
    std     = tree_std(model, X_demo)[0]
    lo, hi  = bounds[target]
    conf    = normalise(np.array([std]), lo, hi)[0]

    per_variable[target] = {
        "prediction":  round(pred, 4),
        "tree_std":    round(std, 4),
        "confidence":  round(conf, 2),
    }
    confidence_scores.append(conf)

    print(f"  {target}")
    print(f"    Prediction:  {pred:.4f}")
    print(f"    Tree std:    {std:.4f}")
    print(f"    Confidence:  {conf:.2f}/100\n")

composite_confidence = geometric_mean(confidence_scores)
print(f"  Composite confidence (geometric mean): {composite_confidence:.2f}/100\n")

# ── Output dict (SHAP-ready) ──────────────────────────────────────────────────
#
# Structured so the SHAP step can consume it directly:
#   output["input_df"]    — the input row as a DataFrame (SHAP expects this)
#   output["predictions"] — dict of target -> predicted value
#   output["confidence"]  — dict of target -> per-variable confidence score
#   output["composite"]   — single composite confidence score (0-100)
#   output["feature_cols"]— feature column list (for SHAP explainer alignment)

output = {
    "input_df":    X_demo,                                          # DataFrame, shape (1, 9)
    "predictions": {t: v["prediction"] for t, v in per_variable.items()},
    "confidence":  {t: v["confidence"] for t, v in per_variable.items()},
    "composite":   round(composite_confidence, 2),
    "feature_cols": FEATURE_COLS,
}

print("── Output summary ────────────────────────────────────────────────────")
print(f"  Composite confidence: {output['composite']}/100")
print(f"  Predictions:")
for t, v in output["predictions"].items():
    print(f"    {t}: {v}")
print(f"  Per-variable confidence:")
for t, v in output["confidence"].items():
    print(f"    {t}: {v}/100")
print(f"\n  input_df shape: {output['input_df'].shape}  (ready for SHAP)")

print("\nStep 4 complete.")