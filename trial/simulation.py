"""
trial/simulation.py
--------------------
Scans all rows in the dataset, runs lightweight prediction + confidence,
and identifies candidate scenarios for the user study.

Does NOT generate JSON files or run SHAP.
Imports model utilities directly and replicates confidence + prediction
change logic inline.

Run from project root:
    python3 trial/simulation.py

# DISSERTATION NOTE (Ch7 — Implementation):
# Used to select representative scenarios for the user study.
# Targets cover the key evaluation dimensions: confidence level,
# outfit variety, and prediction change (FR2, FR6).
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from ml.utils import (
    load_model, load_bounds, tree_std, normalise,
    row_locator, input_to_model, output_to_model,
    TARGET_COLS, FEATURE_COLS, PREDICTION_TO_FEATURE
)
from common import geometric_mean
from explainability.thresholder import threshold
from explainability.clothes_mapper import clothes_mapper
from rules import SCENARIO_COLUMN_TO_SHORT

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "uk_weather_data.csv"
)

LOCATION  = "london"

# Only scan 2023 — holdout year, matches our scenario generation setup
YEAR      = 2023

# ── Inline prediction change thresholds (stricter than prediction_tracker)
# Comparing model output to model output — no real data noise
CHANGE_THRESHOLDS = {
    "rain":     0.8,
    "temp_min": 1.5,
    "temp_max": 1.5,
    "wind":     8.0,
    "humidity": 8.0,
    "cloud":    12.0,
    "wind_dir": 35.0,
}

# ── Target definitions ─────────────────────────────────────────────────────────
#
# A — high confidence, no prediction change, any outfit
# B — high confidence, rainy (umbrella in outfit)
# C — low confidence (uncertain fox)
# D — prediction change detected (apologetic case, FR6)

def matches_A(confidence, outfit, changed):
    return confidence > 75 and len(changed) == 0

def matches_B(confidence, outfit, changed):
    return confidence > 75 and "umbrella" in outfit

def matches_C(confidence, outfit, changed):
    return confidence < 40

def matches_D(confidence, outfit, changed):
    return len(changed) > 0

TARGETS = {
    "A — high confidence, stable, any outfit": matches_A,
    "B — high confidence, rainy (umbrella)":   matches_B,
    "C — low confidence (uncertain fox)":      matches_C,
    "D — prediction change (apologetic)":      matches_D,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def circular_distance(a, b):
    return min(abs(a - b), 360 - abs(a - b))


def predict_with_confidence(X_row, bounds):
    """
    Runs all 7 models on X_row, returns predictions dict and composite confidence.

    Input:  X_row (DataFrame, 1 row), bounds (dict from load_bounds())
    Output: (predictions dict, composite_confidence float)
    """
    predictions  = {}
    confidences  = []

    for target in TARGET_COLS:
        model   = load_model(target)
        pred    = float(model.predict(X_row)[0])
        std     = float(tree_std(model, X_row)[0])
        lo, hi  = bounds[target]
        conf    = normalise(std, lo, hi)

        predictions[target] = pred
        confidences.append(conf)

    composite = geometric_mean(confidences)
    return predictions, composite


def detect_changes(prev_preds, curr_preds):
    """
    Compares two prediction dicts using CHANGE_THRESHOLDS.
    prev_preds and curr_preds use short_name keys.

    Output: list of short_names that changed significantly.
    """
    changed = []
    for short in CHANGE_THRESHOLDS:
        if short not in prev_preds or short not in curr_preds:
            continue
        threshold_val = CHANGE_THRESHOLDS[short]
        if short == "wind_dir":
            diff = circular_distance(prev_preds[short], curr_preds[short])
        else:
            diff = abs(curr_preds[short] - prev_preds[short])
        if diff >= threshold_val:
            changed.append(short)
    return changed


def preds_to_short(raw_preds, day_of_year, location_code):
    """
    Converts raw TARGET_COLS predictions dict to short_name keyed dict
    and builds next-day X_row for chaining.
    """
    short_preds = {}
    for target, val in raw_preds.items():
        short = SCENARIO_COLUMN_TO_SHORT.get(
            PREDICTION_TO_FEATURE[target], None
        )
        if short:
            short_preds[short] = round(val, 4)

    X_next = output_to_model(raw_preds, day_of_year, location_code)
    return short_preds, X_next


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df     = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df     = df[df["date"].dt.year == YEAR]
    bounds = load_bounds()

    # Sort by date so consecutive rows are consecutive days
    df = df.sort_values("date").reset_index(drop=True)

    results   = []   # matched candidates
    prev_short_preds = None  # previous day's short predictions (for change detection)

    print(f"Scanning {len(df)} rows for {LOCATION} in {YEAR}...\n")

    for _, raw_row in df.iterrows():
        date     = str(raw_row["date"])[:10]
        location = str(raw_row.get("location", LOCATION))

        if location.lower() != LOCATION.lower():
            prev_short_preds = None
            continue

        # Step 1 — build model input row
        row_df = pd.DataFrame([raw_row])
        X_row  = input_to_model(row_df)

        # Step 2 — predict with confidence
        try:
            raw_preds, composite = predict_with_confidence(X_row, bounds)
        except Exception as e:
            print(f"  Skipping {date}: {e}")
            prev_short_preds = None
            continue

        day_of_year   = int(raw_row["day_of_year"])
        location_code = int(raw_row["location_code"])

        # Step 3 — convert to short names + build next X_row
        short_preds, _ = preds_to_short(raw_preds, day_of_year, location_code)

        # Step 4 — threshold → categories → outfit
        categories = {
            short: threshold(short, val)
            for short, val in short_preds.items()
        }
        outfit = clothes_mapper(categories)

        # Step 5 — detect prediction change vs previous day
        changed = []
        if prev_short_preds is not None:
            changed = detect_changes(prev_short_preds, short_preds)

        # Step 6 — check targets
        matched = [
            label for label, fn in TARGETS.items()
            if fn(composite, outfit, changed)
        ]

        if matched:
            results.append({
                "date":       date,
                "confidence": composite,
                "outfit":     outfit,
                "changed":    changed,
                "matched":    matched,
            })

        prev_short_preds = short_preds

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"Found {len(results)} candidate(s):\n")
    for r in results:
        print(f"  {r['date']}  |  conf: {r['confidence']}  |  outfit: {r['outfit']}")
        print(f"    changed: {r['changed'] or 'none'}")
        for m in r["matched"]:
            print(f"    ✓ {m}")
        print()


if __name__ == "__main__":
    main()