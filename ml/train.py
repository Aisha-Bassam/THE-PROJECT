"""
ml/train.py
-----------
Step 3 of the WeatherFox ML pipeline.

Trains 7 Random Forest regressors (one per target variable) on 2015–2021,
evaluates on 2022 test set, compares against persistence baseline,
and saves trained models to outputs/models/.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Seven separate regressors are trained rather than a single multi-output
# model. This allows per-variable SHAP explanations to be generated
# independently, which maps cleanly onto the clothing recommendation logic
# (each variable contributes separately to outfit decisions).
#
# Baseline: persistence model — predicts that tomorrow's value equals
# today's value. This is a standard and meaningful baseline for
# next-day weather forecasting. Any model that cannot beat persistence
# is not useful in practice.
#
# Split is temporal (not random) to respect the time-series nature of
# weather data. Random splitting would leak future information into
# training and inflate performance metrics.
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE  = "data/uk_weather_data.csv"
OUTPUT_DIR  = "outputs/models"

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

# Each target maps to its current-day equivalent for the persistence baseline
# (persistence: predict tomorrow = today's value for the same variable)
PERSISTENCE_MAP = {
    "next_min_temp °c":             "min_temp °c",
    "next_max_temp °c":             "max_temp °c",
    "next_rain mm":                 "rain mm",
    "next_humidity %":              "humidity %",
    "next_cloud_cover %":           "cloud_cover %",
    "next_wind_speed km/h":         "wind_speed km/h",
    "next_wind_direction_numerical":"wind_direction_numerical",
}

TRAIN_START = 2015
TRAIN_END   = 2021
TEST_YEAR   = 2022

# Locked hyperparameters
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42
)

# ── Load and split ─────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"  Total rows: {len(df):,}\n")

train_df = df[df["year"].between(TRAIN_START, TRAIN_END)].copy()
test_df  = df[df["year"] == TEST_YEAR].copy()

print(f"  Train rows (2015–2021): {len(train_df):,}")
print(f"  Test rows  (2022):      {len(test_df):,}\n")

X_train = train_df[FEATURE_COLS]
X_test  = test_df[FEATURE_COLS]

# ── Output directory ───────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Train, evaluate, save ──────────────────────────────────────────────────────

results = []  # collect per-target metrics for summary table

for target in TARGET_COLS:
    print(f"── {target} ───────────────────────────────────────────────────────")

    y_train = train_df[target]
    y_test  = test_df[target]

    # ── Train RF ──
    t0 = time.time()
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # ── RF evaluation ──
    y_pred_rf = model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2  = r2_score(y_test, y_pred_rf)
    print(f"  RF  →  MAE: {rf_mae:.4f}   R²: {rf_r2:.4f}")

    # ── Persistence baseline ──
    current_col = PERSISTENCE_MAP[target]
    y_pred_persistence = test_df[current_col].values
    bl_mae = mean_absolute_error(y_test, y_pred_persistence)
    bl_r2  = r2_score(y_test, y_pred_persistence)
    print(f"  BL  →  MAE: {bl_mae:.4f}   R²: {bl_r2:.4f}")

    # ── Improvement ──
    mae_improvement = ((bl_mae - rf_mae) / bl_mae) * 100
    print(f"  MAE improvement over baseline: {mae_improvement:+.1f}%\n")

    # ── Save model ──
    # Filename derived from target column name, made filesystem-safe
    safe_name = target.replace(" ", "_").replace("/", "_").replace("°", "deg")
    model_path = os.path.join(OUTPUT_DIR, f"rf_{safe_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}\n")

    results.append({
        "target":           target,
        "rf_mae":           rf_mae,
        "rf_r2":            rf_r2,
        "baseline_mae":     bl_mae,
        "baseline_r2":      bl_r2,
        "mae_improvement":  mae_improvement,
    })

# ── Summary table ──────────────────────────────────────────────────────────────

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"{'Target':<35} {'RF MAE':>8} {'RF R²':>7} {'BL MAE':>8} {'BL R²':>7} {'MAE Δ%':>8}")
print("-" * 80)
for r in results:
    print(
        f"{r['target']:<35} "
        f"{r['rf_mae']:>8.4f} "
        f"{r['rf_r2']:>7.4f} "
        f"{r['baseline_mae']:>8.4f} "
        f"{r['baseline_r2']:>7.4f} "
        f"{r['mae_improvement']:>+8.1f}%"
    )
print("=" * 80)

# ── Dissertation notes on expected results ────────────────────────────────────

print("""
# DISSERTATION NOTE: (Ch5 - Results/Evaluation)
# Expected pattern in results:
#   - Temperature variables (min/max): RF should substantially beat baseline.
#     Temperature is autocorrelated but drifts — persistence underestimates
#     seasonal change. Expect R² > 0.8.
#   - Rain: R² will be modest even for RF. Rainfall is inherently noisy and
#     difficult to predict at daily resolution. This is expected and documented
#     as a known limitation, not a model failure.
#   - Cloud cover: Similar to rain — moderate predictability. R² may be low.
#   - Humidity, wind speed: Moderate improvement over baseline expected.
#   - Wind direction: Circular variable treated as linear — known limitation.
#     Performance metrics here should be interpreted cautiously.
# Document actual values in evaluation chapter.
""")

print("Step 3 complete. Models saved to outputs/models/")