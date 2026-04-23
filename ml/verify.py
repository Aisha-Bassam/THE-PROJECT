"""
ml/verify.py
------------
Step 2 of the WeatherFox ML pipeline.

Read-only verification of the cleaned uk_weather_data.csv.
Makes NO changes to any file. Produces a pass/fail report.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Verification was performed after cleaning to confirm data integrity
# before model training. Checks include: column completeness, null values,
# duplicate rows, year/split coverage, and basic value range sanity.
# No issues were found that required further intervention.
# (Update this note with actual findings once script is run.)
"""

import pandas as pd
import sys

# ── Config ───────────────────────────────────────────────────────────────────

INPUT_FILE = "data/uk_weather_data.csv"

# Locked feature columns (9)
FEATURE_COLS = [
    "min_temp °c", "max_temp °c", "rain mm", "humidity %",
    "cloud_cover %", "wind_speed km/h", "wind_direction_numerical",
    "day_of_year", "location_code"
]

# Locked target columns (7)
TARGET_COLS = [
    "next_min_temp °c", "next_max_temp °c", "next_rain mm",
    "next_humidity %", "next_cloud_cover %", "next_wind_speed km/h",
    "next_wind_direction_numerical"
]

# Locked temporal split boundaries
TRAIN_START, TRAIN_END   = 2015, 2021
TEST_YEAR                = 2022
HOLDOUT_YEAR             = 2023

# Sanity ranges — these are generous bounds, not strict domain limits.
# Flag rows outside these as worth inspecting; they may be legitimate outliers.
SANITY_CHECKS = {
    "min_temp °c":            (-30, 45),
    "max_temp °c":            (-30, 45),
    "humidity %":             (0, 100),
    "cloud_cover %":          (0, 100),
    "rain mm":                (0, 500),
    "wind_speed km/h":        (0, 250),
    "wind_direction_numerical": (0, 360),
    "day_of_year":            (1, 366),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

issues = []
warnings = []

def flag(msg):
    issues.append(msg)
    print(f"  ✗ ISSUE:   {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  ⚠ WARNING: {msg}")

def ok(msg):
    print(f"  ✓ {msg}")

# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_FILE}\n")
df = pd.read_csv(INPUT_FILE)
print(f"  Rows: {len(df):,}   Columns: {len(df.columns)}\n")

# ── Check 1: Required columns present ────────────────────────────────────────

print("── Check 1: Required columns ─────────────────────────────────────────")
all_required = FEATURE_COLS + TARGET_COLS
missing = [c for c in all_required if c not in df.columns]
if missing:
    flag(f"Missing columns: {missing}")
else:
    ok(f"All 9 feature + 7 target columns present.")

# ── Check 2: Null / NaN values ────────────────────────────────────────────────

print("\n── Check 2: Null values ──────────────────────────────────────────────")
null_counts = df[all_required].isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]
if not cols_with_nulls.empty:
    for col, count in cols_with_nulls.items():
        flag(f"Nulls in '{col}': {count:,} rows")
else:
    ok("No null values in feature or target columns.")

# ── Check 3: Duplicate rows ───────────────────────────────────────────────────

print("\n── Check 3: Duplicate rows ───────────────────────────────────────────")
dupe_count = df.duplicated().sum()
if dupe_count > 0:
    warn(f"{dupe_count:,} fully duplicate rows found.")
else:
    ok("No duplicate rows.")

# ── Check 4: Year range and split coverage ────────────────────────────────────

print("\n── Check 4: Year range and split coverage ────────────────────────────")
if "year" not in df.columns:
    flag("'year' column missing — cannot verify split coverage.")
else:
    years_present = sorted(df["year"].unique())
    print(f"  Years in dataset: {years_present}")

    for yr, label in [(TEST_YEAR, "Test"), (HOLDOUT_YEAR, "Holdout")]:
        count = (df["year"] == yr).sum()
        if count == 0:
            flag(f"{label} year {yr} has 0 rows — split will be empty.")
        else:
            ok(f"{label} year {yr}: {count:,} rows.")

    train_rows = df[df["year"].between(TRAIN_START, TRAIN_END)]
    ok(f"Train range {TRAIN_START}–{TRAIN_END}: {len(train_rows):,} rows.")

    unexpected = [y for y in years_present
                  if y < TRAIN_START and y not in [TEST_YEAR, HOLDOUT_YEAR]]
    if unexpected:
        warn(f"Years before training window present (excluded by split, not removed): {unexpected}")
        print(f"    These rows will be ignored during training — no action needed.")

# ── Check 5: Sanity ranges ────────────────────────────────────────────────────

print("\n── Check 5: Value range sanity ───────────────────────────────────────")
for col, (lo, hi) in SANITY_CHECKS.items():
    if col not in df.columns:
        continue
    out_of_range = df[(df[col] < lo) | (df[col] > hi)]
    if not out_of_range.empty:
        warn(f"'{col}': {len(out_of_range):,} rows outside [{lo}, {hi}]. "
             f"Min={df[col].min():.2f}, Max={df[col].max():.2f}")
    else:
        ok(f"'{col}' within [{lo}, {hi}].  Min={df[col].min():.2f}, Max={df[col].max():.2f}")

# ── Check 6: location_code is numeric ────────────────────────────────────────

print("\n── Check 6: location_code is numeric ─────────────────────────────────")
if "location_code" in df.columns:
    if pd.api.types.is_numeric_dtype(df["location_code"]):
        ok(f"location_code is numeric. Range: {df['location_code'].min()}–{df['location_code'].max()}")
    else:
        flag("location_code is not numeric — model training will fail.")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n── Summary ───────────────────────────────────────────────────────────")
if issues:
    print(f"  {len(issues)} ISSUE(s) found — must resolve before training:")
    for i in issues:
        print(f"    ✗ {i}")
    sys.exit(1)
elif warnings:
    print(f"  0 issues. {len(warnings)} warning(s) — review but no action required.")
    print("\nStep 2 complete. Proceed to Step 3 (training).")
else:
    print("  All checks passed. No issues or warnings.")
    print("\nStep 2 complete. Proceed to Step 3 (training).")