"""
ml/clean.py
-----------
Step 1 of the WeatherFox ML pipeline.

Loads uk_weather_data.csv, removes non-UK locations (Cork, Dublin)
and partial-year 2024 data, then saves the cleaned file in place.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Cleaning decisions are documented here:
#   - Cork and Dublin were present in the source dataset but are not
#     UK locations. Including them would introduce geographic distribution
#     bias into a model intended to predict UK weather.
#   - 2024 data is a partial year (incomplete seasonal cycle), which
#     would skew temporal patterns learned by the model. Removed entirely.
#   - The original source file (all_weather_data.csv) is preserved and
#     never touched. uk_weather_data.csv is the working file.
"""

import pandas as pd
import sys
import os

# ── Config ──────────────────────────────────────────────────────────────────

INPUT_FILE = "data/uk_weather_data.csv"   # path relative to where you run this script
                                      # adjust if running from project root, e.g.
                                      # "data/uk_weather_data.csv"

NON_UK_LOCATIONS = ["Cork", "Dublin"]  # exact strings as they appear in 'location' column
PARTIAL_YEAR     = 2024                # year to drop (incomplete seasonal data)

# ── Load ─────────────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_FILE}")

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: '{INPUT_FILE}' not found. Check your working directory or update INPUT_FILE.")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)

print(f"  Rows loaded:   {len(df):,}")
print(f"  Columns found: {list(df.columns)}\n")

original_count = len(df)

# ── Validate expected columns exist ──────────────────────────────────────────

# We need a 'location' column (string) to filter Cork/Dublin.
if "location" not in df.columns:
    print("ERROR: 'location' column not found.")
    print("  Available columns:", list(df.columns))
    print("  Update the script to match the actual column name.")
    sys.exit(1)

# We need a 'year' column (or will try to parse from 'date') to filter 2024.
if "year" not in df.columns:
    if "date" in df.columns:
        print("  'year' column not found — extracting from 'date' column...")
        df["year"] = pd.to_datetime(df["date"]).dt.year
        print("  Extracted 'year' successfully.\n")
    else:
        print("ERROR: Neither 'year' nor 'date' column found.")
        print("  Cannot determine which rows are 2024. Update the script.")
        sys.exit(1)

# ── Report what we're about to remove ────────────────────────────────────────

print("── Pre-clean summary ─────────────────────────────────────────────────")

# Locations present in dataset
print("  Unique locations:", sorted(df["location"].unique()))

# Cork / Dublin rows
location_mask = df["location"].isin(NON_UK_LOCATIONS)
print(f"  Rows to remove (Cork/Dublin): {location_mask.sum():,}")

# 2024 rows
year_mask = df["year"] == PARTIAL_YEAR
print(f"  Rows to remove (year={PARTIAL_YEAR}): {year_mask.sum():,}")

# Overlap (rows that are both Cork/Dublin AND 2024 — would be double-counted naively)
overlap = (location_mask & year_mask).sum()
combined_mask = location_mask | year_mask
print(f"  Overlap (Cork/Dublin AND 2024): {overlap:,}")
print(f"  Total rows to remove (deduplicated): {combined_mask.sum():,}")
print()

# ── Apply cleaning ────────────────────────────────────────────────────────────

df_clean = df[~combined_mask].copy()

print("── Post-clean summary ────────────────────────────────────────────────")
print(f"  Rows before: {original_count:,}")
print(f"  Rows after:  {len(df_clean):,}")
print(f"  Removed:     {original_count - len(df_clean):,}")
print()

# Confirm no Cork/Dublin remain
remaining_non_uk = df_clean[df_clean["location"].isin(NON_UK_LOCATIONS)]
if not remaining_non_uk.empty:
    print(f"WARNING: {len(remaining_non_uk)} Cork/Dublin rows still present — check filter logic.")
else:
    print("  ✓ No Cork/Dublin rows remain.")

# Confirm no 2024 data remains
remaining_2024 = df_clean[df_clean["year"] == PARTIAL_YEAR]
if not remaining_2024.empty:
    print(f"WARNING: {len(remaining_2024)} rows from 2024 still present — check filter logic.")
else:
    print(f"  ✓ No {PARTIAL_YEAR} rows remain.")

# Confirm year range looks correct
print(f"  Year range in cleaned data: {df_clean['year'].min()} – {df_clean['year'].max()}")
print(f"  Unique locations remaining: {sorted(df_clean['location'].unique())}")
print()

# ── Save ─────────────────────────────────────────────────────────────────────

# Drop the 'year' column if we synthesised it (wasn't in the original file)
# to avoid adding an unexpected column to the output
original_columns = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()
if "year" not in original_columns and "year" in df_clean.columns:
    df_clean = df_clean.drop(columns=["year"])
    print("  Note: Dropped synthesised 'year' column (was not in original file).")

df_clean.to_csv(INPUT_FILE, index=False)
print(f"  Saved cleaned data to: {INPUT_FILE}  ({len(df_clean):,} rows)")
print("\nDone. Step 1 complete.")