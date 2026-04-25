"""
rules.py
--------
Shared rules and mappings for the WeatherFox reasoning layer.
Imported by: Thresholder, Weather Mapper, Labeller, SHAP Translator,
             Confidence Tier, Text Generator, and any component that
             needs to translate between column names, short names, or categories.
"""

# ── Column name mappings ───────────────────────────────────────────────────────

# Short name → full model output column name
SHORT_TO_COLUMN = {
    "rain":     "next_rain mm",
    "cloud":    "next_cloud_cover %",
    "wind":     "next_wind_speed km/h",
    "humidity": "next_humidity %",
    "temp_min": "next_min_temp °c",
    "temp_max": "next_max_temp °c",
    "wind_dir": "next_wind_direction_numerical",
}

# Full model output column name → short name (reverse of above)
COLUMN_TO_SHORT = {v: k for k, v in SHORT_TO_COLUMN.items()}

# ── Category definitions ───────────────────────────────────────────────────────

# All valid categories per short name
# Used for validation and by any component that needs to know the label space
CATEGORIES = {
    "rain":     ["none", "light", "moderate", "heavy"],
    "cloud":    ["sunny", "mostly_sunny", "mostly_cloudy", "overcast"],
    "wind":     ["light", "moderate", "strong"],
    "humidity": ["dry", "normal", "humid"],
    "temp_min": ["cold", "normal"],
    "temp_max": ["hot", "normal"],
    "wind_dir": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
}