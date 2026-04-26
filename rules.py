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
    "temp_min": ["cold", "normal", "hot"],
    "temp_max": ["cold", "normal", "hot"],
    "wind_dir": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
}


# ── Clothing rules ─────────────────────────────────────────────────────────────

# Each item has a list of rules. Each rule is a dict of short_name → allowed values.
# Columns omitted from a rule have no restriction — any value is accepted.
# Rules are ordered: most specific/restrictive first.
# First matching rule wins — its columns become the driving categories for text generation.

CLOTHING_RULES = {
    "sunglasses": [
        {"rain": ["light"], "cloud": ["sunny"]},
        {"rain": ["none"], "cloud": ["sunny", "mostly_sunny"]},
    ],
    "baseball_cap": [
        {"rain": ["none", "light"], "cloud": ["sunny", "mostly_sunny"], "wind": ["light", "moderate"], "humidity": ["dry", "normal"], "temp_min": ["normal", "hot"], "temp_max": ["normal", "hot"]},
    ],
    "beanie": [
        {"rain": ["none", "light"], "temp_min": ["cold"], "temp_max": ["cold", "normal"]},
        {"rain": ["none"], "cloud": ["mostly_cloudy", "overcast"], "humidity": ["dry", "normal"], "temp_min": ["normal"], "temp_max": ["normal"]},
    ],
    "scarf": [
        {"temp_min": ["cold"]},
        {"rain": ["none"], "wind": ["moderate", "strong"], "temp_min": ["normal"], "temp_max": ["normal"]},
    ],
    "handfan": [
        {"rain": ["none"], "wind": ["light"], "temp_min": ["hot"], "temp_max": ["hot"]},
        {"rain": ["none"], "cloud": ["sunny", "mostly_sunny"], "wind": ["light"], "temp_min": ["normal", "hot"], "temp_max": ["normal", "hot"]},
        {"rain": ["none"], "cloud": ["mostly_cloudy", "overcast"], "wind": ["light"], "humidity": ["humid"], "temp_min": ["normal", "hot"], "temp_max": ["normal", "hot"]},
    ],
    "welly": [
        {"rain": ["heavy"]},
        {"rain": ["moderate"], "cloud": ["mostly_cloudy", "overcast"]},
    ],
    "umbrella": [
        {"rain": ["moderate"], "wind": ["light"]},
        {"rain": ["light"], "cloud": ["mostly_cloudy", "overcast"], "wind": ["light"]},
    ],
    "raincoat": [
        {"rain": ["heavy"]},
        {"rain": ["light", "moderate"], "wind": ["moderate", "strong"]},
    ],
    "jacket": [
        {"temp_min": ["cold"], "wind": ["moderate", "strong"]},
        {"temp_min": ["cold"], "temp_max": ["cold", "normal"]},
    ],
}