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

# Scenario/input column name → short name
# Used by day_pipeline to extract predictions from scenario DataFrames.
# These use raw column names (no "next_" prefix) as saved by generate_scenario.
SCENARIO_COLUMN_TO_SHORT = {
    "rain mm":                    "rain",
    "cloud_cover %":              "cloud",
    "wind_speed km/h":            "wind",
    "humidity %":                 "humidity",
    "min_temp \u00b0c":           "temp_min",
    "max_temp \u00b0c":           "temp_max",
    "wind_direction_numerical":   "wind_dir",
}

# Full model output column name → short name (reverse of above)
COLUMN_TO_SHORT = {v: k for k, v in SHORT_TO_COLUMN.items()}

# DAY_TO_SEASON — sorted by end_day ascending.
# Winter wraps around the year boundary, so it is split into two entries.
# Usage: find the first entry where day_of_year <= end_day.
# If no entry matches (day > 334), it is Winter (end of year).
# 2023 assumption (non-leap year). Day 1 = Jan 1, Day 365 = Dec 31.

DAY_TO_SEASON = [
    (59,  "Winter"),   # Jan 1 – Feb 28
    (151, "Spring"),   # Mar 1 – May 31
    (243, "Summer"),   # Jun 1 – Aug 31
    (334, "Autumn"),   # Sep 1 – Nov 30
    (365, "Winter"),   # Dec 1 – Dec 31
]

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

# ── Thresholds ────────────────────────────────────────────────────────────────

# Numeric boundaries for converting raw predicted values into category labels.
# Used exclusively by thresholder.py — no other component reads these directly.
# Each key maps to a dict of boundary values; thresholder.py interprets them
# as upper bounds (< or <=) depending on the variable.
# Grounded in Met Office, WMO, NWS, and Beaufort citations where available.
# UK maritime climate adjustments applied for temperature and humidity.
# Changing a value here propagates automatically to all downstream components.

THRESHOLDS = {
    # Rain (mm) — Met Office / WMO definitions
    # none: < 0.1mm (dry), light: 0.1–2.5mm, moderate: 2.5–10mm, heavy: 10mm+
    "rain":     {"none": 0.1, "light": 2.5, "moderate": 10.0},

    # Cloud cover (%) — Met Office (sunny/overcast), NWS (middle bands)
    # sunny: ≤20%, mostly_sunny: 21–50%, mostly_cloudy: 51–85%, overcast: 85%+
    "cloud":    {"sunny": 20, "mostly_sunny": 50, "mostly_cloudy": 85},

    # Wind speed (km/h) — Beaufort scale adapted for UK context
    # light: <16 km/h (Beaufort 1–3), moderate: 16–39 (Beaufort 4–5), strong: 39+ (Beaufort 6+)
    "wind":     {"light": 16, "moderate": 39},

    # Humidity (%) — Met Office comfort guidance
    # dry: <40%, normal: 40–70%, humid: 70%+
    # Note: UK humidity rarely drops below 40% — "dry" triggers infrequently
    "humidity": {"dry": 40, "normal": 70},

    # Temperature (°C) — UK-specific, accounts for maritime climate feel
    # cold: ≤13°C, normal: 14–19°C, hot: 20°C+
    # Grounded in UK experience: 17°C can feel hot in still sunny conditions,
    # 13°C can feel cold even without wind chill
    "temp":     {"cold": 13, "normal": 20},
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
        {"rain": ["none", "light"], "cloud": ["sunny", "mostly_sunny"], "wind": ["light", "moderate"], "temp_min": ["normal", "hot"], "temp_max": ["normal", "hot"]},
        {"rain": ["none"], "wind": ["light", "moderate"], "temp_min": ["normal", "hot"], "temp_max": ["hot"]},
    ],

    "beanie": [
        {"temp_min": ["cold"], "temp_max": ["cold", "normal"]},
        {"cloud": ["mostly_cloudy", "overcast"], "wind": ["light"], "humidity": ["dry", "normal"], "temp_min": ["normal"], "temp_max": ["normal"]},
        {"cloud": ["mostly_cloudy", "overcast"], "wind": ["moderate", "strong"], "temp_min": ["normal"], "temp_max": ["normal"]},
    ],

    "scarf": [
        {"temp_min": ["cold"]},
        {"rain": ["light", "moderate", "heavy"], "wind": ["moderate", "strong"], "temp_min": ["normal"], "temp_max": ["normal"]},
        {"wind": ["moderate", "strong"], "cloud": ["mostly_cloudy", "overcast"], "temp_min": ["normal"]},
        {"rain": ["light", "moderate", "heavy"], "cloud": ["mostly_cloudy", "overcast"], "wind": ["light"], "temp_min": ["normal", "hot"]}
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
        {"rain": ["moderate", "heavy"], "wind": ["light"]},
        {"rain": ["light"], "cloud": ["mostly_cloudy", "overcast"], "wind": ["light"]},
    ],

    "raincoat": [
        {"rain": ["heavy"]},
        {"rain": ["light", "moderate"], "wind": ["moderate", "strong"]},
    ],

    "jacket": [
        {"rain": ["none"], "wind": ["moderate", "strong"], "temp_min": ["normal"], "temp_max": ["normal"]},
        {"temp_min": ["cold"], "temp_max": ["cold", "normal"]},
    ],
}

CLOTHING_KEY_COLUMNS = {
    "umbrella":     ["rain", "wind"],
    "raincoat":     ["rain", "wind"],
    "baseball_cap": ["cloud", "temp_max"],
    "beanie":       ["temp_min", "cloud"],
    "scarf":        ["temp_min", "wind"],
    "jacket":       ["temp_min", "temp_max"],
    "welly":        ["rain", "cloud"],
    "sunglasses":   ["cloud", "rain"],
    "handfan":      ["temp_max", "wind"],
}

# Top N SHAP contributors to include in text explanations
SHAP_TOP_N = 2


# ── Weather label priority ─────────────────────────────────────────────────────
# Ordered from most to least dominant.
# Weather Mapper iterates through this list and flags all matching labels.
# Labeller picks the first matching label as dominant.
# Grounded in survey finding: rain is most checked, then actionable signals,
# then cloud as fallback. "Good weather" signals are lowest priority.

WEATHER_LABEL_PRIORITY = [
    "Heavy Rain",
    "Moderate Rain",
    "Hot Day",
    "Cold Day",
    "Light Rain",
    "Strong Wind",
    "Overcast",
    "Mostly Cloudy",
    "Mostly Sunny",
    "Sunny",
    "Moderate Wind",
    "Humid",
    "Dry Air",
]

# ── Weather label rules ────────────────────────────────────────────────────────
# Each label maps to a list of rules (same pattern as CLOTHING_RULES).
# A rule is a dict of short_name → allowed threshold categories.
# First matching rule wins.

WEATHER_LABELS = {
    "Heavy Rain":    [{"rain": ["heavy"]}],
    "Moderate Rain": [{"rain": ["moderate"]}],
    "Hot Day":       [{"temp_min": ["hot"], "temp_max": ["hot"]}],
    "Cold Day":      [{"temp_min": ["cold"], "temp_max": ["cold"]}],
    "Light Rain":    [{"rain": ["light"]}],
    "Strong Wind":   [{"wind": ["strong"]}],
    "Overcast":      [{"cloud": ["overcast"]}],
    "Mostly Cloudy": [{"cloud": ["mostly_cloudy"]}],
    "Mostly Sunny":  [{"cloud": ["mostly_sunny"]}],
    "Sunny":         [{"cloud": ["sunny"]}],
    "Moderate Wind": [{"wind": ["moderate"]}],
    "Humid":         [{"humidity": ["humid"]}],
    "Dry Air":       [{"humidity": ["dry"]}],
}

# ── Label to icon mapping ──────────────────────────────────────────────────────
# Maps weather labels to icon short names.
# Only directly icon-mappable labels are included here.
# Non-icon labels (Cold Day, Hot Day, Humid, Dry Air) fall through to
# secondary labels to find an icon — handled by the Labeller.

LABEL_TO_ICON = {
    "Heavy Rain":    "rainy",
    "Moderate Rain": "rainy",
    "Light Rain":    "light_rain",
    "Strong Wind":   "windy",
    "Overcast":      "overcast",
    "Mostly Cloudy": "mostly_cloudy",
    "Mostly Sunny":  "mostly_sunny",
    "Sunny":         "sunny",
}

# ── Non-icon labels ────────────────────────────────────────────────────────────
# Labels that do not map directly to an icon.
# When one of these is dominant, the Labeller looks at secondary labels
# to find the first icon-mappable one. Cloud always produces a label,
# so there is always a fallback.

NON_ICON_LABELS = ["Cold Day", "Hot Day", "Humid", "Dry Air"]


# ─────────────────────────────────────────────────────────────────────────────
# FOX LAYER ORDER
# Defines the stacking order for fox PNG layers in the UI.
# fox_wrapper iterates this list and includes only active layers.
# Order is visual: earlier = further back, later = closer to viewer.
# ─────────────────────────────────────────────────────────────────────────────

LAYER_ORDER = [
    "fox_base",       # always present — the base fox body with no face
    "jacket",         # base clothing layer — under everything else
    "scarf",          # over jacket
    "raincoat",       # over jacket and scarf — transparent, shows layers beneath
    "confident",      # ─┐
    "neutral",        #  │ exactly one emotion will be active at a time
    "hesitant",       #  │ sits above clothes, below accessories
    "apologetic",     # ─┘
    "baseball_cap",   # before beanie — by design only one hat active at a time
    "beanie",         # over baseball cap if both somehow present (safety fallback)
    "sunglasses",     # order relative to hats does not matter visually
    "wellies",        # ─┐
    "umbrella",       #  │ always on top — never under anything
    "handfan",        # ─┘
]