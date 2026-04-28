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