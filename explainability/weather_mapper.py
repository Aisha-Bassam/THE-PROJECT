"""
explainability/weather_mapper.py
---------------------------------
Maps threshold categories to weather labels and picks a dominant label
with an associated icon for display.

Two functions:
  weather_mapper — produces dominant + secondary labels from categories
  labeller       — resolves dominant label to a display icon

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Label priority is grounded in survey findings: rain is the most checked
# signal, followed by actionable conditions (extreme temps, strong wind),
# then cloud as fallback. "Good weather" signals are lowest priority.
# This mirrors how users actually prioritise weather information.
"""

from rules import (
    WEATHER_LABELS,
    WEATHER_LABEL_PRIORITY,
    LABEL_TO_ICON,
    NON_ICON_LABELS,
)

# ── Weather Mapper ─────────────────────────────────────────────────────────────

def weather_mapper(categories):
    """
    Maps threshold categories to all matching weather labels,
    ordered by priority.

    Input:  categories (dict) — short_name → category string
            e.g. {"rain": "heavy", "cloud": "overcast", "wind": "strong", ...}
    Output: dict with dominant label and secondary labels
            e.g. {"dominant": "Heavy Rain", "secondary": ["Overcast", "Strong Wind"]}
    """
    matched = []

    for label in WEATHER_LABEL_PRIORITY:
        rules = WEATHER_LABELS[label]
        for rule in rules:
            if all(categories.get(col) in allowed for col, allowed in rule.items()):
                matched.append(label)
                break

    if not matched:
        # Cloud always produces a label so this should never happen
        return {"dominant": "Mostly Cloudy", "secondary": []}

    return {
        "dominant":  matched[0],
        "secondary": matched[1:],
    }

# ── Labeller ──────────────────────────────────────────────────────────────────

def labeller(mapper_output):
    """
    Resolves dominant label to a display icon.
    If dominant label is not directly icon-mappable, searches secondary
    labels for the first icon-mappable one.

    Input:  mapper_output (dict) — output of weather_mapper()
    Output: dict with label and icon
            e.g. {"label": "Cold Day", "icon": "mostly_cloudy"}
    """
    dominant  = mapper_output["dominant"]
    secondary = mapper_output["secondary"]

    # Direct icon mapping
    if dominant not in NON_ICON_LABELS:
        return {
            "label": dominant,
            "icon_label":  LABEL_TO_ICON[dominant],
        }

    # Search secondary labels for first icon-mappable one
    for label in secondary:
        if label in LABEL_TO_ICON:
            return {
                "label": dominant,
                "icon_label":  LABEL_TO_ICON[label],
            }

    # Fallback — should never reach here since cloud always produces a label
    return {
        "label": dominant,
        "icon_label":  "mostly_cloudy",
    }