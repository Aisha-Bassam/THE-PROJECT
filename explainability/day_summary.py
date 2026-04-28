"""
explainability/day_summary.py
------------------------------
Packages all processed weather data for a single day into a display-ready dict.
Used by Flask for both the main forecast card and the 7-day table.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# day_summary is the bridge between the reasoning layer and the UI layer.
# It takes already-processed data and packages it cleanly — no logic lives here.
# Background colour, emoji icons, and rain dot opacity are handled in the UI layer.
"""

from explainability.utils import rain_percent

def day_summary(labeller_output, predictions, categories):
    """
    Packages processed weather data into a display-ready dict.

    Input:  labeller_output (dict) — {"label": "Heavy Rain", "icon_label": "rainy"}
            predictions (dict)     — short_name → raw predicted value
                                     from extract_predictions()
            categories (dict)      — short_name → category string
                                     from threshold()
    Output: dict ready for UI rendering
    """
    return {
        "label":          labeller_output["label"],
        "icon_label":     labeller_output["icon_label"],
        "rain_mm":        round(predictions["rain"], 2),
        "rain_percent":   rain_percent(predictions["rain"]),
        "max_temp":       round(predictions["temp_max"], 1),
        "min_temp":       round(predictions["temp_min"], 1),
        "humidity":       round(predictions["humidity"], 1),
        "wind_speed":     round(predictions["wind"], 1),
        "wind_dir":       categories["wind_dir"],
        "wind_name":      categories["wind"],
        "humidity_name":  categories["humidity"],
    }