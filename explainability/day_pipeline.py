"""
explainability/day_pipeline.py
------------------------------
Processes a single day's DataFrame row into a display-ready summary.
Used by weather_pipeline (all 7 days) and fox_pipeline (TODAY only).

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# day_pipeline is the shared processing unit for all forecast days.
# It runs threshold → weather_mapper → labeller → day_summary in sequence.
# Returning categories alongside the display dict avoids re-thresholding
# in fox_pipeline, keeping the clothing and SHAP logic grounded in the
# same categorical values used to build the table.
"""

from rules import COLUMN_TO_SHORT
from thresholder import threshold
from weather_mapper import weather_mapper, labeller
from day_summary import day_summary


def day_pipeline(day_df):
    """
    Processes one day's DataFrame row into a display-ready summary.

    Input:  day_df (DataFrame) — one row from the scenario dict
            e.g. scenario["TODAY"], scenario["FOUR"]

    Output: dict with two keys:
            {
                "display":     day_summary output (for table / fox card),
                "categories":  {short_name: category_string} (for fox_pipeline)
            }
    """

    # Step 1 — extract raw predictions from DataFrame using COLUMN_TO_SHORT
    # Skips day_of_year and location_code — not weather variables
    predictions = {
        COLUMN_TO_SHORT[col]: round(float(day_df[col].values[0]), 4)
        for col in day_df.columns
        if col in COLUMN_TO_SHORT
    }

    # Step 2 — threshold each prediction into a category string
    categories = {
        short: threshold(short, val)
        for short, val in predictions.items()
    }

    # Step 3 — map categories to dominant/secondary weather labels
    mapper_output = weather_mapper(categories)

    # Step 4 — produce label and icon_label
    labeller_output = labeller(mapper_output)

    # Step 5 — assemble display-ready summary dict
    display = day_summary(labeller_output, predictions, categories)

    return {
        "display":    display,
        "categories": categories,
    }