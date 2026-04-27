"""
explainability/weather_pipeline.py
-----------------------------------
Processes all 7 days of the scenario dict into display-ready summaries
for the forecast table.

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# weather_pipeline is the table layer — it runs day_pipeline for each day
# in sequence and returns a flat list for the frontend to iterate over.
# No SHAP, no confidence, no text — table only.
"""

from day_pipeline import day_pipeline

DAYS = ["YESTERDAY", "TODAY", "TOMORROW", "FOUR", "FIVE", "SIX", "SEVEN"]


def weather_pipeline(scenario):
    """
    Processes all 7 days of the scenario into display-ready summaries.

    Input:  scenario (dict) — output of generate_seven_predictions()
            keys: Date, YESTERDAY, TODAY, TOMORROW, FOUR, FIVE, SIX, SEVEN

    Output: list of 7 dicts in day order (YESTERDAY → SEVEN), each:
            {
                "day":        day label string (e.g. "TODAY"),
                "display":    day_summary output,
                "categories": {short_name: category_string}
            }
    """
    results = []

    for day in DAYS:
        output = day_pipeline(scenario[day])
        results.append({
            "day":        day,
            "display":    output["display"],
            "categories": output["categories"],
        })

    return {
        "date": scenario["Date"],
        "days": results
    }