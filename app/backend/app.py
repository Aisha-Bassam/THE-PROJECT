"""
app/backend/app.py
------------------
Flask backend for WeatherFox prototype.

Exposes one route:
    GET /api/scenario

Loads pre-generated scenario and TODAY JSON from disk,
runs weather_pipeline and fox flow, returns everything
the UI needs in one JSON response.

Run from project root:
    python3 app/backend/app.py

# DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
# Flask acts as the orchestration layer — it does not run models or compute
# SHAP. All heavy computation happens at generate_scenario time. Flask only
# reads files and runs lightweight pipeline functions at request time.
"""

import sys
import os

# Add project root to path so all layers are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

from explainability.weather_pipeline import weather_pipeline
from explainability.day_pipeline import day_pipeline
from explainability.clothes_mapper import clothes_mapper
from explainability.emotion_generator import emotion_generator
from explainability.fox_wrapper import fox_wrapper

app = Flask(__name__)
CORS(app)  # allows index.html to call the API from the browser

# ── Config ────────────────────────────────────────────────────────────────────

SCENARIOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs", "scenarios"
)

# Hardcoded for now — will become route parameters later
DATE     = "01072023"
LOCATION = "london"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scenario(date, location):
    """Loads SCENARIO_<date>_<location>.json from disk as a dict of DataFrames."""
    filepath = os.path.join(SCENARIOS_DIR, f"SCENARIO_{date}_{location}.json")
    with open(filepath) as f:
        raw = json.load(f)

    scenario = {"Date": raw["Date"]}
    for key in ["YESTERDAY", "TODAY", "TOMORROW", "FOUR", "FIVE", "SIX", "SEVEN"]:
        scenario[key] = pd.DataFrame([raw[key]])

    return scenario


def load_today_json(date, location):
    """
    Loads TODAY_<date>_<location>.json from disk.
    TODAY's date is one day after YESTERDAY's date.
    Date format in filename is DDMMYYYY.
    """
    # TODAY filename uses the day after yesterday
    # e.g. YESTERDAY = 01072023 → TODAY file = TODAY_02072023_london.json
    yesterday = pd.to_datetime(date, format="%d%m%Y")
    today_str  = (yesterday + pd.Timedelta(days=1)).strftime("%d%m%Y")

    filepath = os.path.join(SCENARIOS_DIR, f"TODAY_{today_str}_{location}.json")
    with open(filepath) as f:
        return json.load(f)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/scenario")
def scenario():
    """
    Returns everything the UI needs for the current scenario:
    - date
    - weather: 7-day table data
    - today: today's display dict (extracted from weather)
    - fox: layer stack + emotion
    """

    # Step 1 — load files from disk
    scene      = load_scenario(DATE, LOCATION)
    today_json = load_today_json(DATE, LOCATION)

    # Step 2 — run weather pipeline (7-day table)
    weather = weather_pipeline(scene)

    # Step 3 — extract today's categories from weather pipeline output
    # TODAY is index 1 (YESTERDAY is index 0)
    today_categories = weather["days"][1]["categories"]

    # Step 4 — fox flow
    outfit  = clothes_mapper(today_categories)
    emotion = emotion_generator(today_json)
    layers  = fox_wrapper(outfit, emotion)

    # Step 5 — package and return
    response = {
        "date":    weather["date"],
        "weather": weather["days"],
        "today":   weather["days"][1]["display"],
        "fox": {
            "layers":  layers,
            "emotion": emotion,
        }
    }

    return jsonify(response)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)