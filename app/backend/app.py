# app/backend/app.py
"""
app/backend/app.py
------------------
Flask backend for WeatherFox prototype.

Exposes two routes:
    GET /              — serves the main UI (index.html)
    GET /api/scenario  — returns all data needed to render a scenario

Loads pre-generated scenario and TODAY JSON from disk,
runs weather_pipeline and fox flow, returns everything
the UI needs in one JSON response.

Run from project root:
    python app/backend/app.py

# DISSERTATION NOTE (Ch7 — Implementation):
# Flask acts as the orchestration layer — it does not run models or compute
# SHAP. All heavy computation happens at generate_scenario time (offline).
# At request time Flask only reads files, runs lightweight pipeline
# functions, and returns a structured JSON response to the frontend.
# This separation keeps the prototype fast and stateless.
"""

import sys
import os

# Add project root to path so all layers are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import glob
import pandas as pd
from flask import Flask, jsonify, render_template, request

from explainability.weather_pipeline import weather_pipeline
from explainability.clothes_mapper import clothes_mapper
from explainability.emotion_generator import emotion_generator
from explainability.fox_wrapper import fox_wrapper
from explainability.text.clothes_text import clothes_text
from explainability.text.weather_text import weather_text
from explainability.text.emotion_text import emotion_text

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENARIOS_DIR = os.path.join(ROOT_DIR, "outputs", "scenarios")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "app", "templates")
STATIC_DIR    = os.path.join(ROOT_DIR, "app", "static")

# ── Flask init ─────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# ── Scenario discovery ─────────────────────────────────────────────────────────
# DISSERTATION NOTE (Ch7 — Implementation):
# Scenarios are pre-generated offline. Flask discovers them at startup by
# globbing the scenarios directory — no hardcoded list required.
# Adding a new scenario only requires dropping a file in the folder
# and restarting Flask.

def discover_scenarios():
    """
    Scans SCENARIOS_DIR for SCENARIO_*.json files.

    Output: sorted list of dicts — [{ date, location, label }, ...]
            date     — DDMMYYYY string   e.g. "01072023"
            location — lowercase string  e.g. "london"
            label    — display string    e.g. "2 Jul 2023 · London"
    """
    pattern   = os.path.join(SCENARIOS_DIR, "SCENARIO_*.json")
    scenarios = []

    for path in sorted(glob.glob(pattern)):
        name  = os.path.basename(path).replace(".json", "")
        parts = name.split("_")

        if len(parts) != 3:
            continue

        date     = parts[1]
        location = parts[2]

        parsed = pd.to_datetime(date, format="%d%m%Y")
        next_day = parsed + pd.Timedelta(days=1)          # add one day
        label = next_day.strftime("%-d %b %Y") + " · " + location.capitalize()

        scenarios.append({"date": date, "location": location, "label": label})

    # Sort by date (chronologically)
    scenarios.sort(key=lambda x: pd.to_datetime(x['date'], format="%d%m%Y"))
    return scenarios


AVAILABLE_SCENARIOS = discover_scenarios()

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_scenario(date, location):
    """
    Loads SCENARIO_<date>_<location>.json from disk as a dict of DataFrames.

    Input:  date (DDMMYYYY), location (lowercase string)
    Output: dict — { Date, YESTERDAY, TODAY, TOMORROW, FOUR, FIVE, SIX, SEVEN }
            each day value is a single-row DataFrame
    """
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
    Input date is YESTERDAY (DDMMYYYY) — file uses the following day.

    e.g. date = "01072023" → loads TODAY_02072023_london.json
    """
    yesterday = pd.to_datetime(date, format="%d%m%Y")
    today_str  = (yesterday + pd.Timedelta(days=1)).strftime("%d%m%Y")

    filepath = os.path.join(SCENARIOS_DIR, f"TODAY_{today_str}_{location}.json")
    with open(filepath) as f:
        return json.load(f)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serves the main page. Passes available scenarios to the sidebar."""
    return render_template("index.html", scenarios=AVAILABLE_SCENARIOS)


@app.route("/api/scenario")
def api_scenario():
    """
    Returns all data needed to render one scenario.

    Query params: date (DDMMYYYY), location (lowercase string)

    Response shape:
        date    — DDMMYYYY string (YESTERDAY's date, anchors the scenario)
        today   — display-ready dict for the TODAY card
        weather — list of 7 day dicts ({ day, display, categories })
        fox     — { layers: [...] }
        text    — { clothes: {...}, weather: {...}, emotion: {...} }

    # DISSERTATION NOTE (Ch7 — Implementation):
    # All ML and SHAP work is done offline. At request time Flask only runs
    # lightweight display logic: pipeline reshaping, clothes/emotion mapping,
    # and text generation. No model inference happens here.
    """
    date     = request.args.get("date")
    location = request.args.get("location")

    # Step 1 — load pre-generated scenario files from disk
    scene      = load_scenario(date, location)
    today_json = load_today_json(date, location)

    # Step 2 — run weather pipeline (all 7 days → display + categories)
    weather = weather_pipeline(scene)

    # Step 3 — extract TODAY's display and categories (index 1 — YESTERDAY is index 0)
    today_display    = weather["days"][1]["display"]
    today_categories = weather["days"][1]["categories"]

    # Step 4 — fox flow
    outfit  = clothes_mapper(today_categories)
    emotion = emotion_generator(today_json)
    layers  = fox_wrapper(outfit, emotion)

    # Step 5 — text generation
    text = {
        "clothes": clothes_text(outfit, today_json),
        "weather": weather_text(today_json, today_categories, today_display["icon_label"]),
        "emotion": emotion_text(today_json, emotion, today_display["label"]),
    }

    return jsonify({
        "date":    weather["date"],
        "today":   today_display,
        "weather": weather["days"],
        "fox":     {"layers": layers},
        "text":    text,
    })

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)