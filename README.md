# WeatherFox

WeatherFox is an explainable AI weather-forecast prototype. A Random Forest model predicts 7 UK weather variables one day ahead; a reasoning layer translates those predictions into clothing recommendations, confidence expressions, and plain-language explanations, all surfaced through a fox character rendered in a Flask web UI.

---

## Requirements

Python 3.9+. Install all dependencies from the project root:

```bash
pip install -r requirements.txt
```

---

## How to generate scenarios

Scenarios are pre-computed offline and saved to `outputs/scenarios/`. The Flask app reads them at request time — no model inference happens at runtime.

Run from the **project root**:

```bash
python ml/generate_scenario.py
```

This script runs the full ML pipeline for a list of hard-coded dates and locations (all 2023 London dates defined at the bottom of `generate_scenario.py`). For each date it produces:

- `outputs/scenarios/SCENARIO_<DDMMYYYY>_<location>.json` — 7-day forecast table data
- `outputs/scenarios/TODAY_<DDMMYYYY>_<location>.json` — full prediction + confidence + SHAP for the forecast day
- `outputs/scenarios/TOMORROW_<DDMMYYYY>_<location>.json` — same for the following day (used by the prediction tracker)

To add a new scenario, add a date string to the `scenarios` list in `generate_scenario.py` and re-run.

> **Note:** `generate_scenario.py` requires the trained models in `outputs/models/` and the calibration bounds file `outputs/models/confidence_bounds.pkl`. Run the ML pipeline steps in order (clean → verify → train → confidence) before generating new scenarios.

---

## How to start Flask

Run from the **project root**:

```bash
python app/backend/app.py
```

Flask starts on `http://localhost:5000`. The sidebar lists all discovered scenarios. Click a scenario to load it; the fox card, 7-day table, and explanation popups all render from the pre-generated JSON files.

---

## How to run tests

Install pytest if not already installed:

```bash
pip install pytest
```

Run from the **project root**:

```bash
pytest
```

Or to run a specific file:

```bash
pytest tests/test_clothes_mapper.py
```

Tests do not require scenario files on disk — all file I/O is mocked.

---

## Folder structure

```
THE PROJECT/
│
├── app/
│   ├── backend/
│   │   └── app.py              Flask app — two routes: / and /api/scenario
│   ├── static/                 CSS, JS, and fox layer PNG assets
│   └── templates/
│       └── index.html          Single-page UI template
│
├── explainability/
│   ├── clothes_mapper.py       Maps thresholded categories → outfit items
│   ├── confidence_ranker.py    Identifies variables that deviate from composite confidence
│   ├── confidence_tier.py      Combines confidence scores into a tier + fox expression
│   ├── day_pipeline.py         Processes one scenario day → display + categories
│   ├── day_summary.py          Packages a processed day into a UI-ready dict
│   ├── emotion_generator.py    Generates fox expression from confidence + prediction changes
│   ├── fox_wrapper.py          Assembles ordered list of fox PNG layer names
│   ├── prediction_tracker.py   Detects significant changes between TODAY and TOMORROW predictions
│   ├── thresholder.py          Maps raw predicted values → category labels
│   ├── utils.py                Shared file-loading and extraction utilities
│   ├── weather_mapper.py       Maps categories → dominant weather label + icon
│   ├── weather_pipeline.py     Runs day_pipeline across all 7 scenario days
│   └── text/
│       ├── clothes_text.py     Plain-language clothing explanation generator
│       ├── confidence_translator.py  Translates confidence tier into a summary sentence
│       ├── emotion_text.py     Emotion popup text generator
│       ├── prediction_explainer.py   Explains significant forecast changes
│       ├── shap_extractor.py   Extracts and sorts SHAP attribution data from JSON
│       ├── shap_translator.py  Converts SHAP data into readable text snippets
│       └── weather_text.py     Weather insight popup text generator
│
├── ml/
│   ├── clean.py                Step 1 — removes non-UK rows and partial 2024 data
│   ├── verify.py               Step 2 — validates cleaned data integrity
│   ├── train.py                Step 3 — trains 7 RF regressors, saves to outputs/models/
│   ├── confidence.py           Step 4 — calibrates confidence bounds, saves confidence_bounds.pkl
│   ├── generate_scenario.py    Step 5 — generates 7-day scenario JSON files (run to add scenarios)
│   └── utils.py                Shared ML utilities: model loading, bounds, feature transforms
│
├── outputs/
│   ├── models/                 Trained RF model .pkl files + confidence_bounds.pkl
│   └── scenarios/              Pre-generated scenario and prediction JSON files
│
├── tests/
│   ├── conftest.py             Shared fixtures and data-builder helpers
│   ├── test_thresholder.py     Thresholder unit tests
│   ├── test_clothes_mapper.py  Clothes mapper rule tests
│   ├── test_day_pipeline.py    day_pipeline output shape and value tests
│   ├── test_weather_pipeline.py  weather_pipeline structure and ordering tests
│   ├── test_emotion_generator.py  Confidence tier + prediction change tests
│   ├── test_fox_wrapper.py     Layer ordering and content tests
│   └── test_api.py             Flask /api/scenario integration tests
│
├── trial/                      Archived / experimental scripts (not part of the app)
│
├── common.py                   geometric_mean() — shared across ML and explainability layers
├── rules.py                    Central mapping hub: thresholds, clothing rules, weather labels,
│                               layer order, column name mappings
├── requirements.txt            Python dependencies
└── pytest.ini                  pytest configuration (testpaths = tests)
```

---

## ML pipeline

The pipeline is one-time (re-run only when retraining or adding scenarios):

| Step | Script | Output |
|------|--------|--------|
| 1 | `ml/clean.py` | Cleaned CSV |
| 2 | `ml/verify.py` | Integrity report (stdout) |
| 3 | `ml/train.py` | `outputs/models/rf_*.pkl` |
| 4 | `ml/confidence.py` | `outputs/models/confidence_bounds.pkl` |
| 5 | `ml/generate_scenario.py` | `outputs/scenarios/*.json` |

All scripts are run from the **project root**.
