"""
tests/conftest.py
-----------------
Shared fixtures and data-builder helpers for the WeatherFox test suite.

Run from the project root:
    pytest tests/
"""

import sys
import os
import pytest
import pandas as pd

# Add project root to sys.path so all layers are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 9 input feature columns used by the RF models (matches ml/utils.py FEATURE_COLS)
_FEATURE_COLS = [
    "min_temp °c", "max_temp °c", "rain mm", "humidity %",
    "cloud_cover %", "wind_speed km/h", "wind_direction_numerical",
    "day_of_year", "location_code",
]

# Minimal non-zero SHAP values — one per feature, aligned to _FEATURE_COLS
_SHAP_VALUES = [0.10, 0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, 0.00]


# ── Data builders ─────────────────────────────────────────────────────────────
# These are plain functions (not fixtures) so tests can call them directly
# to construct custom inputs without going through pytest fixture injection.

def make_day_df(rain=0.0, cloud=10.0, wind=10.0, humidity=50.0,
                temp_min=18.0, temp_max=24.0, wind_dir=180.0):
    """Single-row DataFrame with scenario column names used by day_pipeline."""
    return pd.DataFrame([{
        "rain mm":                  rain,
        "cloud_cover %":            cloud,
        "wind_speed km/h":          wind,
        "humidity %":               humidity,
        "min_temp °c":         temp_min,
        "max_temp °c":         temp_max,
        "wind_direction_numerical": wind_dir,
    }])


def make_today_json(confidence=80.0, prediction_map=None):
    """
    Minimal TODAY JSON matching the structure produced by generate_scenario.

    confidence:     single float applied to all 7 variables uniformly,
                    or a dict of {full_target_name: float} for per-variable control.
    prediction_map: optional dict of {full_target_name: float} to override
                    the default prediction values.
    """
    default_preds = {
        "next_rain mm":                  0.5,
        "next_cloud_cover %":            30.0,
        "next_wind_speed km/h":          12.0,
        "next_humidity %":               55.0,
        "next_min_temp °c":         15.0,
        "next_max_temp °c":         22.0,
        "next_wind_direction_numerical": 180.0,
    }
    if prediction_map:
        default_preds.update(prediction_map)

    def _conf(target):
        if isinstance(confidence, dict):
            return float(confidence.get(target, 75.0))
        return float(confidence)

    variables = {
        target: {
            "prediction": pred,
            "confidence": _conf(target),
            "shap": {
                "base_value": 1.0,
                "values":     _SHAP_VALUES,
                "features":   _FEATURE_COLS,
            },
        }
        for target, pred in default_preds.items()
    }

    return {
        "meta": {
            "date":                 "2023-07-02",
            "location":             "London",
            "composite_confidence": _conf("next_rain mm"),
        },
        "input_features": {
            "min_temp °c":         15.0,
            "max_temp °c":         22.0,
            "rain mm":                  0.0,
            "humidity %":               55.0,
            "cloud_cover %":            30.0,
            "wind_speed km/h":          12.0,
            "wind_direction_numerical": 180.0,
            "day_of_year":              183.0,
            "location_code":            355.0,
        },
        "variables": variables,
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sunny_day_df():
    """Warm sunny day: no rain, sunny sky, light wind, normal–hot temps."""
    return make_day_df(rain=0.0, cloud=10.0, wind=10.0, humidity=50.0,
                       temp_min=18.0, temp_max=24.0, wind_dir=180.0)


@pytest.fixture
def rainy_cold_day_df():
    """Cold rainy day: moderate rain, overcast, moderate wind, cold temps."""
    return make_day_df(rain=5.0, cloud=90.0, wind=20.0, humidity=75.0,
                       temp_min=8.0, temp_max=11.0, wind_dir=270.0)


@pytest.fixture
def minimal_scenario(sunny_day_df):
    """7-day scenario dict using sunny_day_df for every day slot."""
    df = sunny_day_df
    return {
        "Date":      "01072023",
        "YESTERDAY": df.copy(),
        "TODAY":     df.copy(),
        "TOMORROW":  df.copy(),
        "FOUR":      df.copy(),
        "FIVE":      df.copy(),
        "SIX":       df.copy(),
        "SEVEN":     df.copy(),
    }


@pytest.fixture
def sunny_categories():
    return {
        "rain":     "none",
        "cloud":    "sunny",
        "wind":     "light",
        "humidity": "normal",
        "temp_min": "normal",
        "temp_max": "hot",
        "wind_dir": "S",
    }


@pytest.fixture
def rainy_cold_categories():
    return {
        "rain":     "moderate",
        "cloud":    "overcast",
        "wind":     "moderate",
        "humidity": "humid",
        "temp_min": "cold",
        "temp_max": "cold",
        "wind_dir": "W",
    }


@pytest.fixture
def high_confidence_today_json():
    """All 7 variables at 80% confidence → geometric mean 80 → tier 'high' → 'confident'."""
    return make_today_json(confidence=80.0)


@pytest.fixture
def medium_confidence_today_json():
    """All variables at 55% confidence → tier 'medium' → 'neutral' (absent changes)."""
    return make_today_json(confidence=55.0)


@pytest.fixture
def low_confidence_today_json():
    """All variables at 15% confidence → tier 'very_low' → 'apologetic'."""
    return make_today_json(confidence=15.0)


@pytest.fixture
def diverging_confidence_today_json():
    """rain=90, temp_max=10, rest=50 — rain deviates 'higher', temp_max 'lower'."""
    return make_today_json(confidence={
        "next_rain mm":                  90.0,
        "next_cloud_cover %":            50.0,
        "next_wind_speed km/h":          50.0,
        "next_humidity %":               50.0,
        "next_min_temp °c":         50.0,
        "next_max_temp °c":         10.0,
        "next_wind_direction_numerical": 50.0,
    })


@pytest.fixture
def zero_shap_today_json():
    """today_json where day_of_year (index 7) has 0.0 SHAP value → direction 'no'."""
    j = make_today_json()
    zero_vals = [0.10, 0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.0, 0.0]
    for target in j["variables"]:
        j["variables"][target]["shap"]["values"] = zero_vals
    return j


@pytest.fixture
def cold_overcast_categories():
    """Cold overcast day — dominant: 'Cold Day' (NON_ICON_LABELS), secondary: ['Overcast']."""
    return {
        "rain":     "none",
        "cloud":    "overcast",
        "wind":     "light",
        "humidity": "normal",
        "temp_min": "cold",
        "temp_max": "cold",
        "wind_dir": "N",
    }


@pytest.fixture
def client():
    """Flask test client — shared across test_api.py and test_coverage_extras.py."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.backend.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c
