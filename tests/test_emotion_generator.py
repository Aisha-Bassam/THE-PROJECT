"""
tests/test_emotion_generator.py
---------------------------------
Tests for explainability/emotion_generator.py.

emotion_generator calls prediction_tracker which reads TOMORROW JSON from disk.
All tests mock explainability.prediction_tracker.load_prediction to avoid
file-system dependencies and to control change detection precisely.
"""

import pytest
from unittest.mock import patch
from explainability.emotion_generator import emotion_generator

_PATCH = "explainability.prediction_tracker.load_prediction"
_NO_FILE = {"side_effect": FileNotFoundError}

VALID_EXPRESSIONS = {"confident", "neutral", "hesitant", "apologetic"}


# ── Output structure ──────────────────────────────────────────────────────────

def test_output_has_required_keys(high_confidence_today_json):
    with patch(_PATCH, **_NO_FILE):
        result = emotion_generator(high_confidence_today_json)
    assert set(result.keys()) == {"expression", "change", "changed", "changes"}


def test_expression_is_always_valid(
    high_confidence_today_json, medium_confidence_today_json, low_confidence_today_json
):
    for today_json in (high_confidence_today_json, medium_confidence_today_json,
                       low_confidence_today_json):
        with patch(_PATCH, **_NO_FILE):
            result = emotion_generator(today_json)
        assert result["expression"] in VALID_EXPRESSIONS


# ── Confidence → expression mapping ──────────────────────────────────────────

def test_high_confidence_gives_confident(high_confidence_today_json):
    # All variables at 80% → geometric mean 80 → tier 'high' → 'confident'
    with patch(_PATCH, **_NO_FILE):
        result = emotion_generator(high_confidence_today_json)
    assert result["expression"] == "confident"


def test_medium_confidence_gives_neutral(medium_confidence_today_json):
    # All variables at 55% → geometric mean 55 → tier 'medium' → 'neutral'
    with patch(_PATCH, **_NO_FILE):
        result = emotion_generator(medium_confidence_today_json)
    assert result["expression"] == "neutral"


def test_low_confidence_gives_apologetic(low_confidence_today_json):
    # All variables at 15% → geometric mean 15 → tier 'very_low' → 'apologetic'
    with patch(_PATCH, **_NO_FILE):
        result = emotion_generator(low_confidence_today_json)
    assert result["expression"] == "apologetic"


# ── No-change path ────────────────────────────────────────────────────────────

def test_no_change_when_tomorrow_file_missing(high_confidence_today_json):
    with patch(_PATCH, **_NO_FILE):
        result = emotion_generator(high_confidence_today_json)
    assert result["change"] is False
    assert result["changed"] == []
    assert result["changes"] == {}


# ── Apologetic override on significant prediction changes ─────────────────────

def test_apologetic_override_on_two_or_more_changes(medium_confidence_today_json):
    """
    Medium confidence would give 'neutral', but two large prediction changes
    (rain +8.5mm, temp_max -17°C) should override expression to 'apologetic'.
    """
    # TOMORROW json — these are yesterday's predictions for today.
    # prediction_tracker compares: old (tomorrow_json) vs new (medium_confidence_today_json).
    # rain:     |0.5 - 9.0|  = 8.5  > threshold 2.0  → change
    # temp_max: |22.0 - 5.0| = 17.0 > threshold 5.0  → change
    tomorrow_json = {
        "meta": {"date": "2023-07-02", "location": "London"},
        "variables": {
            "next_rain mm":                  {"prediction": 9.0,   "confidence": 55.0},
            "next_cloud_cover %":            {"prediction": 30.0,  "confidence": 55.0},
            "next_wind_speed km/h":          {"prediction": 12.0,  "confidence": 55.0},
            "next_humidity %":               {"prediction": 55.0,  "confidence": 55.0},
            "next_min_temp °c":         {"prediction": 15.0,  "confidence": 55.0},
            "next_max_temp °c":         {"prediction": 5.0,   "confidence": 55.0},
            "next_wind_direction_numerical": {"prediction": 180.0, "confidence": 55.0},
        },
    }
    with patch(_PATCH, return_value=tomorrow_json):
        result = emotion_generator(medium_confidence_today_json)

    assert result["expression"] == "apologetic"
    assert result["change"] is True
    assert len(result["changed"]) >= 2
    assert "rain" in result["changed"]
    assert "temp_max" in result["changed"]
