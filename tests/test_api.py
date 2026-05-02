"""
tests/test_api.py
-----------------
Integration tests for the Flask /api/scenario route.

File I/O (load_scenario, load_today_json) and prediction_tracker's disk read
are all mocked so the tests run without pre-existing scenario files on disk.
"""

import pytest
from contextlib import ExitStack
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.backend.app import app as flask_app


_LOAD_SCENARIO  = "app.backend.app.load_scenario"
_LOAD_TODAY     = "app.backend.app.load_today_json"
_TRACKER_READ   = "explainability.prediction_tracker.load_prediction"


@pytest.fixture
def patched(minimal_scenario, high_confidence_today_json):
    """Context manager that patches all file I/O for one /api/scenario call."""
    with ExitStack() as stack:
        stack.enter_context(patch(_LOAD_SCENARIO,  return_value=minimal_scenario))
        stack.enter_context(patch(_LOAD_TODAY,     return_value=high_confidence_today_json))
        stack.enter_context(patch(_TRACKER_READ,   side_effect=FileNotFoundError))
        yield


# ── Status code ───────────────────────────────────────────────────────────────

def test_api_scenario_returns_200(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    assert resp.status_code == 200


# ── Response shape ────────────────────────────────────────────────────────────

def test_response_has_required_top_level_keys(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    assert {"date", "today", "weather", "fox", "text"}.issubset(data.keys())


def test_weather_has_seven_days(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    assert len(data["weather"]) == 7


def test_weather_days_have_required_keys(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    for day in data["weather"]:
        assert "day" in day
        assert "display" in day
        assert "categories" in day


def test_fox_has_layers_list(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    assert isinstance(data["fox"]["layers"], list)
    assert len(data["fox"]["layers"]) > 0


def test_fox_base_in_layers(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    assert "fox_base" in data["fox"]["layers"]


def test_text_has_clothes_weather_emotion(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    assert set(data["text"].keys()) == {"clothes", "weather", "emotion"}


def test_today_display_has_label(client, patched):
    resp = client.get("/api/scenario?date=01072023&location=london")
    data = resp.get_json()
    assert "label" in data["today"]
    assert "icon_label" in data["today"]
