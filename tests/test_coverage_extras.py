"""
tests/test_coverage_extras.py
------------------------------
Targeted tests to close remaining coverage gaps:
- explainability/confidence_tier.py — empty columns ValueError
- explainability/weather_mapper.py  — no-match fallback, NON_ICON_LABELS paths
- explainability/utils.py           — load_prediction with real file / missing file
- app/backend/app.py                — load_scenario, load_today_json, index() route
"""

import pytest
import pandas as pd

from explainability.confidence_tier import confidence_tier
from explainability.weather_mapper import weather_mapper, labeller
from explainability.utils import load_prediction


# ── confidence_tier ───────────────────────────────────────────────────────────

def test_confidence_tier_empty_columns_raises(high_confidence_today_json):
    with pytest.raises(ValueError, match="columns must not be empty"):
        confidence_tier([], high_confidence_today_json)


# ── weather_mapper ────────────────────────────────────────────────────────────

def test_weather_mapper_no_rule_match_falls_back_to_mostly_cloudy():
    # All categories are invalid strings → no WEATHER_LABELS rule matches
    # → fallback branch returns {"dominant": "Mostly Cloudy", "secondary": []}
    bad_cats = {k: "???" for k in ("rain", "cloud", "wind", "humidity",
                                    "temp_min", "temp_max", "wind_dir")}
    result = weather_mapper(bad_cats)
    assert result["dominant"] == "Mostly Cloudy"
    assert result["secondary"] == []


# ── labeller — NON_ICON_LABELS paths ─────────────────────────────────────────

def test_labeller_non_icon_dominant_uses_secondary_icon(cold_overcast_categories):
    # "Cold Day" is NON_ICON_LABELS; "Overcast" is in secondary → icon from secondary
    mapper_output = weather_mapper(cold_overcast_categories)
    result = labeller(mapper_output)
    assert result["label"] == "Cold Day"
    assert result["icon_label"] == "overcast"


def test_labeller_non_icon_dominant_no_icon_secondary_fallback():
    # All secondary labels are also NON_ICON_LABELS → fallback to "mostly_cloudy"
    mapper_output = {"dominant": "Cold Day", "secondary": ["Humid", "Dry Air"]}
    result = labeller(mapper_output)
    assert result["label"] == "Cold Day"
    assert result["icon_label"] == "mostly_cloudy"


# ── utils.load_prediction ─────────────────────────────────────────────────────

def test_load_prediction_reads_existing_file():
    result = load_prediction("2023-07-02", "London", "TODAY")
    assert "meta" in result
    assert "variables" in result
    assert result["meta"]["date"] == "2023-07-02"
    assert result["meta"]["location"] == "London"


def test_load_prediction_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_prediction("2000-01-01", "Nowhere", "TODAY")


# ── app.py loaders and index route ────────────────────────────────────────────

def test_load_scenario_returns_dataframe_dict():
    from app.backend.app import load_scenario
    result = load_scenario("01072023", "london")
    assert isinstance(result, dict)
    assert "Date" in result
    for key in ("YESTERDAY", "TODAY", "TOMORROW", "FOUR", "FIVE", "SIX", "SEVEN"):
        assert isinstance(result[key], pd.DataFrame)
        assert len(result[key]) == 1


def test_load_today_json_returns_prediction_structure():
    from app.backend.app import load_today_json
    # scenario date "01072023" (YESTERDAY) → loads TODAY_02072023_london.json
    result = load_today_json("01072023", "london")
    assert "meta" in result
    assert "variables" in result
    assert len(result["variables"]) == 7


def test_index_route_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_discover_scenarios_skips_malformed_filenames():
    from unittest.mock import patch
    from app.backend.app import discover_scenarios
    with patch("app.backend.app.glob.glob") as mock_glob:
        mock_glob.return_value = [
            "/path/SCENARIO_01072023_london.json",  # valid: 3 parts
            "/path/SCENARIO_MALFORMED.json",         # invalid: 2 parts → continue
        ]
        result = discover_scenarios()
    assert len(result) == 1
    assert result[0]["date"] == "01072023"
