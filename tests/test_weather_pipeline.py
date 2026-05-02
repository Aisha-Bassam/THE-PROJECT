"""
tests/test_weather_pipeline.py
--------------------------------
Tests for explainability/weather_pipeline.py.

weather_pipeline runs day_pipeline for all 7 scenario days and returns a
structured dict. Tests verify shape, ordering, and date pass-through.
"""

import pytest
from explainability.weather_pipeline import weather_pipeline, DAYS


def test_output_has_date_and_days(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    assert "date" in result
    assert "days" in result


def test_days_count_is_seven(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    assert len(result["days"]) == 7


def test_day_labels_are_in_correct_order(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    labels = [d["day"] for d in result["days"]]
    assert labels == list(DAYS)


def test_each_day_has_required_keys(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    for day in result["days"]:
        assert "day" in day
        assert "display" in day
        assert "categories" in day


def test_date_is_preserved_from_scenario(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    assert result["date"] == minimal_scenario["Date"]


def test_today_is_at_index_one(minimal_scenario):
    # YESTERDAY is index 0, TODAY is index 1 — Flask depends on this offset
    result = weather_pipeline(minimal_scenario)
    assert result["days"][1]["day"] == "TODAY"


def test_display_is_dict_not_none(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    for day in result["days"]:
        assert isinstance(day["display"], dict)
        assert day["display"]  # not empty


def test_categories_is_dict_not_none(minimal_scenario):
    result = weather_pipeline(minimal_scenario)
    for day in result["days"]:
        assert isinstance(day["categories"], dict)
        assert day["categories"]
