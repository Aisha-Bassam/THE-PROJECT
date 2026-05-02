"""
tests/test_day_pipeline.py
---------------------------
Tests for explainability/day_pipeline.py.

day_pipeline takes a single-row DataFrame and returns display + categories.
Tests are kept at the boundary: correct keys, valid values, no model calls.
"""

import pytest
from explainability.day_pipeline import day_pipeline
from rules import CATEGORIES


REQUIRED_DISPLAY_KEYS = {
    "label", "icon_label",
    "rain_mm", "rain_percent",
    "max_temp", "min_temp",
    "humidity", "wind_speed",
    "wind_dir", "wind_name", "humidity_name",
}


# ── Output structure ──────────────────────────────────────────────────────────

def test_output_has_display_and_categories(sunny_day_df):
    result = day_pipeline(sunny_day_df)
    assert "display" in result
    assert "categories" in result


def test_categories_has_all_seven_short_names(sunny_day_df):
    categories = day_pipeline(sunny_day_df)["categories"]
    assert set(categories.keys()) == set(CATEGORIES.keys())


def test_display_has_all_required_keys(sunny_day_df):
    display = day_pipeline(sunny_day_df)["display"]
    assert REQUIRED_DISPLAY_KEYS.issubset(display.keys())


# ── Category correctness ──────────────────────────────────────────────────────

def test_categories_are_valid_values(sunny_day_df, rainy_cold_day_df):
    for df in (sunny_day_df, rainy_cold_day_df):
        categories = day_pipeline(df)["categories"]
        for short, cat in categories.items():
            assert cat in CATEGORIES[short], (
                f"{short}: got {cat!r}, expected one of {CATEGORIES[short]}"
            )


# ── Display values ────────────────────────────────────────────────────────────

def test_sunny_day_icon_label(sunny_day_df):
    # cloud=10% → "sunny" threshold → icon_label should be "sunny"
    display = day_pipeline(sunny_day_df)["display"]
    assert display["icon_label"] == "sunny"


def test_rainy_cold_day_icon_label(rainy_cold_day_df):
    # rain=5mm (moderate) → rainy icon
    display = day_pipeline(rainy_cold_day_df)["display"]
    assert display["icon_label"] == "rainy"


def test_rain_percent_bounded(sunny_day_df, rainy_cold_day_df):
    for df in (sunny_day_df, rainy_cold_day_df):
        rain_pct = day_pipeline(df)["display"]["rain_percent"]
        assert 0.0 <= rain_pct <= 1.0


def test_rain_percent_zero_for_no_rain(sunny_day_df):
    # sunny_day_df has rain=0.0
    assert day_pipeline(sunny_day_df)["display"]["rain_percent"] == 0.0


def test_temp_values_are_rounded(sunny_day_df):
    display = day_pipeline(sunny_day_df)["display"]
    # Rounded to 1 decimal place
    assert display["max_temp"] == round(display["max_temp"], 1)
    assert display["min_temp"] == round(display["min_temp"], 1)
