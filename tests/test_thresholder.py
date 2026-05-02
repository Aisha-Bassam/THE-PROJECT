"""
tests/test_thresholder.py
-------------------------
Unit tests for explainability/thresholder.py.
Thresholder is purely deterministic — every branch has an expected output.
"""

import pytest
from explainability.thresholder import threshold


@pytest.mark.parametrize("value,expected", [
    (0.05,  "none"),
    (0.5,   "light"),
    (5.0,   "moderate"),
    (15.0,  "heavy"),
])
def test_rain(value, expected):
    assert threshold("rain", value) == expected


@pytest.mark.parametrize("value,expected", [
    (10.0,  "sunny"),
    (35.0,  "mostly_sunny"),
    (70.0,  "mostly_cloudy"),
    (90.0,  "overcast"),
])
def test_cloud(value, expected):
    assert threshold("cloud", value) == expected


@pytest.mark.parametrize("value,expected", [
    (10.0,  "light"),
    (25.0,  "moderate"),
    (50.0,  "strong"),
])
def test_wind(value, expected):
    assert threshold("wind", value) == expected


@pytest.mark.parametrize("value,expected", [
    (30.0,  "dry"),
    (55.0,  "normal"),
    (80.0,  "humid"),
])
def test_humidity(value, expected):
    assert threshold("humidity", value) == expected


@pytest.mark.parametrize("value,expected", [
    (8.0,   "cold"),
    (16.0,  "normal"),
    (22.0,  "hot"),
])
def test_temp_min(value, expected):
    assert threshold("temp_min", value) == expected


@pytest.mark.parametrize("value,expected", [
    (8.0,   "cold"),
    (16.0,  "normal"),
    (22.0,  "hot"),
])
def test_temp_max(value, expected):
    assert threshold("temp_max", value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.0,   "N"),
    (45.0,  "NE"),
    (90.0,  "E"),
    (135.0, "SE"),
    (180.0, "S"),
    (225.0, "SW"),
    (270.0, "W"),
    (315.0, "NW"),
    (350.0, "N"),   # wraps back to N past 337.5°
])
def test_wind_dir(value, expected):
    assert threshold("wind_dir", value) == expected


def test_invalid_short_name_raises():
    with pytest.raises(ValueError, match="Unrecognised short_name"):
        threshold("pressure", 1013.0)
