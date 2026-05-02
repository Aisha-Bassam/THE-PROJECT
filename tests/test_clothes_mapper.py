"""
tests/test_clothes_mapper.py
-----------------------------
Unit tests for explainability/clothes_mapper.py.
Each test constructs a categories dict and asserts which outfit items trigger.
"""

import pytest
from explainability.clothes_mapper import clothes_mapper


def cats(**kwargs):
    """Build a full categories dict with safe defaults, overriding with kwargs."""
    base = {
        "rain":     "none",
        "cloud":    "mostly_sunny",
        "wind":     "light",
        "humidity": "normal",
        "temp_min": "normal",
        "temp_max": "normal",
        "wind_dir": "S",
    }
    base.update(kwargs)
    return base


# ── Sunglasses ────────────────────────────────────────────────────────────────

def test_sunglasses_sunny_no_rain():
    assert "sunglasses" in clothes_mapper(cats(rain="none", cloud="sunny"))


def test_sunglasses_mostly_sunny_no_rain():
    assert "sunglasses" in clothes_mapper(cats(rain="none", cloud="mostly_sunny"))


def test_sunglasses_not_on_overcast():
    assert "sunglasses" not in clothes_mapper(cats(rain="none", cloud="overcast"))


# ── Baseball cap ──────────────────────────────────────────────────────────────

def test_baseball_cap_warm_sunny():
    assert "baseball_cap" in clothes_mapper(cats(
        rain="none", cloud="sunny", wind="light",
        temp_min="normal", temp_max="hot",
    ))


# ── Umbrella ──────────────────────────────────────────────────────────────────

def test_umbrella_light_rain_overcast_light_wind():
    assert "umbrella" in clothes_mapper(cats(
        rain="light", cloud="overcast", wind="light",
    ))


def test_umbrella_moderate_rain_light_wind():
    assert "umbrella" in clothes_mapper(cats(rain="moderate", wind="light"))


def test_no_umbrella_moderate_rain_strong_wind():
    # strong wind → umbrella rule requires light wind; raincoat takes over instead
    assert "umbrella" not in clothes_mapper(cats(rain="moderate", wind="strong"))


# ── Raincoat ──────────────────────────────────────────────────────────────────

def test_raincoat_heavy_rain():
    assert "raincoat" in clothes_mapper(cats(rain="heavy", wind="light"))


def test_raincoat_moderate_rain_strong_wind():
    assert "raincoat" in clothes_mapper(cats(rain="moderate", wind="strong"))


# ── Welly ─────────────────────────────────────────────────────────────────────

def test_welly_heavy_rain():
    assert "welly" in clothes_mapper(cats(rain="heavy"))


def test_welly_moderate_rain_overcast():
    assert "welly" in clothes_mapper(cats(rain="moderate", cloud="overcast"))


# ── Jacket ────────────────────────────────────────────────────────────────────

def test_jacket_cold_temps():
    assert "jacket" in clothes_mapper(cats(temp_min="cold", temp_max="cold"))


def test_no_jacket_hot_temps():
    assert "jacket" not in clothes_mapper(cats(
        rain="none", cloud="overcast", wind="strong",
        temp_min="hot", temp_max="hot",
    ))


# ── Scarf ─────────────────────────────────────────────────────────────────────

def test_scarf_cold_temp_min():
    assert "scarf" in clothes_mapper(cats(temp_min="cold"))


# ── Beanie ────────────────────────────────────────────────────────────────────

def test_beanie_cold_temps():
    assert "beanie" in clothes_mapper(cats(temp_min="cold", temp_max="cold"))


# ── Hand fan ──────────────────────────────────────────────────────────────────

def test_handfan_hot_still_no_rain():
    assert "handfan" in clothes_mapper(cats(
        rain="none", wind="light", temp_min="hot", temp_max="hot",
    ))


# ── Empty outfit ──────────────────────────────────────────────────────────────

def test_empty_outfit_hot_gusty_overcast_no_rain():
    # Verified: hot temps block cold-item rules; strong wind blocks handfan/umbrella;
    # overcast + no rain blocks sunglasses/baseball_cap; no rain blocks raincoat/welly.
    outfit = clothes_mapper(cats(
        rain="none", cloud="overcast", wind="strong",
        humidity="humid", temp_min="hot", temp_max="hot",
    ))
    assert outfit == {}


# ── Output structure ──────────────────────────────────────────────────────────

def test_driving_categories_are_strings():
    outfit = clothes_mapper(cats(rain="light", cloud="overcast", wind="light"))
    for item, driving in outfit.items():
        assert isinstance(driving, dict)
        assert all(isinstance(k, str) and isinstance(v, str)
                   for k, v in driving.items())


def test_driving_categories_are_subset_of_all_categories():
    from rules import CATEGORIES
    outfit = clothes_mapper(cats(rain="moderate", wind="strong", temp_min="cold"))
    for item, driving in outfit.items():
        for col, val in driving.items():
            assert col in CATEGORIES
            assert val in CATEGORIES[col]
