"""
tests/test_fox_wrapper.py
--------------------------
Tests for explainability/fox_wrapper.py.

fox_wrapper assembles an ordered list of PNG layer names from the outfit
and emotion. Tests verify presence, order, and silent-drop of unknown items.
"""

import pytest
from explainability.fox_wrapper import fox_wrapper
from rules import LAYER_ORDER


def wrap(outfit=None, expression="confident"):
    return fox_wrapper(outfit or {}, {"expression": expression})


# ── fox_base ──────────────────────────────────────────────────────────────────

def test_fox_base_always_present():
    assert "fox_base" in wrap()


def test_fox_base_present_with_full_outfit():
    outfit = {"jacket": {}, "scarf": {}, "umbrella": {}}
    assert "fox_base" in wrap(outfit=outfit)


# ── Expression ────────────────────────────────────────────────────────────────

def test_expression_included():
    for expr in ("confident", "neutral", "hesitant", "apologetic"):
        assert expr in wrap(expression=expr)


def test_only_one_expression_per_call():
    layers = wrap(outfit={}, expression="hesitant")
    expressions = [l for l in layers if l in {"confident", "neutral", "hesitant", "apologetic"}]
    assert len(expressions) == 1


# ── Outfit items ──────────────────────────────────────────────────────────────

def test_outfit_items_are_included():
    outfit = {"jacket": {"temp_min": "cold"}, "scarf": {"temp_min": "cold"}}
    layers = wrap(outfit=outfit)
    assert "jacket" in layers
    assert "scarf" in layers


def test_empty_outfit_gives_only_base_and_expression():
    layers = wrap(outfit={}, expression="neutral")
    assert set(layers) == {"fox_base", "neutral"}


# ── Layer ordering ────────────────────────────────────────────────────────────

def test_order_matches_layer_order():
    outfit = {"jacket": {}, "scarf": {}, "raincoat": {}, "umbrella": {}}
    layers = wrap(outfit=outfit, expression="confident")
    positions = [LAYER_ORDER.index(l) for l in layers if l in LAYER_ORDER]
    assert positions == sorted(positions)


def test_fox_base_is_first():
    outfit = {"umbrella": {}, "beanie": {}}
    layers = wrap(outfit=outfit)
    assert layers[0] == "fox_base"


# ── Unknown layers ────────────────────────────────────────────────────────────

def test_unknown_item_silently_dropped():
    outfit = {"magic_cape": {"source": "unknown"}}
    layers = wrap(outfit=outfit)
    assert "magic_cape" not in layers
    assert "fox_base" in layers


def test_none_expression_does_not_crash():
    layers = fox_wrapper({}, {"expression": None})
    assert "fox_base" in layers
    assert not any(e in layers for e in ("confident", "neutral", "hesitant", "apologetic"))
