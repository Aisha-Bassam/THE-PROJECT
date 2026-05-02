"""
tests/test_text_layer.py
-------------------------
Tests for the explainability/text/ layer:
shap_extractor, shap_translator, confidence_translator, confidence_ranker,
clothes_text, prediction_explainer, emotion_text, weather_text.
"""

import pytest
from explainability.text.shap_extractor import shap_extractor
from explainability.text.shap_translator import shap_translator
from explainability.text.confidence_translator import confidence_translator
from explainability.text.clothes_text import clothes_text
from explainability.text.weather_text import weather_text, _explain_columns
from explainability.text.prediction_explainer import prediction_explainer
from explainability.text.emotion_text import emotion_text
from explainability.confidence_ranker import confidence_ranker


# ── shap_extractor ────────────────────────────────────────────────────────────

def test_shap_extractor_returns_dict_keyed_by_column(high_confidence_today_json):
    result = shap_extractor(high_confidence_today_json, "rain")
    assert "rain" in result
    assert isinstance(result["rain"], list)


def test_shap_extractor_filters_location_code(high_confidence_today_json):
    contributors = shap_extractor(high_confidence_today_json, "rain")["rain"]
    features = [c["feature"] for c in contributors]
    assert "location_code" not in features


def test_shap_extractor_sorted_by_abs_magnitude(high_confidence_today_json):
    contributors = shap_extractor(high_confidence_today_json, "rain")["rain"]
    magnitudes = [abs(c["shap_value"]) for c in contributors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_shap_extractor_zero_shap_value_direction_is_no(zero_shap_today_json):
    # day_of_year is at index 7 and has 0.0 shap value → direction must be "no"
    contributors = shap_extractor(zero_shap_today_json, "rain")["rain"]
    zero_contribs = [c for c in contributors if c["shap_value"] == 0.0]
    assert len(zero_contribs) > 0
    assert all(c["direction"] == "no" for c in zero_contribs)


def test_shap_extractor_positive_shap_direction_is_increases(high_confidence_today_json):
    contributors = shap_extractor(high_confidence_today_json, "rain")["rain"]
    positive = [c for c in contributors if c["shap_value"] > 0]
    assert all(c["direction"] == "increases" for c in positive)


def test_shap_extractor_negative_shap_direction_is_decreases(high_confidence_today_json):
    contributors = shap_extractor(high_confidence_today_json, "rain")["rain"]
    negative = [c for c in contributors if c["shap_value"] < 0]
    assert all(c["direction"] == "decreases" for c in negative)


# ── shap_translator ───────────────────────────────────────────────────────────

def test_shap_translator_returns_snippet_for_each_column(high_confidence_today_json):
    result = shap_translator(high_confidence_today_json, ["rain", "cloud"])
    assert "rain" in result
    assert "cloud" in result


def test_shap_translator_snippet_is_non_empty_string(high_confidence_today_json):
    result = shap_translator(high_confidence_today_json, ["rain"])
    assert isinstance(result["rain"], str)
    assert len(result["rain"]) > 0


def test_shap_translator_snippet_mentions_column_display_name(high_confidence_today_json):
    result = shap_translator(high_confidence_today_json, ["rain"])
    assert "rainfall" in result["rain"].lower()


def test_shap_translator_single_contributor_snippet_format(high_confidence_today_json):
    # Reduce the shap_block to 2 features (1 real + location_code) so only 1
    # contributor survives after filtering → single-contributor format triggered
    import copy
    j = copy.deepcopy(high_confidence_today_json)
    j["variables"]["next_rain mm"]["shap"]["features"] = ["min_temp °c", "location_code"]
    j["variables"]["next_rain mm"]["shap"]["values"] = [0.5, 0.0]
    result = shap_translator(j, ["rain"])
    assert "mainly due to" in result["rain"]
    assert " and " not in result["rain"]   # no "and" in a single-contributor snippet


def test_shap_translator_empty_contributors_returns_empty_string(high_confidence_today_json):
    # shap_block with only location_code → all contributors filtered → empty string
    import copy
    j = copy.deepcopy(high_confidence_today_json)
    j["variables"]["next_rain mm"]["shap"]["features"] = ["location_code"]
    j["variables"]["next_rain mm"]["shap"]["values"] = [0.0]
    result = shap_translator(j, ["rain"])
    assert result["rain"] == ""


def test_shap_translator_day_of_year_resolves_to_season_name(high_confidence_today_json):
    # Make day_of_year the top SHAP contributor so _resolve_feature_name line 31 is hit
    import copy
    j = copy.deepcopy(high_confidence_today_json)
    # day_of_year is at index 7 — give it the largest value
    day_of_year_top = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5, 0.0]
    j["variables"]["next_rain mm"]["shap"]["values"] = day_of_year_top
    result = shap_translator(j, ["rain"])
    # day_of_year=183 → Summer; snippet must mention the season
    assert "Summer" in result["rain"]


def test_shap_translator_unknown_feature_passes_through(high_confidence_today_json):
    # Replace the first feature name with something not in SCENARIO_COLUMN_TO_SHORT
    # so _resolve_feature_name line 34 (unknown fallback) is hit
    import copy
    j = copy.deepcopy(high_confidence_today_json)
    features = list(j["variables"]["next_rain mm"]["shap"]["features"])
    features[0] = "mystery_input"
    j["variables"]["next_rain mm"]["shap"]["features"] = features
    # Give mystery_input the largest shap value so it enters the top-2
    j["variables"]["next_rain mm"]["shap"]["values"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = shap_translator(j, ["rain"])
    assert "mystery_input" in result["rain"]


# ── confidence_translator ─────────────────────────────────────────────────────

def test_confidence_translator_returns_string(high_confidence_today_json):
    result = confidence_translator(["rain"], high_confidence_today_json)
    assert isinstance(result, str) and len(result) > 0


def test_high_confidence_produces_confident_label(high_confidence_today_json):
    result = confidence_translator(["rain"], high_confidence_today_json)
    assert "Confident" in result


def test_uniform_confidence_produces_single_sentence(high_confidence_today_json):
    # All variables at 80% → no per-column deviation → no second sentence
    result = confidence_translator(["rain", "wind"], high_confidence_today_json)
    assert "\n" not in result


def test_diverging_confidence_produces_deviation_clause(diverging_confidence_today_json):
    # rain=90, temp_max=10 → both deviate from baseline → second sentence added
    result = confidence_translator(["rain", "temp_max"], diverging_confidence_today_json)
    assert "higher" in result or "lower" in result
    assert "\n" in result


# ── confidence_ranker ─────────────────────────────────────────────────────────

def test_confidence_ranker_detects_higher_deviation(diverging_confidence_today_json):
    # rain=90 vs baseline ≈ 30 → diff +60 > 15 → "higher"
    result = confidence_ranker(["rain", "temp_max"], diverging_confidence_today_json)
    assert result.get("rain") == "higher"


def test_confidence_ranker_detects_lower_deviation(diverging_confidence_today_json):
    # temp_max=10 vs baseline ≈ 30 → diff -20 < -15 → "lower"
    result = confidence_ranker(["rain", "temp_max"], diverging_confidence_today_json)
    assert result.get("temp_max") == "lower"


def test_confidence_ranker_uniform_confidence_no_deviations(high_confidence_today_json):
    # All 80% → geometric mean 80 → no column deviates by 15+
    result = confidence_ranker(["rain", "wind", "cloud"], high_confidence_today_json)
    assert result == {}


def test_confidence_ranker_caps_at_two_columns(diverging_confidence_today_json):
    result = confidence_ranker(
        ["rain", "cloud", "wind", "humidity", "temp_min", "temp_max", "wind_dir"],
        diverging_confidence_today_json,
    )
    assert len(result) <= 2


# ── clothes_text ──────────────────────────────────────────────────────────────

def test_clothes_text_empty_outfit_returns_fox_base_message(high_confidence_today_json):
    result = clothes_text({}, high_confidence_today_json)
    assert result == {"fox_base": "Nothing special to wear today."}


def test_clothes_text_item_key_in_result(high_confidence_today_json):
    outfit = {"umbrella": {"rain": "light", "wind": "light"}}
    result = clothes_text(outfit, high_confidence_today_json)
    assert "umbrella" in result


def test_clothes_text_fox_base_always_added(high_confidence_today_json):
    outfit = {"jacket": {"temp_min": "cold"}}
    result = clothes_text(outfit, high_confidence_today_json)
    assert "fox_base" in result


def test_clothes_text_no_key_columns_skips_shap_and_confidence(high_confidence_today_json):
    # Item not in CLOTHING_KEY_COLUMNS → filtered_columns=[] → branches 60 & 66 hit
    outfit = {"mystery_item": {"rain": "light"}}
    result = clothes_text(outfit, high_confidence_today_json)
    assert "mystery_item" in result
    assert isinstance(result["mystery_item"], str)


def test_clothes_text_result_values_are_strings(high_confidence_today_json):
    outfit = {"scarf": {"temp_min": "cold"}, "jacket": {"temp_min": "cold"}}
    result = clothes_text(outfit, high_confidence_today_json)
    for key, text in result.items():
        assert isinstance(text, str)


# ── prediction_explainer ──────────────────────────────────────────────────────

def test_prediction_explainer_empty_changes_returns_empty_string(high_confidence_today_json):
    assert prediction_explainer(high_confidence_today_json, {}) == ""


def test_prediction_explainer_with_changes_returns_non_empty(high_confidence_today_json):
    changes = {"rain": {"old": 3.67, "new": 0.60, "direction": "decrease"}}
    result = prediction_explainer(high_confidence_today_json, changes)
    assert isinstance(result, str) and len(result) > 0


def test_prediction_explainer_mentions_changed_column(high_confidence_today_json):
    changes = {"rain": {"old": 3.67, "new": 0.60, "direction": "decrease"}}
    result = prediction_explainer(high_confidence_today_json, changes)
    assert "rainfall" in result.lower()


def test_prediction_explainer_multiple_changes(high_confidence_today_json):
    changes = {
        "rain":     {"old": 3.67, "new": 0.60, "direction": "decrease"},
        "temp_max": {"old": 15.0, "new": 22.0, "direction": "increase"},
    }
    result = prediction_explainer(high_confidence_today_json, changes)
    assert "rainfall" in result.lower()
    assert "daytime temperature" in result.lower()


# ── emotion_text ──────────────────────────────────────────────────────────────

def test_emotion_text_confident_returns_dict(high_confidence_today_json):
    emotion = {"expression": "confident", "change": False, "changed": [], "changes": {}}
    result = emotion_text(high_confidence_today_json, emotion, "Sunny")
    assert isinstance(result, dict)
    assert "confident" in result


def test_emotion_text_apologetic_no_change(high_confidence_today_json):
    # expression="apologetic", change=False → "Fox is not very confident"
    emotion = {"expression": "apologetic", "change": False, "changed": [], "changes": {}}
    result = emotion_text(high_confidence_today_json, emotion, "Cold Day")
    assert "apologetic" in result
    assert "not very confident" in result["apologetic"]


def test_emotion_text_apologetic_with_change_calls_explainer(high_confidence_today_json):
    # expression="apologetic", change=True + changes → prediction_explainer called
    emotion = {
        "expression": "apologetic",
        "change":     True,
        "changed":    ["rain"],
        "changes":    {"rain": {"old": 3.67, "new": 0.60, "direction": "decrease"}},
    }
    result = emotion_text(high_confidence_today_json, emotion, "Light Rain")
    assert "apologetic" in result
    assert len(result["apologetic"]) > 0


def test_emotion_text_neutral_expression(medium_confidence_today_json):
    emotion = {"expression": "neutral", "change": False, "changed": [], "changes": {}}
    result = emotion_text(medium_confidence_today_json, emotion, "Mostly Cloudy")
    assert "neutral" in result


# ── weather_text ──────────────────────────────────────────────────────────────

def test_weather_text_returns_dict_keyed_by_icon_label(high_confidence_today_json, sunny_categories):
    result = weather_text(high_confidence_today_json, sunny_categories, "sunny")
    assert "sunny" in result
    assert isinstance(result["sunny"], str) and len(result["sunny"]) > 0


def test_weather_text_cold_day_with_secondary_overcast(high_confidence_today_json, cold_overcast_categories):
    # Cold Day (NON_ICON_LABELS) → labeller uses secondary "Overcast" for icon
    # weather_text secondary loop exercises lines 94–101
    result = weather_text(high_confidence_today_json, cold_overcast_categories, "overcast")
    assert "overcast" in result
    assert isinstance(result["overcast"], str)
    assert len(result["overcast"]) > 0


def test_explain_columns_empty_returns_empty_string(high_confidence_today_json):
    # Directly tests the guard clause at line 35 of weather_text.py
    result = _explain_columns([], {}, high_confidence_today_json)
    assert result == ""
