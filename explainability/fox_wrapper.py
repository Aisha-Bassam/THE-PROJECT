import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules import LAYER_ORDER

# DISSERTATION NOTE: (Ch4/Ch5) fox_wrapper is the final step of the fox pipeline.
# It takes the full outputs of clothes_mapper and emotion_generator and produces
# an ordered list of layer names for the UI to render. The clothing recommendation
# IS the XAI interface — this function is what makes that concrete in the UI.


def fox_wrapper(clothes_mapper_output, emotion_generator_output):
    """
    Assembles the ordered list of fox layer names to render in the UI.

    Parameters
    ----------
    clothes_mapper_output : dict
        Full output of clothes_mapper.
        Keys are clothing item names, values are driving categories (ignored here).
        e.g. {"jacket": {"temp_min": "cold"}, "raincoat": {"rain": "heavy"}}

    emotion_generator_output : dict
        Full output of emotion_generator.
        Only the "expression" key is used here; everything else is ignored.
        e.g. {"expression": "confident", "label": ..., "changed": ..., ...}

    Returns
    -------
    list of str
        Ordered list of layer names, from bottom to top of the visual stack.
        e.g. ["fox_base", "jacket", "raincoat", "confident", "umbrella"]
        The UI maps these to filenames (e.g. "jacket" → "jacket.png").
    """

    # Start with the active layers as a set — order does not matter here,
    # LAYER_ORDER enforces the final stack order below
    active = set()

    # fox_base is always present regardless of weather or confidence
    active.add("fox_base")

    # Extract clothing item names from clothes_mapper output
    # Only keys matter — driving categories are for the text flow, not the fox
    active.update(clothes_mapper_output.keys())

    # Extract the expression string from emotion_generator output
    # Only the expression is needed — everything else goes to the text flow
    expression = emotion_generator_output.get("expression")
    if expression:
        active.add(expression)

    # Iterate LAYER_ORDER and build the final stack, preserving visual order
    # Any active layer not in LAYER_ORDER is silently ignored (safety fallback)
    stack = [layer for layer in LAYER_ORDER if layer in active]

    return stack