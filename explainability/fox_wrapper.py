import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules import LAYER_ORDER

# DISSERTATION NOTE: (Ch4/Ch5) fox_wrapper is the final step of the fox pipeline.
# It takes clothes_mapper and emotion_generator outputs and produces an ordered
# list of layer names for the UI to render. The clothing recommendation IS the
# XAI interface — this function is what makes that concrete in the UI.


def fox_wrapper(clothes_mapper_output, emotion_generator_output):
    """
    Assembles the ordered list of fox layer names to render in the UI.

    Parameters
    ----------
    clothes_mapper_output : dict
        Output of clothes_mapper. Keys are clothing item names.
        e.g. {"jacket": {...}, "raincoat": {...}, "umbrella": {...}}

    emotion_generator_output : dict
        Output of emotion_generator. Must contain key "expression".
        e.g. {"expression": "confident", "label": ..., "changed": ...}

    Returns
    -------
    list of str
        Ordered list of layer names, from bottom to top of the visual stack.
        e.g. ["fox_base", "jacket", "raincoat", "confident", "umbrella"]
        The UI is responsible for mapping these to filenames (e.g. "jacket.png").
    """

    # Build the set of active layers from both inputs
    active = set()

    # fox_base is always present
    active.add("fox_base")

    # Add clothing items — only the keys matter, not the driving categories
    for item in clothes_mapper_output:
        active.add(item)

    # Add the emotion expression — only the expression value matters
    expression = emotion_generator_output.get("expression")
    if expression:
        active.add(expression)

    # Iterate LAYER_ORDER and include only active layers, preserving stack order
    stack = [layer for layer in LAYER_ORDER if layer in active]

    return stack