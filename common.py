"""
common.py
---------
Shared utilities across all WeatherFox layers.
Imported by: ml/, explainability/, and any component that needs
cross-layer functionality.
"""

import numpy as np

def geometric_mean(values):
    """
    Geometric mean of a list of confidence scores (0-100).
    Clips to 0.01 to avoid log(0) if a score hits exactly 0.

    # DISSERTATION NOTE: (Ch4 - Methodology/Implementation)
    # Used to combine per-variable confidence scores into a single composite.
    # Empirically justified: correlation with actual prediction error was
    # 0.2878 (geometric) vs 0.2712 (minimum) vs 0.2644 (average).
    """
    values = np.clip(values, 0.01, 100)
    return round(float(np.exp(np.mean(np.log(values)))), 2)