"""
Utility functions for Color Blind Accessibility Environment
"""

from functools import lru_cache
from typing import Tuple

from colorspacious import cspace_convert
from skimage.color import deltaE_ciede2000


# ---------------------------------
# BASIC CONVERSIONS
# ---------------------------------

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def rgb_to_lab(r: int, g: int, b: int):
    rgb_norm = [r / 255.0, g / 255.0, b / 255.0]
    return cspace_convert(rgb_norm, "sRGB1", "CIELab")


# ---------------------------------
# COLOR BLIND SIMULATION (FAST)
# ---------------------------------

def simulate_cb_rgb(rgb: Tuple[int, int, int], cb_type, severity: int = 100) -> Tuple[int, int, int]:
    """
    Simulate how a color is perceived by a color-blind user.

    Args:
        rgb: (R, G, B) in [0,255]
        cb_type: Enum or str
        severity: 100 → full color blindness

    Returns:
        Simulated RGB in [0,255]
    """

    cb_type_str = cb_type.value if hasattr(cb_type, "value") else cb_type

    opia_to_anomaly = {
        "protanopia": "protanomaly",
        "deuteranopia": "deuteranomaly",
        "tritanopia": "tritanomaly",
    }

    cvd_space = {
        "name": "sRGB1+CVD",
        "cvd_type": opia_to_anomaly[cb_type_str],
        "severity": severity,
    }

    # Normalize
    r, g, b = rgb
    rgb_norm = [r / 255.0, g / 255.0, b / 255.0]

    # Simulate (From normal to CVD perception)
    simulated = cspace_convert(rgb_norm, cvd_space, "sRGB1")

    # Clip (IMPORTANT)
    simulated = [max(0, min(1, c)) for c in simulated]

    # Convert back to [0,255]
    return tuple(int(c * 255) for c in simulated)


# ---------------------------------
# DELTA E COMPUTATION (OPTIMIZED)
# ---------------------------------

@lru_cache(maxsize=10000)
def compute_delta_e(hex1: str, hex2: str, cb_type) -> float:
    """
    Compute perceptual difference (CIEDE2000) under color blindness.
    Cached for performance.
    """

    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)

    sim1 = simulate_cb_rgb(rgb1, cb_type)
    sim2 = simulate_cb_rgb(rgb2, cb_type)

    lab1 = rgb_to_lab(*sim1)
    lab2 = rgb_to_lab(*sim2)

    return float(deltaE_ciede2000(lab1, lab2))


# ---------------------------------
# OPTIONAL: HEX WRAPPER (DEBUG)
# ---------------------------------

def simulate_cb_hex(hex_code: str, cb_type) -> str:
    """
    Convenience wrapper (not used in training).
    """
    rgb = hex_to_rgb(hex_code)
    sim_rgb = simulate_cb_rgb(rgb, cb_type)
    return rgb_to_hex(*sim_rgb)