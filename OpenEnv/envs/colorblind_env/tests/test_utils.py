import sys
import os
import pytest
import numpy as np

# Add the directory containing color_simulation.py to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the functions we want to test
from server.utils import simulate_cb, compute_delta_e, hex_to_rgb, rgb_to_hex, rgb_to_lab

def test_hex_to_rgb():
    assert hex_to_rgb("#FF0000") == (255, 0, 0)
    assert hex_to_rgb("#00FF00") == (0, 255, 0)
    assert hex_to_rgb("#0000FF") == (0, 0, 255)
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#000000") == (0, 0, 0)

def test_rgb_to_hex():
    assert rgb_to_hex(255, 0, 0) == "#FF0000"
    assert rgb_to_hex(0, 255, 0) == "#00FF00"
    assert rgb_to_hex(0, 0, 255) == "#0000FF"
    assert rgb_to_hex(255, 255, 255) == "#FFFFFF"
    assert rgb_to_hex(0, 0, 0) == "#000000"

def test_rgb_to_lab():
    lab = rgb_to_lab(255, 0, 0)
    # approximate values for red in Lab
    assert abs(lab[0] - 53.24) < 0.1   # L*
    assert abs(lab[1] - 80.09) < 0.1   # a*
    assert abs(lab[2] - 67.20) < 0.1   # b*

def test_simulate_cb_known_case():
    """Test that simulate_cb returns a known simulated color for a specific input."""
    # Known: protanopia simulation of red (#FF0000) should be a dark reddish/brownish color.
    simulated = simulate_cb("#FF0000", "protanopia", severity=100)
    # We can't hardcode an exact hex because it's complex, but we can check it's not the same as original.
    assert simulated != "#FF0000"
    # Also check that the output is a valid hex code (6 hex digits).
    assert len(simulated) == 7
    assert simulated[0] == '#'
    assert all(c in '0123456789ABCDEFabcdef' for c in simulated[1:])

def test_simulate_cb_invalid_type():
    """Invalid blindness type should raise KeyError."""
    with pytest.raises(KeyError):
        simulate_cb("#FF0000", "unknown_type")

def test_compute_delta_e_same_colors():
    """Delta E between identical colors should be 0."""
    delta = compute_delta_e("#FF0000", "#FF0000", "deuteranopia")
    assert delta == 0.0

def test_compute_delta_e_different_colors():
    """Delta E between different colors should be positive."""
    delta = compute_delta_e("#FF0000", "#00FF00", "deuteranopia")
    assert delta > 0
