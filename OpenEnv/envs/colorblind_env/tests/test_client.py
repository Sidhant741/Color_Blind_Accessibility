import sys
import os

import pytest

# Add the environment directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client import CBAEnv
from models import CBAAction, FixType, Shape


def test_step_payload_recolor():
    env = CBAEnv(base_url="http://test")
    action = CBAAction(
        target="Class A",
        fix_type=FixType.RECOLOR,
        change_hex="#0077BB",
        change_shape=None,
    )
    payload = env._step_payload(action)
    assert payload == {
        "target": "Class A",
        "fix_type": "recolor",
        "change_hex": "#0077BB",
        "change_shape": None,
    }


def test_step_payload_reshape():
    env = CBAEnv(base_url="http://test")
    action = CBAAction(
        target="Class A",
        fix_type=FixType.RESHAPE,
        change_shape=Shape.SQUARE,
    )
    payload = env._step_payload(action)
    assert payload == {
        "target": "Class A",
        "fix_type": "reshape",
        "change_hex": None,
        "change_shape": "s",
    }


def test_parse_result():
    env = CBAEnv(base_url="http://test")
    payload = {
        "observation": {
            "scatter_plot": "base64-encoded-image",
            "scatter_plot_shape": [100, 100, 3],
            "hex_code_per_category": {"Class A": "#FF0000"},
            "shape_per_category": {"Class A": "o"},
            "colorblind_types": ["deuteranopia"],
            "step_count": 1,
            "max_steps": 10,
            "is_done": False,
        },
        "reward": 0.5,
        "done": False,
    }

    result = env._parse_result(payload)
    assert result.reward == 0.5
    assert result.done is False
    assert result.observation.hex_code_per_category["Class A"] == "#FF0000"
    assert result.observation.shape_per_category["Class A"] == Shape.CIRCLE


def test_parse_state():
    env = CBAEnv(base_url="http://test")
    payload = {
        "episode_id": "ep-1",
        "step_count": 2,
        "scatter_plot": "base64-encoded-image",
        "scatter_plot_shape": [100, 100, 3],
        "categories": {
            "Class A": {"hex": "#FF0000", "shape": "o", "points": [(0.1, 0.2)]}
        },
        "colorblind_types": ["deuteranopia"],
        "max_steps": 10,
        "fixes_applied": ["FixType.RECOLOR Class A → #0077BB"],
        "delta_E_matrix": {"Class A|Class B": {"deuteranopia": 12.3}},
        "is_solved": False,
    }

    state = env._parse_state(payload)
    assert state.episode_id == "ep-1"
    assert state.step_count == 2
    assert state.categories["Class A"].hex == "#FF0000"
    assert state.categories["Class A"].shape == Shape.CIRCLE
