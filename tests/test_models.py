import sys
import os
from pydantic import ValidationError

# Add the directory containing models.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import using simple module name
from models import (
    Category,
    CBAState,
    CBAObservation,   # note: not Observation
    CBAAction,
    FixType,
    Shape,
    ColorBlindType
)

import pytest

def test_category_valid():
    cat = Category(
        hex="#FF0000",
        shape=Shape.CIRCLE,
        points=[(1,2), (3,4)]
    )
    assert cat.hex == "#FF0000"
    assert cat.shape == Shape.CIRCLE
    assert len(cat.points) == 2

def test_category_invalid_hex():
    with pytest.raises(ValidationError) as excinfo:
        Category(hex="invalid", shape=Shape.CIRCLE)
    assert "pattern" in str(excinfo.value)

def test_category_invalid_shape():
    with pytest.raises(ValidationError) as excinfo:
        Category(hex="#FF0000", shape="pentagon")  # not in Shape enum
    # The error message from Pydantic for enum mismatch includes "Input should be"
    assert "Input should be" in str(excinfo.value)

def test_category_empty_points():
    cat = Category(hex="#FF0000", shape=Shape.CIRCLE)
    assert cat.points == []

def test_state_valid():
    cat = Category(hex="#FF0000", shape=Shape.CIRCLE)
    state = CBAState(
        scatter_plot="base64-encoded-image",
        scatter_plot_shape=[100, 100, 3],
        categories={"Class A": cat},
        colorblind_types=[ColorBlindType.DEUTERANOPIA],
        step_count=0,
        max_steps=10,
        fixes_applied=[],
        is_solved=False,
    )
    assert state.step_count == 0
    assert len(state.categories) == 1
    assert state.scatter_plot == "base64-encoded-image"

def test_state_optional_fields():
    state = CBAState(
        scatter_plot="base64-encoded-image",
        scatter_plot_shape=[10, 10, 3],
        categories={},
        colorblind_types=[],
        step_count=0,
    )
    assert state.max_steps is None
    assert state.fixes_applied == []
    assert state.is_solved is False

def test_observation_valid():
    obs = CBAObservation(
        scatter_plot="base64-encoded-image",
        scatter_plot_shape=[200, 200, 3],
        hex_code_per_category={"A": "#FF0000"},
        shape_per_category={"A": Shape.CIRCLE},
        colorblind_types=[ColorBlindType.DEUTERANOPIA.value],
        step_count=2,
        max_steps=10,
        is_done=False
    )
    assert isinstance(obs.scatter_plot, str)
    assert obs.hex_code_per_category["A"] == "#FF0000"

def test_observation_arbitrary_type_allowed():
    obs = CBAObservation(
        scatter_plot="base64-encoded-image",
        scatter_plot_shape=[100, 100, 3],
        hex_code_per_category={},
        shape_per_category={},
        colorblind_types=[],
        step_count=0,
        max_steps=10,
        is_done=False
    )
    assert isinstance(obs.scatter_plot, str)

def test_action_recolor_valid():
    action = CBAAction(
        target="Class A",
        fix_type=FixType.RECOLOR,
        change_hex="#0077BB"
    )
    assert action.target == "Class A"
    assert action.fix_type == FixType.RECOLOR
    assert action.change_hex == "#0077BB"
    assert action.change_shape is None

def test_action_recolor_invalid_hex():
    with pytest.raises(ValidationError) as excinfo:
        CBAAction(
            target="Class A",
            fix_type=FixType.RECOLOR,
            change_hex="invalid"
        )
    # The validator raises a ValueError with the message we defined
    assert "new_hex must be a valid hex code" in str(excinfo.value)

def test_action_reshape_valid():
    action = CBAAction(
        target="Class A",
        fix_type=FixType.RESHAPE,
        change_shape=Shape.SQUARE
    )
    assert action.change_shape == Shape.SQUARE

def test_action_reshape_invalid_shape():
    with pytest.raises(ValidationError) as excinfo:
        CBAAction(
            target="Class A",
            fix_type=FixType.RESHAPE,
            change_shape="pentagon"  # Not a Shape enum member
        )
    # Pydantic will raise a ValidationError because the field type is Shape
    assert "Input should be" in str(excinfo.value)

def test_category_points_mutation():
    cat = Category(hex="#FF0000", shape=Shape.CIRCLE, points=[(1,2)])
    cat.points.append((3,4))
    assert len(cat.points) == 2

def test_state_with_large_points():
    cat = Category(
        hex="#FF0000",
        shape=Shape.CIRCLE,
        points=[(i, i*2) for i in range(1000)]
    )
    state = CBAState(
        scatter_plot="base64-encoded-image",
        scatter_plot_shape=[100, 100, 3],
        categories={"A": cat},
        colorblind_types=[ColorBlindType.DEUTERANOPIA],
        step_count=0,
    )
    assert len(state.categories["A"].points) == 1000
