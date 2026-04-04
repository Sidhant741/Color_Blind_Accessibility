"""
Data models for the Color Blind Accessibility Environment
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

from openenv.core.env_server import Action, Observation, State


# -------------------------------
# ENUMS
# -------------------------------

class ColorBlindType(str, Enum):
    DEUTERANOPIA = "deuteranopia"
    PROTANOPIA = "protanopia"
    TRITANOPIA = "tritanopia"


class Shape(str, Enum):
    CIRCLE = "o"
    TRIANGLE_UP = "^"
    STAR = "*"
    CROSS = "x"
    PLUS = "+"
    PENTAGON = "p"
    SQUARE = "s"


class FixType(str, Enum):
    RECOLOR = "recolor"
    RESHAPE = "reshape"


# -------------------------------
# CATEGORY MODEL
# -------------------------------

class Category(BaseModel):
    hex: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')
    shape: Shape
    points: List[Tuple[float, float]] = Field(default_factory=list)

    @field_validator("points")
    @classmethod
    def validate_points(cls, v):
        if len(v) == 0:
            raise ValueError("Category must contain at least one point")
        return v


# -------------------------------
# STATE (FULL INTERNAL TRUTH)
# -------------------------------

class CBAState(State):
    episode_id: str

    scatter_plot: str  # base64 PNG
    scatter_plot_shape: List[int]

    categories: Dict[str, Category]
    colorblind_types: List[ColorBlindType]

    step_count: int
    max_steps: Optional[int] = None

    fixes_applied: List[str] = Field(default_factory=list)

    # (cat_i, cat_j) → {cb_type: delta_E}
    delta_E_matrix: Dict[Tuple[str, str], Dict[ColorBlindType, float]] = Field(default_factory=dict)

    is_solved: bool = False


# -------------------------------
# OBSERVATION (AGENT VIEW)
# -------------------------------

class CBAObservation(Observation):
    scatter_plot: str
    scatter_plot_shape: List[int]

    hex_code_per_category: Dict[str, str]
    shape_per_category: Dict[str, Shape]

    colorblind_types: List[ColorBlindType]

    step_count: int
    max_steps: Optional[int]

    is_done: bool

    # Optional (for training/debugging)
    reward: Optional[float] = None


# -------------------------------
# ACTION
# -------------------------------

class CBAAction(Action):
    target: str
    fix_type: FixType

    change_hex: Optional[str] = None
    change_shape: Optional[Shape] = None

    @field_validator("change_hex")
    @classmethod
    def validate_hex(cls, v):
        if v is not None and not re.match(r'^#[0-9A-Fa-f]{6}$', v):
            raise ValueError("Invalid hex format (e.g. #FF0000)")
        return v

    @model_validator(mode="after")
    def validate_action(self):
        if self.fix_type == FixType.RECOLOR:
            if self.change_hex is None:
                raise ValueError("RECOLOR requires change_hex")
            if self.change_shape is not None:
                raise ValueError("RECOLOR cannot include shape")

        elif self.fix_type == FixType.RESHAPE:
            if self.change_shape is None:
                raise ValueError("RESHAPE requires change_shape")
            if self.change_hex is not None:
                raise ValueError("RESHAPE cannot include hex")

        return self