"""
Data models for the Color Blind Accessibility Environment.

A reinforcement learning environment that trains an AI agent to fix 
scatter plot data for the color blind users
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field, BaseModel, ConfigDict, field_validator, model_validator
from enum import Enum

from openenv.core.env_server import Action, Observation, State

# # Support both in-repo and standalone imports
# try:
#     # In-repo imports (when running from OpenEnv repository)
#     from core.env_server.types import Action, Observation
# except ImportError:
#     try:
#         # Standalone imports with the current openenv package namespace
#         from openenv.core.env_server.types import Action, Observation
#     except ImportError:
#         # Backward-compatible standalone imports with the legacy namespace
#         from openenv_core.env_server.types import Action, Observation

# ColorBlind Types
class ColorBlindType(str, Enum):
    DEUTERANOPIA = "deuteranopia"
    PROTANOPIA = "protanopia"
    TRITANOPIA = "tritanopia"

# These are the Shape Constants
class Shape(str, Enum):
    CIRCLE = "o"
    TRIANGLE_UP = "^"
    STAR = "*"
    CROSS = "x"
    PLUS = "+"
    PENTAGON = "p"
    SQUARE = "s"

# Category 
class Category(BaseModel):
    hex : str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$') # validates hex format
    shape: Shape
    points: List[Tuple[float, float]] = Field(default_factory=list)

# State (full internal truth)
class CBAState(State):
    # model_config = ConfigDict(arbitrary_types_allowed = True, frozen=False)
    # episode_id: str
    # scatter_plot : np.ndarray
    scatter_plot: str  # base64 encoded
    scatter_plot_shape: List[int]
    categories : Dict[str, Category]
    colorblind_types : List[ColorBlindType]
    # step_count : int
    max_steps : Optional[int] = None 
    fixes_applied : List[str] = Field(default_factory=list)
    # delta_E_matrix: Dict[Tuple[str, str], Dict[ColorBlindType, float]] = Field(default_factory=dict)
    delta_E_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    is_solved : bool = False    # is solved means colors are fixed

# Observation (what agent sees)
class CBAObservation(Observation):
    # model_config = ConfigDict(arbitrary_types_allowed = True, frozen=False)
    # scatter_plot : np.ndarray
    # scatter_plot: List  # 3D array: height x width x 3 (RGB)
    # scatter_plot_shape: List[int]        # original shape [height, width, 3]
    scatter_plot: str  # base64 encoded image string
    scatter_plot_shape: List[int]  # [height, width, 3] so agent can reconstruct
    hex_code_per_category: Dict[str, str]
    shape_per_category: Dict[str, Shape]
    colorblind_types : List[str]
    step_count : int
    max_steps : Optional[int] = None
    is_done : bool  # is_done means episode ended, it doesnt means we have fixed the colors

class FixType(str, Enum):
    RECOLOR = "recolor"
    RESHAPE = "reshape"

# Action (agent's decision)    
class CBAAction(Action):
    #target: str
    #fix_type: FixType
    #change_hex: Optional[str] = None
    #change_shape: Optional[Shape] = None
    target: str = Field(
        json_schema_extra={"placeholder": "Class A, Class B"}
    )
    fix_type: FixType = Field(
        json_schema_extra={"placeholder": "Recolor or Reshape"}
    )
    change_hex: Optional[str] = Field(
        default=None, 
        json_schema_extra={"placeholder": "e.g. #FF0000"}
    )
    change_shape: Optional[Shape] = Field(
        default=None, 
        json_schema_extra={"placeholder": "O, X, ^, +, s, p, *"}
    )

    @field_validator('change_hex')
    @classmethod
    def validate_hex(cls, v):
        if v is not None:
            if not re.match(r'^#[0-9A-Fa-f]{6}$', v):
                raise ValueError('new_hex must be a valid hex code like #FF0000')
        return v

    @model_validator(mode='after')
    def validate_action_consistency(self):
        if self.fix_type == FixType.RECOLOR:
            if self.change_hex is None:
                raise ValueError('new_hex must be provided when fix_type is RECOLOR')
            if self.change_shape is not None:
                raise ValueError('new_shape must be None when fix_type is RECOLOR')
        
        elif self.fix_type == FixType.RESHAPE:
            if self.change_shape is None:
                raise ValueError('new_shape must be provided when fix_type is RESHAPE')
            if self.change_hex is not None:
                raise ValueError('new_hex must be None when fix_type is RESHAPE')
        
        return self