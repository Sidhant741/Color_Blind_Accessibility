"""
Data models for Color-only RL Environment
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from openenv.core.env_server import Action, Observation


# -----------------------------
# Colorblind Types
# -----------------------------
class ColorBlindType(str, Enum):
    DEUTERANOPIA = "deuteranopia"
    PROTANOPIA = "protanopia"
    TRITANOPIA = "tritanopia"


# -----------------------------
# Category (ONLY color + points)
# -----------------------------
class Category(BaseModel):
    hex: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')
    points: List[Tuple[float, float]] = Field(default_factory=list)


# -----------------------------
# State
# -----------------------------
class State(BaseModel):
    categories: Dict[str, Category]
    colorblind_types: List[ColorBlindType]
    step_count: int = 0
    max_steps: int = 10
    reward: float = 0.0
    done: bool = False

# -----------------------------
# Backwards Compatibility / Server Models
# -----------------------------

class FixType(str, Enum):
    RECOLOR = "recolor"
    RESHAPE = "reshape"

class Shape(str, Enum):
    CIRCLE = "o"

class CBAAction(Action):
    # Old discrete fields
    target: Optional[str] = None
    fix_type: Optional[FixType] = None
    change_hex: Optional[str] = None
    change_shape: Optional[Shape] = None
    
    # New continuous field
    continuous_action: Optional[List[float]] = None

class CBAObservation(Observation):
    # Support both formats
    vector: Optional[List[float]] = None
    hex_code_per_category: Optional[Dict[str, str]] = None
    colorblind_types: Optional[List[ColorBlindType]] = None
    is_done: bool = False
    reward: float = 0.0