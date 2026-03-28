"""
Color Blind Accessibility Environment - 
A single agent environment that fix scatter plot data for the color blind users."""

# from .client import SnakeEnv
from .models import CBAAction, CBAObservation, CBAState

__all__ = ["CBAAction", "CBAObservation", "CBAState"]
