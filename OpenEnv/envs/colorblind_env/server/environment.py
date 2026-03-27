"""
Color Blind Accessibility Environment

# Methods to write
1. __init__
2. reset
3. step
4. _generate_categories
5. _assign_broken_colors
6. _render_scatter_plot
7. _build_observation
8. _compute_reward
9. _check_done

"""

from typing import Any

from uuid import uuid4

import gym
import marlenv.envs  # Register marlenv environments with gym
import numpy as np

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server.interfaces import Environment
    from core.env_server.types import State

    from ..models import CBAAction, CBAObservation, CBAState
except ImportError:
    from models import CBAAction, CBAObservation, CBAState

    try:
        # Standalone imports with the current openenv package namespace
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import State
    except ImportError:
        # Backward-compatible standalone imports with the legacy namespace
        from openenv_core.env_server.interfaces import Environment
        from openenv_core.env_server.types import State

class CBAEnvironment(Environment):
    """
    """

    def __init__(
        self,
        task=None,
    )
