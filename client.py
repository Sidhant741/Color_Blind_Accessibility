

"""
CBA Environment Client

This module provides the client for connecting to an CBA Environment server
via WebSocket for persistent sessions.
"""

from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from models import CBAAction, CBAObservation, CBAState

class CBAEnv(EnvClient[CBAAction, CBAObservation, CBAState]):

    def _step_payload(self, action: CBAAction) -> Dict[str, Any]:
        return {
            "target": action.target,
            "fix_type": action.fix_type.value,
            "change_hex": action.change_hex,
            "change_shape": action.change_shape.value if action.change_shape else None,        
        }
    
    def _parse_result(self, payload:Dict[str, Any]) -> StepResult[CBAObservation]:

        obs_data = payload.get("observation", {})

        observation = CBAObservation(
            scatter_plot = obs_data.get("scatter_plot", ""),
            scatter_plot_shape = obs_data.get("scatter_plot_shape", []),
            hex_code_per_category = obs_data.get("hex_code_per_category", {}),
            shape_per_category = obs_data.get("shape_per_category", {}),
            colorblind_types = obs_data.get("colorblind_types", []),
            step_count = obs_data.get("step_count", 0),
            max_steps = obs_data.get("max_steps", None),
            is_done = obs_data.get("is_done", False),
            done = payload.get("done", False),
            reward = payload.get("reward", None),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: Dict[str, Any]) -> CBAState:
        return CBAState(
            episode_id = payload.get("episode_id", ""),
            step_count = payload.get("step_count", 0),
            scatter_plot = payload.get("scatter_plot", ""),
            scatter_plot_shape = payload.get("scatter_plot_shape", []),
            categories = payload.get("categories", {}),
            colorblind_types = payload.get("colorblind_types", []),
            max_steps = payload.get("max_steps", None),
            fixes_applied = payload.get("fixes_applied", []),
            delta_E_matrix = payload.get("delta_E_matrix", {}),
            is_solved = payload.get("is_solved", False),
        )