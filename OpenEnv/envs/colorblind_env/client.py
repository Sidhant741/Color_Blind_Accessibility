"""
CBA Environment Client

Connects to CBA Environment server via WebSocket.
"""

from __future__ import annotations
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import (
    CBAAction,
    CBAObservation,
    CBAState,
    ColorBlindType,
    Shape,
    Category,
)


class CBAEnv(EnvClient[CBAAction, CBAObservation, CBAState]):

    # -----------------------------
    # STEP PAYLOAD
    # -----------------------------
    def _step_payload(self, action: CBAAction) -> Dict[str, Any]:
        return {
            "target": action.target,
            "fix_type": action.fix_type.value,
            "change_hex": action.change_hex,
            "change_shape": action.change_shape.value if action.change_shape else None,
        }

    # -----------------------------
    # PARSE STEP RESULT
    # -----------------------------
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CBAObservation]:

        try:
            obs_data = payload.get("observation", {})

            # Convert enums properly
            colorblind_types = [
                ColorBlindType(cb) for cb in obs_data.get("colorblind_types", [])
            ]

            shape_per_category = {
                k: Shape(v)
                for k, v in obs_data.get("shape_per_category", {}).items()
            }

            observation = CBAObservation(
                scatter_plot=obs_data.get("scatter_plot", ""),
                scatter_plot_shape=obs_data.get("scatter_plot_shape", []),
                hex_code_per_category=obs_data.get("hex_code_per_category", {}),
                shape_per_category=shape_per_category,
                colorblind_types=colorblind_types,
                step_count=obs_data.get("step_count", 0),
                max_steps=obs_data.get("max_steps"),
                is_done=obs_data.get("is_done", False),
                reward=payload.get("reward", 0.0),
            )

            return StepResult(
                observation=observation,
                reward=payload.get("reward", 0.0),
                done=observation.is_done,
            )

        except Exception as e:
            raise ValueError(f"Failed to parse step result: {e}")

    # -----------------------------
    # PARSE STATE
    # -----------------------------
    def _parse_state(self, payload: Dict[str, Any]) -> CBAState:

        try:
            # Convert enums
            colorblind_types = [
                ColorBlindType(cb) for cb in payload.get("colorblind_types", [])
            ]

            # Convert categories
            categories = {
                k: Category(**v)
                for k, v in payload.get("categories", {}).items()
            }

            # Convert delta matrix keys back to tuple
            raw_matrix = payload.get("delta_E_matrix", {})
            delta_E_matrix = {}

            for key, value in raw_matrix.items():
                if isinstance(key, str):
                    if "|" in key:
                        k1, k2 = key.split("|", 1)
                    else:
                        k1, k2 = key.split(",", 1)
                    pair = (k1.strip(), k2.strip())
                else:
                    pair = tuple(key)

                delta_E_matrix[pair] = {
                    ColorBlindType(cb): val
                    for cb, val in value.items()
                }

            return CBAState(
                episode_id=payload.get("episode_id", ""),
                step_count=payload.get("step_count", 0),
                scatter_plot=payload.get("scatter_plot", ""),
                scatter_plot_shape=payload.get("scatter_plot_shape", []),
                categories=categories,
                colorblind_types=colorblind_types,
                max_steps=payload.get("max_steps"),
                fixes_applied=payload.get("fixes_applied", []),
                delta_E_matrix=delta_E_matrix,
                is_solved=payload.get("is_solved", False),
            )

        except Exception as e:
            raise ValueError(f"Failed to parse state: {e}")

    # -----------------------------
    # OPTIONAL: RENDER HELPER
    # -----------------------------
    def render(self, observation: CBAObservation):
        """
        Decode and display image (for debugging/demo)
        """
        import base64
        import io
        from PIL import Image

        img_bytes = base64.b64decode(observation.scatter_plot)
        img = Image.open(io.BytesIO(img_bytes))
        img.show()