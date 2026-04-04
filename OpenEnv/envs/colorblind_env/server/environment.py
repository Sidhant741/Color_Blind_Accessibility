"""
Color Blind Accessibility Environment
"""

import random
import numpy as np
from itertools import combinations
from uuid import uuid4
import base64
import io
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openenv.core.env_server import Environment

from models import (
    CBAAction, CBAObservation, CBAState,
    Shape, Category, ColorBlindType, FixType
)
from .config import TASK_CONFIGS
from .utils import compute_delta_e, rgb_to_hex


class CBAEnvironment(Environment):

    def __init__(self, task="easy"):
        super().__init__()

        assert task in TASK_CONFIGS, "Invalid task"

        self.task = task
        self.config = TASK_CONFIGS[task]

        self.categories = None
        self.colorblind_types = None
        self.delta_E_matrix = {}
        self.steps_taken = 0
        self.is_done = False
        self.is_solved = False
        self.fixes_applied = []
        self._state = None

    @property
    def state(self):
        return self._state

    def reset(self, **kwargs):
    
        self.steps_taken = 0
        self.is_done = False
        self.is_solved = False
        self.fixes_applied = []
        self.delta_E_matrix = {}

        import os, json
        preset_data = None
        if os.path.exists("/tmp/preset_layout.json"):
            try:
                with open("/tmp/preset_layout.json", "r") as f:
                    preset_data = json.load(f)
            except Exception:
                pass

        if preset_data is not None:
            # Rehydrate layout
            layout = preset_data.get("categories", preset_data) # handle old/new format
            self.categories = {k: Category(**v) for k, v in layout.items()}
            
            # Rehydrate colorblind types if present
            if "colorblind_types" in preset_data:
                self.colorblind_types = [ColorBlindType(t) for t in preset_data["colorblind_types"]]
            else:
                self.colorblind_types = random.sample(
                    list(ColorBlindType),
                    self.config["no_of_cb_types"]
                )
        else:
            self.colorblind_types = random.sample(
                list(ColorBlindType),
                self.config["no_of_cb_types"]
            )
            self.categories = self._generate_categories()
            self._assign_broken_colors()
            
        self._compute_delta_e_matrix()

        image = self._render()

        self._state = CBAState(
            episode_id=str(uuid4()),
            step_count=0,
            scatter_plot=image,
            scatter_plot_shape=self.image_shape,
            categories=self.categories,
            colorblind_types=self.colorblind_types,
            max_steps=self.config["max_steps"],
            fixes_applied=self.fixes_applied,
            delta_E_matrix=self.delta_E_matrix,
            is_solved=self.is_solved,
        )

        return self._build_observation()

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: CBAAction):

        if self.is_done:
            raise RuntimeError("Episode finished. Call reset().")

        if action.target not in self.categories:
            raise ValueError("Invalid category")

        prev_matrix = self.delta_E_matrix.copy()

        # Apply action
        if action.fix_type == FixType.RECOLOR:
            self.categories[action.target] = self.categories[action.target].model_copy(
                update={"hex": action.change_hex}
            )
        else:
            self.categories[action.target] = self.categories[action.target].model_copy(
                update={"shape": action.change_shape}
            )

        self.steps_taken += 1

        # Recompute only affected pairs
        self._update_delta_e_for_target(action.target)

        reward = self._compute_reward(action, prev_matrix)

        self._check_done()

        if self.config["render"]:
            image = self._render()
            self._state.scatter_plot = image
            self._state.scatter_plot_shape = self.image_shape

        self.fixes_applied.append(str(action))

        # Update state
        self._state.step_count = self.steps_taken
        self._state.categories = self.categories
        self._state.delta_E_matrix = self.delta_E_matrix
        self._state.is_solved = self.is_solved

        obs = self._build_observation()
        obs.reward = reward

        return obs

    # -----------------------------
    # CATEGORY GENERATION
    # -----------------------------
    def _generate_categories(self):

        categories = {}

        for i in range(self.config["n_categories"]):
            name = f"Class {chr(65+i)}"

            points = [
                (random.random(), random.random())
                for _ in range(self.config["n_points"])
            ]

            categories[name] = Category(
                hex="#FFFFFF",
                shape=Shape.CIRCLE,
                points=points
            )

        return categories

    # -----------------------------
    # BROKEN COLORS
    # -----------------------------
    def _assign_broken_colors(self):

        category_names = list(self.categories.keys())
        threshold = self.config["delta_E_threshold"]

        while True:
            base = tuple(random.randint(30, 220) for _ in range(3))
            base_hex = rgb_to_hex(*base)

            colors = [base_hex]

            for _ in range(len(category_names) - 1):
                candidate = rgb_to_hex(
                    *(random.randint(30, 220) for _ in range(3))
                )

                valid = all(
                    compute_delta_e(base_hex, candidate, cb) < threshold
                    for cb in self.colorblind_types
                )

                if not valid:
                    break

                colors.append(candidate)

            if len(colors) == len(category_names):
                for i, name in enumerate(category_names):
                    self.categories[name] = self.categories[name].model_copy(
                        update={"hex": colors[i]}
                    )
                return

    # -----------------------------
    # DELTA E MATRIX
    # -----------------------------
    def _compute_delta_e_matrix(self):

        self.delta_E_matrix = {}

        for i, j in combinations(self.categories.keys(), 2):
            self.delta_E_matrix[(i, j)] = {
                cb: compute_delta_e(
                    self.categories[i].hex,
                    self.categories[j].hex,
                    cb
                )
                for cb in self.colorblind_types
            }

    def _update_delta_e_for_target(self, target):

        for key in list(self.delta_E_matrix.keys()):
            if target in key:
                del self.delta_E_matrix[key]

        for other in self.categories:
            if other == target:
                continue

            pair = tuple(sorted([target, other]))

            self.delta_E_matrix[pair] = {
                cb: compute_delta_e(
                    self.categories[pair[0]].hex,
                    self.categories[pair[1]].hex,
                    cb
                )
                for cb in self.colorblind_types
            }

    # -----------------------------
    # REWARD
    # -----------------------------
    def _compute_reward(self, action=None, prev_matrix=None):

        total_score = 0.0
        pair_count = 0

        categories = list(self.categories.keys())
        n = len(categories)

        if n < 2:
            return 0.0

        for i in range(n):
            for j in range(i + 1, n):

                cat1 = categories[i]
                cat2 = categories[j]

                hex1 = self.categories[cat1].hex
                hex2 = self.categories[cat2].hex

                shape1 = self.categories[cat1].shape
                shape2 = self.categories[cat2].shape

                # COLOR
                color_score = 0.0
                for cb_type in self.colorblind_types:
                    delta = compute_delta_e(hex1, hex2, cb_type)

                    norm_delta = delta / (self.config["delta_E_threshold"] + 1e-8)
                    norm_delta = max(0.0, min(1.0, norm_delta))

                    color_score += norm_delta

                color_score /= len(self.colorblind_types)

                # SHAPE
                shape_score = 1.0 if shape1 != shape2 else 0.0

                # 🔥 NORMALIZED COMBINATION
                total_weight = self.config["color_weight"] + self.config["shape_weight"] + 1e-8

                pair_score = (
                    (self.config["color_weight"] * color_score +
                    self.config["shape_weight"] * shape_score)
                    / total_weight
                )

                total_score += pair_score
                pair_count += 1

        # TRUE AVERAGE
        reward = total_score / pair_count if pair_count > 0 else 0.0

        # FINAL CLAMP
        reward = max(0.0, min(1.0, reward))

        return reward
    # -----------------------------
    # DONE CHECK
    # -----------------------------
    def _check_done(self):

        threshold = self.config["delta_E_threshold"]

        self.is_solved = all(
            v >= threshold
            for pair in self.delta_E_matrix.values()
            for v in pair.values()
        )

        self.is_done = self.is_solved or self.steps_taken >= self.config["max_steps"]

    # -----------------------------
    # RENDER
    # -----------------------------
    def _render(self):

        fig = plt.figure(figsize=(6, 4), dpi=72)

        for name, cat in self.categories.items():
            pts = np.array(cat.points)
            plt.scatter(pts[:, 0], pts[:, 1], color=cat.hex, marker=cat.shape.value)

        plt.title("CBA")
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )[:, :, :3]

        plt.close(fig)

        self.image_shape = list(image.shape)

        return self._encode(image)

    def _encode(self, image):
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    # -----------------------------
    # OBSERVATION
    # -----------------------------
    def _build_observation(self):

        return CBAObservation(
            scatter_plot=self._state.scatter_plot,
            scatter_plot_shape=self._state.scatter_plot_shape,
            hex_code_per_category={k: v.hex for k, v in self.categories.items()},
            shape_per_category={k: v.shape for k, v in self.categories.items()},
            colorblind_types=self.colorblind_types,
            step_count=self.steps_taken,
            max_steps=self.config["max_steps"],
            is_done=self.is_done,
        )