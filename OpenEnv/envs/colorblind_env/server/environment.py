"""
Color Blind Accessibility Environment

# Methods to write
DONE 1. __init__
DONE 2. reset
DONE 3. step
DONE 4. _generate_categories
DONE 5. _assign_broken_colors
DONE 6. _render_scatter_plot
DONE 7. _build_observation
DONE 8. _compute_reward
DONE 9. _check_done
"""

from typing import Any

from uuid import uuid4

import numpy as np
import random
from .config import *
from .utils import compute_delta_e, rgb_to_hex
import matplotlib
from itertools import combinations
import copy

matplotlib.use('Agg')  # Use non-interactive backend — required for tostring_rgb / buffer_rgba
import matplotlib.pyplot as plt

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server.interfaces import Environment
    from core.env_server.types import State

    from ..models import CBAAction, CBAObservation, CBAState, Shape, Category, ColorBlindType, FixType
except ImportError:
    from models import CBAAction, CBAObservation, CBAState, Shape, Category, ColorBlindType, FixType

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

    def __init__(self, task="easy"):
        super().__init__()

        """
        Initialize the CBA Environment
        """

        assert task in ['easy', 'medium', 'hard'], "task value must be from ['easy', 'medium', 'hard']"

        self.task = task
        
        self.task_config = TASK_CONFIGS[self.task]

        self._state = None

        self.categories = None
        self.colorblind_types = None
        self.current_image = None
        self.delta_E_matrix = {}
        self.steps_taken = 0
        self.is_done = False
        self.is_solved = False
        self.fixes_applied = []
    
    @property
    def state(self):
        return self._state
    
    def _generate_categories(self):
        """
        It creates the initial category structure for the scatter plot.
        Step 1 : Decide category names , simple names like "Class A", "Class B", etc
        Step 2 : For each category, generate random points
        Step 3 : Assign any color for now
        Step 4 : Assign circle shape for now
        Step 5 : Return a dict of category objects 

        {
            "Class A": Category(hex="#FF0000", shape=Shape.CIRCLE, points=[...]),
            "Class B": Category(hex="#00FF00", shape=Shape.CIRCLE, points=[...]),
        }
        """
        n_points = self.task_config['n_points']

        categories = {}
        for i in range(self.task_config['n_categories']):
            label_name = "Class " + chr(65+i)

            points_defined = [(random.random(), random.random()) for _ in range(n_points)]
            color_defined = f"#FFFFFF"
            shape_defined = Shape.CIRCLE

            categories[label_name] = Category(hex=color_defined, shape=shape_defined, points=points_defined)
        
        return categories
    
    def _assign_broken_colors(self):
        """
        Deliberately assigns colors that are indistinguishable for the active CB types.
        Uses LMS color space to generate confusion pairs.
        """
        l_shift = self.task_config['l_shift']
        cb_types = self.colorblind_types
        threshold = self.task_config['delta_E_threshold']
        max_retries = 5000
 
        # LMS matrices
        RGB_TO_LMS = np.array([
            [0.3139902, 0.6395129, 0.0464975],
            [0.1553728, 0.7578945, 0.0867014],
            [0.0177523, 0.1094431, 0.8725692]
        ])
        LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)
 
        # CB type to cone index mapping
        cb_cone_index = {
            ColorBlindType.PROTANOPIA: 0,    # L cone missing
            ColorBlindType.DEUTERANOPIA: 1,  # M cone missing
            ColorBlindType.TRITANOPIA: 2,    # S cone missing
        }
 
        category_names = list(self.categories.keys())
 
        # We need to assign a color to each category such that
        # every pair is confusing for all active CB types
        # Strategy: pick a base color, generate confusion partner,
        # verify delta_E < threshold for all CB types
 
        # Assign first category a random base color
        for attempt in range(max_retries):
            # Generate random base color
            r = random.randint(30, 225)
            g = random.randint(30, 225)
            b = random.randint(30, 225)
            base_hex = rgb_to_hex(r, g, b)
 
            # Generate confusion partner for each subsequent category
            # using the first active CB type's cone
            rgb_norm = np.array([r, g, b]) / 255.0
            lms = RGB_TO_LMS @ rgb_norm
 
            valid_partners = [base_hex]
            all_valid = True
 
            cone_idx = cb_cone_index[cb_types[0]]
 
            for i in range(1, len(category_names)):
                # Shift by i * l_shift from the base so each partner is a
                # distinct point on the same confusion line. Try both directions.
                partner_hex = None
                for sign in (1, -1):
                    lms_partner = lms.copy()
                    lms_partner[cone_idx] += sign * i * l_shift
                    rgb_partner = LMS_TO_RGB @ lms_partner
 
                    if np.any((rgb_partner < 0) | (rgb_partner > 1)):
                        continue  # out of gamut, try other sign
 
                    candidate = rgb_to_hex(*(rgb_partner * 255).astype(int))
 
                    # Verify delta_E < threshold for ALL active CB types vs base
                    if all(compute_delta_e(base_hex, candidate, cb_type) < threshold
                           for cb_type in cb_types):
                        partner_hex = candidate
                        break
 
                if partner_hex is None:
                    all_valid = False
                    break
 
                valid_partners.append(partner_hex)
 
            if all_valid and len(valid_partners) == len(category_names):
                # Assign colors to categories
                for i, name in enumerate(category_names):
                    self.categories[name] = self.categories[name].model_copy(
                        update={"hex": valid_partners[i]}
                    )
                return
 
        raise ValueError(
            f"Could not find valid broken color pairs after {max_retries} attempts. "
            f"Try increasing max_retries or adjusting l_shift."
        )

    def _render_scatter_plot(self, ):   # return an image
        fig = plt.figure(figsize=(16, 10))

        for label_name, value_ in self.categories.items():
            hex_code = value_.hex
            shape_using = value_.shape.value
            points = value_.points

            points_np = np.array(points)

            plt.scatter(points_np[:, 0], points_np[:, 1], marker=shape_using, color=hex_code, label=label_name)

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Color Blind Accessibility")
        fig.canvas.draw()
        # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # plt.close(fig)

        # tostring_rgb was removed in newer matplotlib; buffer_rgba works on the Agg backend
        buf = fig.canvas.buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )
        image = image[:, :, :3]  # Drop alpha channel → keep RGB
 
        plt.close(fig)

        return image
    
    def _build_observation(self) -> CBAObservation:
        hex_code_per_category_observed = {}
        shape_per_category_observed = {}

        for label_name, value_ in self.categories.items():
            hex_code_per_category_observed[label_name] = value_.hex
            shape_per_category_observed[label_name] = value_.shape

        return CBAObservation(
            scatter_plot=self._state.scatter_plot,
            hex_code_per_category=hex_code_per_category_observed,
            shape_per_category=shape_per_category_observed,
            colorblind_types=self._state.colorblind_types,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            is_done=self.is_done
        )

    def reset(self) -> CBAObservation:
        """
        Reset the environment.

        reset()
        → generate categories        → stored in self.categories
        → assign broken colors       → stored in self.categories
        → render scatter plot        → stored in self.current_image
        → build state                → stored in self.state (full truth)
        → build observation          → derived from self.state
        → return observation         → this is what agent receives

        Returns:
            CBAObservation with initial game state
        """
        self.steps_taken = 0
        self.is_done = False
        self.is_solved = False
        self.fixes_applied = []
        self.delta_E_matrix = {}

        all_cb_types = list(ColorBlindType)
        self.colorblind_types = random.sample(all_cb_types, self.task_config["no_of_cb_types"])

        self.categories = self._generate_categories()
        self._assign_broken_colors()
        self._compute_delta_e_matrix()

        self.current_image = self._render_scatter_plot()

        self._state = CBAState(episode_id = str(uuid4()), step_count=0, 
                        scatter_plot = self.current_image, categories=self.categories,
                        colorblind_types = self.colorblind_types, max_steps=self.task_config['max_steps'],
                        fixes_applied = self.fixes_applied, delta_E_matrix = self.delta_E_matrix,
                        is_solved = self.is_solved)
        
        return self._build_observation()

    def _compute_delta_e_matrix(self):  # returns nothing
        pairs = list(combinations(self.categories.keys(), 2))   # pairs = [("Class A", "Class B"), ("Class A", "Class C"), ...]

        for cat_i, cat_j in pairs:
            for_each_cb_delta_value = {}
            hex_i, hex_j = self.categories[cat_i].hex, self.categories[cat_j].hex

            for cb_type in self.colorblind_types:
                delta = compute_delta_e(hex_i, hex_j, cb_type)
                for_each_cb_delta_value[cb_type] = delta

            self.delta_E_matrix[(cat_i, cat_j)] = for_each_cb_delta_value
    
    def _check_done(self):
        is_solved_lst = []

        if not self.delta_E_matrix:
            self.is_solved = False
            self.is_done = False
            return
        
        for pair in self.delta_E_matrix:
            for value in self.delta_E_matrix[pair].values():
                if value < self.task_config['delta_E_threshold']:
                    is_solved_lst.append(False)
                else:
                    is_solved_lst.append(True)
        
        self.is_solved = all(is_solved_lst)
        
        if self.task == "easy":
            self.is_done = self.is_solved
        else:
            self.is_done = self.is_solved or self.steps_taken >= self.task_config['max_steps']
            
    def _compute_reward(self, action, previous_delta_E_matrix):

        total_score = 0
        color_weight = self.task_config['color_weight']
        shape_weight = self.task_config['shape_weight']
        core_reward = 0
        penalty = 0
        bonus = 0

        for pair in self.delta_E_matrix:
            shape_score, color_score = 0, 0
            
            shape_i, shape_j = self.categories[pair[0]].shape, self.categories[pair[1]].shape
            if shape_i != shape_j:
                shape_score = 1
            
            for value in self.delta_E_matrix[pair].values():
                color_score += value
            
            avg_color_score = color_score / len(self.colorblind_types)
            norm_color_score = min(avg_color_score / 40.0, 1.0)

            pair_reward = color_weight * norm_color_score + shape_weight * shape_score 

            total_score += pair_reward

        core_reward = total_score / len(self.delta_E_matrix)    

        if self.task == "hard" and core_reward >= 0.8:
            bonus = (1 - self.steps_taken / self.task_config['max_steps']) * self.task_config['efficiency_weight']
        if self.task == "medium" and self.steps_taken > self.task_config['max_steps']:
            steps_over = self.steps_taken - self.task_config['max_steps']
            penalty -= 0.02 * steps_over

        ## Redundant Action Penalty ##
        previous_core = 0
        already_solved = False
        if previous_delta_E_matrix is not None:
            target = action.target
            threshold = self.task_config['delta_E_threshold']

            # find all pairs involving the target category
            target_pairs = [pair for pair in previous_delta_E_matrix if target in pair]

            # check if ALL those pairs were already well distinguished
            already_solved = all(
                delta >= threshold
                for pair in target_pairs
                for delta in previous_delta_E_matrix[pair].values()
            )

            previous_total = 0
            for pair in previous_delta_E_matrix:
                prev_color_score = sum(previous_delta_E_matrix[pair].values()) / len(self.colorblind_types)
                prev_norm = min(prev_color_score / 40.0, 1.0)
                previous_total += prev_norm
            previous_core = previous_total / len(previous_delta_E_matrix)

        if already_solved:
            penalty -= 0.05

        if core_reward < previous_core:
            penalty -= 0.1 * (previous_core - core_reward)

        ### Final Score ###
        total_reward = core_reward + bonus + penalty
        return max(0.0, min(1.0, total_reward))

    def step(self, action:CBAAction):

        if self.is_done :
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

        if action.target not in self.categories:
            raise ValueError(f"Target category '{action.target}' does not exist")

        previous_delta_E_matrix = copy.deepcopy(self.delta_E_matrix) if self.delta_E_matrix else None
        fix_str = ""

        if action.fix_type == FixType.RECOLOR:
            self.categories[action.target] = self.categories[action.target].model_copy(
                                                update={"hex": action.change_hex}
                                            )

            fix_str = f"{action.fix_type} {action.target} → {action.change_hex}"
        else:
            self.categories[action.target] = self.categories[action.target].model_copy(
                                                update={"shape": action.change_shape}
                                            )
            fix_str = f"{action.fix_type} {action.target} → {action.change_shape}"

        self.steps_taken += 1

        self.current_image = self._render_scatter_plot()

        self._compute_delta_e_matrix()

        reward = self._compute_reward(action, previous_delta_E_matrix)

        self._check_done()

        self.fixes_applied.append(fix_str)

        self._state.step_count = self.steps_taken
        self._state.scatter_plot = self.current_image
        self._state.categories = self.categories
        self._state.fixes_applied = self.fixes_applied
        self._state.delta_E_matrix = self.delta_E_matrix
        self._state.is_solved = self.is_solved
        
        return self._build_observation(), reward, self.is_done, {}