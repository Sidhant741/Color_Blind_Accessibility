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

from typing import Any, Optional, List

from uuid import uuid4

import numpy as np
import random
import matplotlib
from itertools import combinations
import copy
import base64
from PIL import Image
import io

matplotlib.use('Agg')  # Use non-interactive backend — required for tostring_rgb / buffer_rgba
import matplotlib.pyplot as plt

from openenv.core.env_server import Environment

try:
    from ..models import CBAAction, CBAObservation, CBAState, Shape, Category, ColorBlindType, FixType
    from .config import *
    from .utils import compute_delta_e, rgb_to_hex
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import CBAAction, CBAObservation, CBAState, Shape, Category, ColorBlindType, FixType
    from server.config import *
    from server.utils import compute_delta_e, rgb_to_hex

class CBAEnvironment(Environment):
    """
    """
    SUPPORTS_CONCURRENT_SESSIONS = True
    def __init__(self,):
        super().__init__()

        """
        Initialize the CBA Environment
        """
        self.task = "easy"  # default task is easy, can be overridden by reset() argument or env variable
        
        # self.task_config = TASK_CONFIGS[self.task]  # this will be set in reset() when task is finalized

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
    
    def _simulate_cb(self, rgb, cb_type):
        rgb = rgb / 255.0

        if cb_type == ColorBlindType.PROTANOPIA:
            matrix = np.array([
                [0.567, 0.433, 0],
                [0.558, 0.442, 0],
                [0, 0.242, 0.758]
            ])
        elif cb_type == ColorBlindType.DEUTERANOPIA:
            matrix = np.array([
                [0.625, 0.375, 0],
                [0.7, 0.3, 0],
                [0, 0.3, 0.7]
            ])
        else:
            matrix = np.array([
                [0.95, 0.05, 0],
                [0, 0.433, 0.567],
                [0, 0.475, 0.525]
            ])

        cb_rgb = matrix @ rgb
        return np.clip(cb_rgb, 0, 1) * 255

    def _color_distance(self, c1, c2):
        return np.linalg.norm(c1 - c2)

    
    def _assign_broken_colors(self):
        category_names = list(self.categories.keys())
        cb_types = self.colorblind_types

        for _ in range(3000):

            base = np.array([
                random.randint(60, 200),
                random.randint(60, 200),
                random.randint(60, 200)
            ])

            colors = [base]

            for _ in range(1, len(category_names)):
                found = False

                for _ in range(300):
                    candidate = np.clip(base + np.random.uniform(-50, 50, 3), 0, 255)

                    if all(
                        self._color_distance(
                            self._simulate_cb(base, cb),
                            self._simulate_cb(candidate, cb)
                        ) < 35
                        for cb in cb_types
                    ):
                        colors.append(candidate)
                        found = True
                        break

                if not found:
                    break

            if len(colors) == len(category_names):
                for i, name in enumerate(category_names):
                    self.categories[name] = self.categories[name].model_copy(
                        update={"hex": rgb_to_hex(*colors[i].astype(int))}
                    )
                return

        raise ValueError("Failed to generate CB-confusing colors")

    def _render_scatter_plot(self, ):   # return an image
        # a 16x10 inch figure at matplotlib's default DPI (100) gives a 
        # 1000x1600x3 image = 4.8 million integers being sent as JSON. 
        # That's why the browser is struggling to display it.
        
        # fig = plt.figure(figsize=(16, 10))        # A heavy computation bug
        fig = plt.figure(figsize=(8, 6), dpi=72)  # default is 100

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

        # scatter_plot = base64.b64encode(self._state.scatter_plot.tobytes()).decode('utf-8')
        return CBAObservation(
            scatter_plot=self._state.scatter_plot,
            scatter_plot_shape = self._state.scatter_plot_shape,  # read from state directly
            hex_code_per_category=hex_code_per_category_observed,
            shape_per_category=shape_per_category_observed,
            colorblind_types=self._state.colorblind_types,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            is_done=self.is_done
        )
    
    def _encode_image(self, image: np.ndarray) -> str:
        img = Image.fromarray(image)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue())
        return img_b64.decode('utf-8')


<<<<<<< HEAD
    def reset(self, task: Optional[str] = None, cb_types: Optional[List[str]] = None) -> CBAObservation:
=======
    def reset(self, task: str = "easy") -> CBAObservation:
>>>>>>> 04b891dc56c9109712d241495d7bc229c9e302e2
        """
        Reset the environment.

        Args:
        task: Difficulty level - 'easy', 'medium', or 'hard'.
              If None, keeps the current task set during __init__.

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
        if task is not None:
<<<<<<< HEAD
            assert task in ['easy', 'medium', 'hard'], "task value must be from ['easy', 'medium', 'hard']"
            self.task = task
            self.task_config = TASK_CONFIGS[self.task]
=======
            assert task in ['easy', 'medium', 'hard'], "task must be 'easy', 'medium', or 'hard'"
            self.task = task
        
        self.task_config = TASK_CONFIGS[self.task]  # set task_config based on the finalized task

>>>>>>> 04b891dc56c9109712d241495d7bc229c9e302e2
        self.steps_taken = 0
        self.is_done = False
        self.is_solved = False
        self.fixes_applied = []
        self.delta_E_matrix = {}

        if cb_types:
            self.colorblind_types = [ColorBlindType(ct) for ct in cb_types]
        else:
            all_cb_types = list(ColorBlindType)
            self.colorblind_types = random.sample(all_cb_types, self.task_config["no_of_cb_types"])

        print("Starting reset...")
        self.categories = self._generate_categories()
        print("Categories generated")
        self._assign_broken_colors()
        print("Broken colors assigned")
        self._compute_delta_e_matrix()
        print("Delta E matrix computed")

        self.current_image = self._render_scatter_plot()
        print("Scatter plot rendered")

        scatter_plot_b64 = self._encode_image(self.current_image)

        self._state = CBAState(
            episode_id = str(uuid4()), 
            step_count=0, 
            scatter_plot = scatter_plot_b64, 
            scatter_plot_shape = list(self.current_image.shape),
            categories=self.categories,
            colorblind_types = self.colorblind_types, 
            max_steps=self.task_config['max_steps'],
            fixes_applied = self.fixes_applied,
            delta_E_matrix = self.delta_E_matrix,
            is_solved = self.is_solved
        )
        
        return self._build_observation()

    def _compute_delta_e_matrix(self):  # returns nothing
        pairs = list(combinations(self.categories.keys(), 2))   # pairs = [("Class A", "Class B"), ("Class A", "Class C"), ...]

        for cat_i, cat_j in pairs:
            for_each_cb_delta_value = {}
            hex_i, hex_j = self.categories[cat_i].hex, self.categories[cat_j].hex

            for cb_type in self.colorblind_types:
                delta = compute_delta_e(hex_i, hex_j, cb_type)
                for_each_cb_delta_value[cb_type.value] = delta

            # self.delta_E_matrix[(cat_i, cat_j)] = for_each_cb_delta_value
            self.delta_E_matrix[f"{cat_i}|{cat_j}"] = for_each_cb_delta_value
    
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
            cat_i, cat_j = pair.split("|")
            
            shape_i, shape_j = self.categories[cat_i].shape, self.categories[cat_j].shape
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
        # self._state.scatter_plot = self.current_image
        self._state.scatter_plot = self._encode_image(self.current_image)
        self._state.scatter_plot_shape = list(self.current_image.shape)
        self._state.categories = self.categories
        self._state.fixes_applied = self.fixes_applied
        self._state.delta_E_matrix = self.delta_E_matrix
        self._state.is_solved = self.is_solved

        observation = self._build_observation()
        observation.reward = reward
        observation.done = self.is_done
        
        # return self._build_observation(), reward, self.is_done, {}
        return observation