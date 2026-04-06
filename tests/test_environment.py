import sys
import os
import pytest
import numpy as np
import random
from unittest.mock import MagicMock

# Add the environment directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.environment import CBAEnvironment
from models import CBAAction, FixType, Shape, ColorBlindType
from server.config import TASK_CONFIGS

# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class TestCBAEnvironment:
    """Test suite for CBAEnvironment"""

    @pytest.fixture
    def env_easy(self):
        return CBAEnvironment(task="easy")

    @pytest.fixture
    def env_medium(self):
        return CBAEnvironment(task="medium")

    @pytest.fixture
    def env_hard(self):
        return CBAEnvironment(task="hard")

    # ----------------------------------------------------------------------
    #  Initialization tests
    # ----------------------------------------------------------------------
    def test_init_easy(self, env_easy):
        assert env_easy.task == "easy"
        assert env_easy.task_config == TASK_CONFIGS["easy"]
        assert env_easy._state is None
        assert env_easy.categories is None
        assert env_easy.steps_taken == 0
        assert env_easy.is_done is False
        assert env_easy.is_solved is False

    def test_init_medium(self, env_medium):
        assert env_medium.task == "medium"
        assert env_medium.task_config == TASK_CONFIGS["medium"]

    def test_init_hard(self, env_hard):
        assert env_hard.task == "hard"
        assert env_hard.task_config == TASK_CONFIGS["hard"]

    def test_init_invalid_task(self):
        with pytest.raises(AssertionError):
            CBAEnvironment(task="invalid")

    # ----------------------------------------------------------------------
    #  Reset and observation tests (with attribute access)
    # ----------------------------------------------------------------------
    def test_reset(self, env_easy):
        obs = env_easy.reset()
        # obs is a CBAObservation (Pydantic model)
        assert hasattr(obs, "scatter_plot")
        assert hasattr(obs, "hex_code_per_category")
        assert hasattr(obs, "shape_per_category")
        assert hasattr(obs, "colorblind_types")
        assert hasattr(obs, "step_count")
        assert hasattr(obs, "max_steps")
        assert hasattr(obs, "is_done")

        # Check internal state
        assert env_easy.steps_taken == 0
        assert env_easy.is_done is False
        assert env_easy.is_solved is False
        assert env_easy.fixes_applied == []

        n_cats = TASK_CONFIGS["easy"]["n_categories"]
        assert len(env_easy.categories) == n_cats

    def test_generate_categories(self, env_easy):
        # Directly call _generate_categories without resetting (so no broken colors)
        categories = env_easy._generate_categories()
        n_cats = TASK_CONFIGS["easy"]["n_categories"]
        assert len(categories) == n_cats
        for i, (name, cat) in enumerate(categories.items()):
            assert name == f"Class {chr(65+i)}"
            assert cat.shape == Shape.CIRCLE
            assert cat.hex == "#FFFFFF"
            assert len(cat.points) == TASK_CONFIGS["easy"]["n_points"]
            for x, y in cat.points:
                assert 0 <= x <= 1
                assert 0 <= y <= 1

    def test_assign_broken_colors(self, env_easy):
        # First generate categories without broken colors
        env_easy.categories = env_easy._generate_categories()
        old_hexes = [cat.hex for cat in env_easy.categories.values()]
        # All should be white
        assert all(h == "#FFFFFF" for h in old_hexes)

        # Set colorblind types (needed for broken assignment)
        all_cb_types = list(ColorBlindType)
        env_easy.colorblind_types = random.sample(all_cb_types, env_easy.task_config["no_of_cb_types"])

        # Assign broken colors
        env_easy._assign_broken_colors()
        new_hexes = [cat.hex for cat in env_easy.categories.values()]
        # Should have changed
        assert not all(h == "#FFFFFF" for h in new_hexes)

        # Check that each pair is broken (delta_E < threshold)
        threshold = TASK_CONFIGS["easy"]["delta_E_threshold"]
        # Compute delta_E matrix manually for verification
        env_easy._compute_delta_e_matrix()
        for pair, deltas in env_easy.delta_E_matrix.items():
            for cb_type, delta in deltas.items():
                assert delta < threshold, f"Pair {pair} delta_E={delta} >= {threshold} for {cb_type}"

    def test_render_scatter_plot(self, env_easy):
        env_easy.reset()
        img = env_easy._render_scatter_plot()
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.dtype == np.uint8

    def test_build_observation(self, env_easy):
        obs = env_easy.reset()
        # Verify attribute values
        assert obs.scatter_plot == env_easy._state.scatter_plot
        assert obs.scatter_plot_shape == env_easy._state.scatter_plot_shape
        assert obs.hex_code_per_category == {name: cat.hex for name, cat in env_easy.categories.items()}
        assert obs.shape_per_category == {name: cat.shape for name, cat in env_easy.categories.items()}
        assert obs.colorblind_types == [cb.value for cb in env_easy.colorblind_types]
        assert obs.step_count == env_easy.steps_taken
        assert obs.max_steps == env_easy.task_config["max_steps"]
        assert obs.is_done == env_easy.is_done

    # ----------------------------------------------------------------------
    #  Delta_E matrix and done conditions
    # ----------------------------------------------------------------------
    def test_compute_delta_e_matrix(self, env_easy):
        env_easy.reset()
        # Clear existing matrix
        env_easy.delta_E_matrix = {}
        env_easy._compute_delta_e_matrix()
        categories = list(env_easy.categories.keys())
        n_cats = len(categories)
        expected_pairs = [
            f"{categories[i]}|{categories[j]}"
            for i in range(n_cats)
            for j in range(i + 1, n_cats)
        ]
        assert set(env_easy.delta_E_matrix.keys()) == set(expected_pairs)
        for pair, deltas in env_easy.delta_E_matrix.items():
            assert isinstance(deltas, dict)
            for cb_type in env_easy.colorblind_types:
                assert cb_type.value in deltas
                assert isinstance(deltas[cb_type.value], float)

    def test_check_done_easy(self, env_easy):
        env_easy.reset()
        # Not solved initially
        env_easy._check_done()
        assert env_easy.is_done is False

        # Force a solved state by manually setting delta_E_matrix to all above threshold
        # We'll set each pair to have delta_E >= threshold for all CB types
        threshold = env_easy.task_config['delta_E_threshold']
        for pair in env_easy.delta_E_matrix:
            for cb_type in env_easy.colorblind_types:
                env_easy.delta_E_matrix[pair][cb_type.value] = threshold + 1
        env_easy._check_done()
        assert env_easy.is_done is True

    def test_check_done_medium_hard(self, env_medium):
        # For medium/hard tasks, done can be due to solved OR max steps.
        env_medium.task = "medium"
        env_medium.task_config = TASK_CONFIGS["medium"]
        threshold = env_medium.task_config['delta_E_threshold']

        # Setup a single pair with low delta (unsolved)
        env_medium.delta_E_matrix = {"A|B": {ColorBlindType.DEUTERANOPIA.value: threshold - 5}}
        env_medium.colorblind_types = [ColorBlindType.DEUTERANOPIA]
        env_medium.steps_taken = 0
        env_medium.is_solved = False
        env_medium.is_done = False

        # Not solved, steps < max_steps -> not done
        env_medium._check_done()
        assert env_medium.is_done is False

        # Solved: set delta above threshold
        env_medium.delta_E_matrix["A|B"][ColorBlindType.DEUTERANOPIA.value] = threshold + 1
        env_medium._check_done()
        assert env_medium.is_done is True

        # Max steps reached
        env_medium.is_solved = False
        env_medium.steps_taken = env_medium.task_config['max_steps']
        env_medium._check_done()
        assert env_medium.is_done is True

        # ----------------------------------------------------------------------
    #  Reward computation tests (with proper setup)
    # ----------------------------------------------------------------------
    def test_compute_reward_core(self, env_easy):
        # Create a controlled state with delta_E matrix and dummy categories
        env_easy.delta_E_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: 10.0, ColorBlindType.PROTANOPIA.value: 20.0},
            "A|C": {ColorBlindType.DEUTERANOPIA.value: 30.0, ColorBlindType.PROTANOPIA.value: 40.0},
            "B|C": {ColorBlindType.DEUTERANOPIA.value: 50.0, ColorBlindType.PROTANOPIA.value: 60.0},
        }
        env_easy.colorblind_types = [ColorBlindType.DEUTERANOPIA, ColorBlindType.PROTANOPIA]

        # Provide dummy categories with any shapes (only needed for shape_score, but for easy task shape_weight=0)
        env_easy.categories = {
            "A": MagicMock(shape=Shape.CIRCLE),
            "B": MagicMock(shape=Shape.CIRCLE),
            "C": MagicMock(shape=Shape.CIRCLE),
        }
        reward = env_easy._compute_reward(action=None, previous_delta_E_matrix=None)
        # Expected core reward = average of normalized scores:
        # Pair 1: avg delta = (10+20)/2 =15, norm = min(15/40,1)=0.375
        # Pair 2: avg =35, norm=0.875
        # Pair 3: avg=55, norm=1.0
        # total = 2.25, average = 0.75
        assert abs(reward - 0.75) < 0.01

    def test_compute_reward_shape_weight(self, env_medium):
        # Setup with different shapes
        env_medium.categories = {
            "A": MagicMock(shape=Shape.CIRCLE),
            "B": MagicMock(shape=Shape.SQUARE),
        }
        env_medium.delta_E_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: 10.0}
        }
        env_medium.colorblind_types = [ColorBlindType.DEUTERANOPIA]
        reward = env_medium._compute_reward(action=None, previous_delta_E_matrix=None)
        # color_score = 10/40 = 0.25, shape_score = 1
        # pair_reward = 0.8*0.25 + 0.2*1 = 0.2 + 0.2 = 0.4
        assert abs(reward - 0.4) < 0.01

    def test_compute_reward_penalties(self, env_easy):
        env_easy.reset()
        target = "A"
        threshold = env_easy.task_config['delta_E_threshold']
        previous_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: threshold + 1},
            "A|C": {ColorBlindType.DEUTERANOPIA.value: threshold + 2},
        }
        # Current state same as previous (no change)
        env_easy.delta_E_matrix = previous_matrix
        env_easy.colorblind_types = [ColorBlindType.DEUTERANOPIA]
        env_easy.categories = {
            "A": MagicMock(shape=Shape.CIRCLE),
            "B": MagicMock(shape=Shape.CIRCLE),
            "C": MagicMock(shape=Shape.CIRCLE),
        }
        action = MagicMock(target=target)

        # Compute expected core reward from previous_matrix
        expected_core = 0
        for pair in previous_matrix:
            color_score = sum(previous_matrix[pair].values()) / len(env_easy.colorblind_types)
            norm = min(color_score / 40.0, 1.0)
            expected_core += norm
        expected_core /= len(previous_matrix)

        # Redundant action penalty = -0.05 (target already well distinguished)
        expected_reward = expected_core - 0.05
        reward = env_easy._compute_reward(action, previous_matrix)
        assert abs(reward - expected_reward) < 0.01

    def test_compute_reward_regression_penalty(self, env_easy):
        env_easy.reset()
        target = "A"
        threshold = env_easy.task_config['delta_E_threshold']

        # Previous state: high core but not solved (delta below threshold, but we want regression)
        # Actually regression penalty applies regardless of solved status. We just need core to drop.
        # To avoid redundant penalty, make previous deltas below threshold.
        prev_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: threshold - 1},
            "A|C": {ColorBlindType.DEUTERANOPIA.value: threshold - 1},
        }
        # Current state: lower core
        curr_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: 10.0},
            "A|C": {ColorBlindType.DEUTERANOPIA.value: 10.0},
        }
        env_easy.delta_E_matrix = curr_matrix
        env_easy.colorblind_types = [ColorBlindType.DEUTERANOPIA]
        env_easy.categories = {
            "A": MagicMock(shape=Shape.CIRCLE),
            "B": MagicMock(shape=Shape.CIRCLE),
            "C": MagicMock(shape=Shape.CIRCLE),
        }
        action = MagicMock(target=target)

        # Compute previous and current core manually
        prev_core = 0
        for pair in prev_matrix:
            color_score = sum(prev_matrix[pair].values()) / 1
            norm = min(color_score / 40.0, 1.0)
            prev_core += norm
        prev_core /= 2

        curr_core = 0
        for pair in curr_matrix:
            color_score = sum(curr_matrix[pair].values()) / 1
            norm = min(color_score / 40.0, 1.0)
            curr_core += norm
        curr_core /= 2

        # Regression penalty only (no redundant penalty)
        expected = curr_core - 0.1 * (prev_core - curr_core)
        reward = env_easy._compute_reward(action, prev_matrix)
        assert abs(reward - expected) < 0.01

    def test_compute_reward_efficiency_bonus(self, env_hard):
        # For hard task, efficiency bonus is added when core_reward >= 0.8 and steps < max_steps
        env_hard.delta_E_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: 40.0},
        }
        env_hard.colorblind_types = [ColorBlindType.DEUTERANOPIA]
        env_hard.steps_taken = 3
        env_hard.task_config['max_steps'] = 10
        env_hard.task_config['efficiency_weight'] = 0.1
        # Use different shapes so shape_score=1
        env_hard.categories = {
            "A": MagicMock(shape=Shape.CIRCLE),
            "B": MagicMock(shape=Shape.SQUARE),
        }
        reward = env_hard._compute_reward(action=None, previous_delta_E_matrix=None)
        # core_reward = 1.0 (color 1.0 * 0.6 + shape 1.0 * 0.4 = 1.0)
        # bonus = (1 - 3/10) * 0.1 = 0.07
        # total = 1.07, capped to 1.0
        assert reward == 1.0

    def test_compute_reward_medium_penalty(self, env_medium):
        # For medium task, penalty for exceeding max steps
        env_medium.delta_E_matrix = {
            "A|B": {ColorBlindType.DEUTERANOPIA.value: 40.0},
        }
        env_medium.colorblind_types = [ColorBlindType.DEUTERANOPIA]
        env_medium.steps_taken = 12
        env_medium.task_config['max_steps'] = 10
        env_medium.categories = {
            "A": MagicMock(shape=Shape.CIRCLE),
            "B": MagicMock(shape=Shape.SQUARE),   # different shapes
        }
        reward = env_medium._compute_reward(action=None, previous_delta_E_matrix=None)
        # core = 1.0 (color 1.0 * 0.8 + shape 1.0 * 0.2 = 1.0)
        # penalty = 0.02 * (12-10) = 0.04 → total = 0.96
        assert abs(reward - 0.96) < 0.01

    # ----------------------------------------------------------------------
    #  Step tests (with proper attribute access)
    # ----------------------------------------------------------------------
    def test_step_recolor(self, env_easy):
        env_easy.reset()
        initial_cat = env_easy.categories["Class A"]
        action = CBAAction(
            target="Class A",
            fix_type=FixType.RECOLOR,
            change_hex="#123456"
        )
        obs = env_easy.step(action)
        # Check category updated
        updated_cat = env_easy.categories["Class A"]
        assert updated_cat.hex == "#123456"
        assert updated_cat.shape == initial_cat.shape
        # Check steps taken incremented
        assert env_easy.steps_taken == 1
        # Check fixes_applied appended
        assert len(env_easy.fixes_applied) == 1
        assert "FixType.RECOLOR Class A → #123456" in env_easy.fixes_applied[0]
        # Check observation contains updated image
        assert hasattr(obs, "scatter_plot")
        # Check reward is computed (non-negative)
        assert 0 <= obs.reward <= 1
        # Done may become True if solved (easy task solved after recolor maybe)
        # We don't assert done state here.

    def test_step_reshape(self, env_easy):
        env_easy.reset()
        initial_cat = env_easy.categories["Class A"]
        action = CBAAction(
            target="Class A",
            fix_type=FixType.RESHAPE,
            change_shape=Shape.SQUARE
        )
        obs = env_easy.step(action)
        updated_cat = env_easy.categories["Class A"]
        assert updated_cat.shape == Shape.SQUARE
        assert updated_cat.hex == initial_cat.hex

    def test_step_invalid_target(self, env_easy):
        env_easy.reset()
        action = CBAAction(
            target="NonExistent",
            fix_type=FixType.RECOLOR,
            change_hex="#123456"
        )
        with pytest.raises(ValueError, match="does not exist"):
            env_easy.step(action)

    def test_step_after_done(self, env_easy):
        env_easy.reset()
        # Force done
        env_easy.is_done = True
        action = CBAAction(
            target="Class A",
            fix_type=FixType.RECOLOR,
            change_hex="#123456"
        )
        with pytest.raises(RuntimeError, match="Episode is already done"):
            env_easy.step(action)

    def test_step_returns_observation(self, env_easy):
        env_easy.reset()
        action = CBAAction(
            target="Class A",
            fix_type=FixType.RECOLOR,
            change_hex="#123456"
        )
        obs = env_easy.step(action)
        # Check that the returned observation is consistent with internal state
        assert obs.hex_code_per_category["Class A"] == "#123456"
        assert obs.step_count == 1
        assert obs.is_done == env_easy.is_done
