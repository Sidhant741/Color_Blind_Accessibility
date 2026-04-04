"""
Configuration for Color Blind Accessibility Environment
"""

TASK_CONFIGS = {

    # ---------------------------------
    # EASY
    # ---------------------------------
    "easy": {
        "n_categories": 3,
        "no_of_cb_types": 1,
        "max_steps": 10,
        "n_points": 50,

        # Reward weights
        "color_weight": 1.0,
        "shape_weight": 0.0,

        # Reward behavior
        "efficiency_bonus": False,
        "efficiency_weight": 0.1,

        "step_penalty_weight": 0.0,
        "redundant_penalty": 0.05,
        "regression_penalty": 0.1,

        # Perception threshold
        "delta_E_threshold": 20.0,

        # Color confusion generation
        "l_shift_range": (0.05, 0.15),

        # Rendering (enable for demo, disable for training)
        "render": True,
    },


    # ---------------------------------
    # MEDIUM
    # ---------------------------------
    "medium": {
        "n_categories": 5,
        "no_of_cb_types": 2,
        "max_steps": 20,
        "n_points": 100,

        "color_weight": 0.8,
        "shape_weight": 0.2,

        "efficiency_bonus": False,
        "efficiency_weight": 0.1,

        "step_penalty_weight": 0.02,
        "redundant_penalty": 0.05,
        "regression_penalty": 0.1,

        "delta_E_threshold": 20.0,

        "l_shift_range": (0.1, 0.25),

        "render": False,
    },


    # ---------------------------------
    # HARD
    # ---------------------------------
    "hard": {
        "n_categories": 10,
        "no_of_cb_types": 3,
        "max_steps": 20,
        "n_points": 200,

        "color_weight": 0.6,
        "shape_weight": 0.4,

        "efficiency_bonus": True,
        "efficiency_weight": 0.1,

        "step_penalty_weight": 0.03,
        "redundant_penalty": 0.05,
        "regression_penalty": 0.1,

        "delta_E_threshold": 15.0,

        "l_shift_range": (0.2, 0.4),

        "render": False,
    }
}