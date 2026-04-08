"""Easy task grader.
Score based on color distinguishability for 1 colorblind type.
color_weight = 1.0, shape_weight = 0.0. No bonus or penalty.
"""
import numpy as np


def grade(
    delta_E_matrix: dict,
    categories: dict,
    colorblind_types: list,
    steps_taken: int,
    task_config: dict,
    action=None,
    previous_delta_E_matrix: dict = None,
) -> float:
    """Grade easy task performance.

    Args:
        delta_E_matrix: Current per-pair delta-E values keyed as "Cat A|Cat B".
        categories: Dict of category name -> Category object (has .shape, .hex).
        colorblind_types: List of active ColorBlindType values.
        steps_taken: Number of steps taken so far in the episode.
        task_config: Config dict for the current task (color_weight, shape_weight, etc).
        action: Last action taken (unused for easy, kept for uniform signature).
        previous_delta_E_matrix: Delta-E matrix before the last action (unused for easy).

    Returns:
        Score in [0.001, 0.999].
    """
    color_weight = task_config['color_weight']
    shape_weight = task_config['shape_weight']

    total_score = 0
    for pair in delta_E_matrix:
        color_score = sum(delta_E_matrix[pair].values()) / len(colorblind_types)
        norm_color_score = min(color_score / 40.0, 1.0)

        cat_i, cat_j = pair.split("|")
        shape_score = 1 if categories[cat_i].shape != categories[cat_j].shape else 0

        total_score += color_weight * norm_color_score + shape_weight * shape_score

    core_reward = total_score / len(delta_E_matrix)
    return round(float(np.clip(core_reward, 0.001, 0.999)), 4)


class EasyGrader:
    """Callable grader class for the easy task."""

    def __call__(
        self,
        delta_E_matrix: dict,
        categories: dict,
        colorblind_types: list,
        steps_taken: int,
        task_config: dict,
        action=None,
        previous_delta_E_matrix: dict = None,
    ) -> float:
        return grade(delta_E_matrix, categories, colorblind_types, steps_taken, task_config, action, previous_delta_E_matrix)