"""Hard task grader.
Score based on color + shape distinguishability for 3 colorblind types.
color_weight = 0.6, shape_weight = 0.4.
Efficiency bonus (up to +efficiency_weight if core >= 0.8 within budget),
redundant action penalty (-0.05), and regression penalty (-0.1 * drop).
No soft over-budget penalty — environment stops the episode at max_steps.
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
    """Grade hard task performance.

    Args:
        delta_E_matrix: Current per-pair delta-E values keyed as "Cat A|Cat B".
        categories: Dict of category name -> Category object (has .shape, .hex).
        colorblind_types: List of active ColorBlindType values.
        steps_taken: Number of steps taken so far in the episode.
        task_config: Config dict for the current task (color_weight, shape_weight,
                     max_steps, delta_E_threshold, efficiency_weight, etc).
        action: Last action taken (used for redundant action penalty check).
        previous_delta_E_matrix: Delta-E matrix before the last action (used for
                                  regression penalty).

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
    bonus = 0
    penalty = 0

    # Efficiency bonus
    if core_reward >= 0.8:
        bonus = (1 - steps_taken / task_config['max_steps']) * task_config['efficiency_weight']

    # Redundant action + regression penalty
    if previous_delta_E_matrix is not None and action is not None:
        target = action.target
        threshold = task_config['delta_E_threshold']

        target_pairs = [pair for pair in previous_delta_E_matrix if target in pair]
        already_solved = all(
            delta >= threshold
            for pair in target_pairs
            for delta in previous_delta_E_matrix[pair].values()
        )
        if already_solved:
            penalty -= 0.05

        previous_total = sum(
            min(sum(previous_delta_E_matrix[pair].values()) / len(colorblind_types) / 40.0, 1.0)
            for pair in previous_delta_E_matrix
        )
        previous_core = previous_total / len(previous_delta_E_matrix)

        if core_reward < previous_core:
            penalty -= 0.1 * (previous_core - core_reward)

    return round(float(np.clip(core_reward + bonus + penalty, 0.001, 0.999)), 4)


class HardGrader:
    """Callable grader class for the hard task."""

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