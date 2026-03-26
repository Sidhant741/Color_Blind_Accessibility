# Color_Blind_Accessibility

## Project Overview

This document captures the full discussion about building a Color Blind Accessibility OpenEnv environment — a reinforcement learning environment that trains an AI agent to fix scatter plot data for the color blind users

### Types of Color Blindness Supported

| Type         | Description                       |
|--------------|-----------------------------------|
| Deueranopia  | Cannot distinguish red and green  |
| Protanopia   | Red appears very dark or black    |
| Tritanopia   | Blue and yellow are confused      |

## Some Concepts : Tasks and Graders

Tasks : It is the specific problem the agent is asked to solve in a single episode. It defines:
* What the starting state looks like — what the agent sees when the episode begins
* What a done state looks like — when the episode ends
* What actions are available to the agent

Grader : It is a function that evaluates how well the agent solved the task. It returns a score between 0.0 and 1.0, with partial credit for partial progress.

### Difficulty Tiers - Overview

| Level   | Task                                         | Grader Checks                          |
|---------|----------------------------------------------|----------------------------------------|
| Easy    | Fix a scatter plot with two similar color, one CB types, no budget          | Are points now distinguishable?        |
| Medium  | Fix a scatter plot with 5-7 similar color, two CB types, soft budget        | CIEDE2000 Acceptable                    |
| Hard    | Fix a scatter plot with 10-15 similar color, three CB types, strict budget  | CIEDE2000 Acceptable with minimum steps |

## Detailed Task Design

## Environment Details

### State vs Observation
* State : Full Truth of the environment
* Observation : What the agent is allowed to see

| Field                   | Description                                                          | Will be a State | Will be a Observation |
|-------------------------|----------------------------------------------------------------------|-----------------|-----------------------|
| scatter_plot_image      | np.array image of the scatter plot                                   | True            | True                  |
| hex_code_per_category   | hex code of each category (e.g. #FF0000)                           | True            | True                  |
| shape_per_category      | shape of each category (e.g. circle)                                 | True            | True                  |
| all_points_per_category | points per category in the format of (x,y)                           | Debatable       | False                 |
| colorblind_type         | Which CB Type is the user is or agent solving for                    | True            | True                  |
| step                    | Current Step                                                         | True            | True                  |
| max_steps               | Max no of steps for the episode (different per difficulty)           | True            | True                  |
| fixes_applied           | List of fixes done per step (e.g "recolor Class A → #0077BB")      | True            | False                 |
| delta_E_matrix          | A table (matrix) of color differences between multiple samples.      | True            | False                 |
| is_done                 | calculated based on each step and threshold value                    | True            | True                  |


```
state = {
    # the chart itself
    "scatter_plot": np.array,        # full image
    "categories": {
        "Class A": {
            "hex":       "#FF0000",
            "shape":     "circle",
            "points":    [(x1,y1), (x2,y2), ...],  # all data points },
        "Class B": {
            "hex":       "#00FF00", 
            "shape":     "circle",
            "points":    [(x3,y3), (x4,y4), ...], }
        },

    # user context
    "colorblind_type": ["deuteranopia", "protonopia"],

    # episode context
    "step":            3,
    "max_steps":       10,
    "fixes_applied":   ["recolor Class A → #0077BB"],

    # grader's internal truth (hidden from agent)
    "delta_E_matrix": {
        ("Class A", "Class B"): 1.2,   # still broken
    },
    "is_solved":       False
}
```
### Reset
* Pick which task to load(easy, medium or hard)
* Generate a fresh scatter plot
* Assign broken colors to categories
* Set the colorblind type(s)
* Reset the stop counter
* Return the first observation

Example of reset, will modify it later
```
def reset(self, task=None):

    if task is None:
        self.current_task = random.choice(["easy", "medium", "hard"])
    
    assert task == ("easy", "medium", "hard"), "y must be ('easy', 'med', 'hard')"

    self.current_task = task

    config = TASK_CONFIGS[self.current_task]
    # config looks like:
    # {
    #   "n_categories":    2,
    #   "colorblind_types": ["deuteranopia"],
    #   "steps_allowed":   None,
    #   "n_points":        100
    # }

    self.categories = self._generate_categories(config)
    # generates something like:
    # {
    #   "Class A": { points: [(x,y)...], hex: "#FF0000", shape: "circle" }
    #   "Class B": { points: [(x,y)...], hex: "#00FF00", shape: "circle" }
    # }

    # ── STEP 4: assign broken colors ──
    # deliberately picks colors that are
    # indistinguishable for the CB type
    self.categories = self._assign_broken_colors(
        self.categories,
        config["colorblind_types"]
    )

    # ── STEP 5: render the image ──
    self.current_image = self._render_scatter_plot(self.categories)

    # ── STEP 6: reset episode tracking ──
    self.steps_taken    = 0
    self.steps_allowed  = config["steps_allowed"]
    self.fixes_applied  = []
    self.is_done        = False

    # ── STEP 7: return first observation ──
    return self._build_observation()
```

### Action

### Step

## How the RL loop works

## What the Agent Learns over time

## Real World Deployment Vision

## To-Do

- [ ] Setup Environment
- [ ] 