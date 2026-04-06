---
title: colorblind_env
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
---
# Color_Blind_Accessibility

## Project Overview

This document captures the full discussion about building a Color Blind Accessibility OpenEnv environment — a reinforcement learning environment that trains an AI agent to fix scatter plot data for the color blind users

### Types of Color Blindness Supported

| Type          | Description                       |
|---------------|-----------------------------------|
| DEUTERANOPIA  | Cannot distinguish red and green  |
| Protanopia    | Red appears very dark or black    |
| Tritanopia    | Blue and yellow are confused      |

## Some Concepts : Tasks and Graders

Tasks : It is the specific problem the agent is asked to solve in a single episode. It defines:
* What the starting state looks like — what the agent sees when the episode begins
* What a done state looks like — when the episode ends
* What actions are available to the agent

Grader : It is a function that evaluates how well the agent solved the task. It returns a score between 0.0 and 1.0, with partial credit for partial progress.

### Difficulty Tiers - Overview

| Level   | Task                                                                      | Grader Checks                           |
|---------|---------------------------------------------------------------------------|-----------------------------------------|
| Easy    | Fix a scatter plot with 2 similar color, one CB types, no budget          | Are points now distinguishable?         |
| Medium  | Fix a scatter plot with 5 similar color, two CB types, soft budget        | CIEDE2000 Acceptable                    |
| Hard    | Fix a scatter plot with 10 similar color, three CB types, strict budget   | CIEDE2000 Acceptable with minimum steps |

## Detailed Task Design

Each task is fully specified below - including scenario, what is broken, what the agent must do, how the grader scores it, and what a typical agent score looks like

### Easy Task
* Scenario : The agent receives a simple scatter plot with 2 similar CB colors, and one CB type
* What's wrong : A CB user might not able completely distinguishable points
* What the Agent must do 
    * Detect the color pair
    * Recolor to the safe palette
    * Changing pattern has no value here 
* Grader Check : Are the colors now distinguishable for the CB (through delta_E)
* Expected Baseline agent score : 0.85 - 0.95

### Medium Task
* Scenario : The agent receives a scatter plot with 5-7 similar CB colors, and two CB types
* What's wrong : A CB users might not able completely distinguishable points
* What the Agent must do 
    * Detect the color pair which creating issue for both CB
    * Recolor to the safe palette
    * Changing pattern has little value, but more value still remains to the color  
* Grader Check : Are the colors now distinguishable for the CBs (through delta_E). Also if it exceeds max_no_of_steps , dont stop, rather start giving penalities. And at the end dont give efficiency bonus.
* Expected Baseline agent score : 0.60 - 0.80

### Hard Task
* Scenario : The agent receives a scatter plot with 10-15 similar CB colors, and three CB types
* What's wrong : A CB users might not able completely distinguishable points
* What the Agent must do 
    * Detect the color pair which creating issue for both CB
    * Recolor to the safe palette
    * Changing pattern has value, but higher value still remains to the color  
* Grader Check : Are the colors now distinguishable for the CBs (through delta_E). Also it has to stay within the strict action budget. And at the end give efficieny bonus.
* Expected Baseline agent score : 0.30 - 0.50

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

Agent decides an action() -> Environment validates the action -> Environment applies the action (updates legend_info + redraws image) 
-> Grader runs and scores the new state -> Environment checks if episode is done -> Returns (observation, reward, done, info)

```
# Option 1 — just recolor
action = {
    "target":   "Class A",
    "fix_type": "recolor",
    "new_hex":  "#0077BB",
    "new_shape": None
}

# Option 2 — just add shape
action = {
    "target":    "Class A",
    "fix_type":  "change_shape",
    "new_hex":   None,
    "new_shape": "triangle"
}
```

### Step
Journey of One Action

```
Agent looks at observation decides 
Class A needs fixing calls step()
            ↓
Environement receives the action
            ↓
        Validates it
            ↓
Applies it to internal state
            ↓
Rerenders the scatter plot image
            ↓
Grader runs delta_E simulation
            ↓
    Reward computated
            ↓
        Done check
            ↓
    New observation built
            ↓
Return (obs, reward, done, info)
        back to agent
```

### Reward
1. Give HIGH reward when CB user can distinguish categories
2. Give LOW reward when categories still look identical
3. Give PARTIAL reward for partial progress
4. Not leak the answer to the agent
5. Work consistently across easy/medium/hard

Therefore the Core Signal - delta_E i.e. perceptual color difference between two colors AS SEEN by a CB user

* delta_E = 0.0   → colors look completely identical
* delta_E = 10.0  → barely distinguishable
* delta_E = 20.0  → clearly different
* delta_E = 40.0  → very different
* delta_E = 50+   → maximum difference

| Task   | Color Weight | Shape Weight | Allowed Actions         | Meaning                                                              |
|--------|--------------|--------------|-------------------------|----------------------------------------------------------------------|
| Easy   | 1.0          | 0.0          | recolor only            | Only color matters; shapes are fixed                                 |
| Medium | 0.8          | 0.2          | recolor or change shape | Color is primary; shapes give a small bonus                          |
| Hard   | 0.6          | 0.4          | recolor or change shape | Both color and shape are required; high reward only if both are good |

Other Bonus
* Efficiency Bonus : If hard task is solved well (base reward >= Threshold), the agent gets an extra bonus for finishing quickly
    i.e. bonus = (1 - steps_taken / max_steps ) * 0.1
* Redundant Action Penalty : If the categories are already well-distinguishable for every other category, and agent do the action step,
    then we have to add -ve penalty reward, which tells the agent to not touch categories that are already solved


```
FUNCTION compute_reward(state, previous_state, task, steps_taken, max_steps):
    # ------------------------------------------------------------
    # 1. Set task weights
    # ------------------------------------------------------------
    IF task == "easy":
        color_weight = 1.0, shape_weight = 0.0
    ELSE IF task == "medium":
        color_weight = 0.8, shape_weight = 0.2
    ELSE IF task == "hard":
        color_weight = 0.6, shape_weight = 0.4

    # ------------------------------------------------------------
    # 2. Core reward: average over all category pairs
    # ------------------------------------------------------------
    total_pair_reward = 0
    FOR each pair of categories (i, j):
        # Color score: average over all CB types
        color_score = 0
        FOR each cb_type in state.colorblind_types:
            sim_i = simulate_cb(state.categories[i].hex, cb_type)
            sim_j = simulate_cb(state.categories[j].hex, cb_type)
            delta = CIEDE2000(sim_i, sim_j)
            color_score += min(delta / 40.0, 1.0)
        color_score = color_score / len(state.colorblind_types)

        # Shape score: 1 if shapes differ, else 0
        shape_score = 1 if state.categories[i].shape != state.categories[j].shape else 0

        # Pair reward
        pair_reward = color_weight * color_score + shape_weight * shape_score
        total_pair_reward += min(pair_reward, 1.0)

    core_reward = total_pair_reward / number_of_pairs

    # ------------------------------------------------------------
    # 3. Efficiency bonus (only for hard task)
    # ------------------------------------------------------------
    bonus = 0
    IF task == "hard" AND core_reward >= 0.8:
        bonus = (1.0 - steps_taken / max_steps) * 0.1

    # ------------------------------------------------------------
    # 4. Redundant action penalty (if action was unnecessary)
    # ------------------------------------------------------------
    penalty = 0
    IF action.target_category was already well_distinguished:
        penalty += -0.05

    # ------------------------------------------------------------
    # 5. Regression penalty (if overall reward dropped)
    # ------------------------------------------------------------
    IF previous_state IS NOT None:
        previous_core = compute_core_reward(previous_state)   # call recursively
        IF core_reward < previous_core:
            penalty += -0.1 * (previous_core - core_reward)

    # ------------------------------------------------------------
    # 6. Final reward (capped)
    # ------------------------------------------------------------
    total_reward = core_reward + bonus + penalty
    RETURN clamp(total_reward, 0.0, 1.0)
```

## How the RL loop works

```
Real Webpage / Image
    |
Agent 'sees' it  (reset / state)
    |
Agent takes actions  (recolor, relabel, repattern)
    |
Grader simulates color blind vision
    |
Grader scores how much better it got  (0.0 - 1.0)
    |
Agent learns from that score

```

## Real World Deployment Vision

Once trained, this agent becomes a browser extension that:
* Scans any webpage the user visits
* Detects the user's specific color blindness type automatically
* Applies fixes in real time — recoloring, relabeling, adding patterns
* Gets smarter with every plot it processes

## To-Do (From Scaler Hackathon)

- [X] Setup Environment
- [X] Real-world Task simulation
- [ ] OpenEnv spec compliance
- [X] Easy Task Code Written + Reward
- [ ] Medium Task Code Written + Reward
- [ ] Hard Task Code Written + Reward
- [ ] Baseline Inference Script
- [ ] Deploy to a HuggingFace Space
- [ ] Containerized Execution
- [ ] Documentation
- [ ] HF Space deploys
- [X] Validator
- [ ] Additional Endpoints to Expose
- [ ] Move the hardcoded value to config (Optional)
- [ ] Better logic in _assign_broken_colors()
