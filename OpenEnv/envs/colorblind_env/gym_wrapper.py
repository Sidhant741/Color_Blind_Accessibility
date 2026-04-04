
import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from client import CBAEnv
from models import CBAAction, FixType, Shape

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

class CBAGymEnv(gym.Env):

    def __init__(self, url="ws://localhost:8000"):
        super().__init__()

        self.env = CBAEnv(url)

        self.max_categories = 10
        self._needs_reset = True

        # ACTION SPACE
        # [category_id, fix_type, value_index]
        self.action_space = spaces.MultiDiscrete([
            self.max_categories,  # category index
            2,                    # recolor / reshape
            10                    # palette index OR shape index
        ])

        # OBSERVATION SPACE
        # (colors + shapes)
        obs_dim = self.max_categories * 4  # (r,g,b,shape)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.palette = [
            "#0077BB", "#33BBEE", "#009988",
            "#EE7733", "#CC3311", "#EE3377",
            "#BBBBBB"
        ]

        self.shapes = list(Shape)

        self.category_keys = []

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        kwargs = {}
        if options is not None and "preset_categories" in options:
            kwargs["preset_categories"] = {
                k: {"hex": v.hex, "shape": v.shape.value, "points": v.points}
                for k, v in options["preset_categories"].items()
            }

        result = run_async(self.env.reset(**kwargs))
        obs = result.observation

        self.category_keys = list(obs.hex_code_per_category.keys())
        self._needs_reset = False

        return self._obs_to_vector(obs), {}

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):

        # If previous step ended episode → reset BEFORE stepping
        if self._needs_reset:
            reset_result = run_async(self.env.reset())
            obs = reset_result.observation
            self._needs_reset = False
            return self._obs_to_vector(obs), 0.0, False, False, {}

        cat_idx, fix_type, value_idx = action

        if cat_idx >= len(self.category_keys):
            return self._dummy_step()

        target = self.category_keys[cat_idx]

        if fix_type == 0:
            hex_val = self.palette[value_idx % len(self.palette)]
            act = CBAAction(
                target=target,
                fix_type=FixType.RECOLOR,
                change_hex=hex_val
            )
        else:
            shape_val = self.shapes[value_idx % len(self.shapes)]
            act = CBAAction(
                target=target,
                fix_type=FixType.RESHAPE,
                change_shape=shape_val
            )

        try:
            result = run_async(self.env.step(act))
        except RuntimeError as e:
            # 🔥 CRITICAL: handle server "episode finished" error
            if "Episode finished" in str(e):
                reset_result = run_async(self.env.reset())
                obs = reset_result.observation
                self._needs_reset = False
                return self._obs_to_vector(obs), 0.0, False, False, {}
            else:
                raise

        obs = result.observation
        reward = result.reward
        done = result.done

        terminated = done
        truncated = False

        # mark for next step
        if done:
            self._needs_reset = True

        return self._obs_to_vector(obs), reward, terminated, truncated, {} 
    # OBS → VECTOR


    # -------------------------
    def _obs_to_vector(self, obs):

        vec = []

        for k in self.category_keys:
            hex_code = obs.hex_code_per_category[k]

            r = int(hex_code[1:3], 16) / 255.0
            g = int(hex_code[3:5], 16) / 255.0
            b = int(hex_code[5:7], 16) / 255.0

            shape = list(Shape).index(obs.shape_per_category[k]) / len(Shape)

            vec.extend([r, g, b, shape])

        # padding
        while len(vec) < self.max_categories * 4:
            vec.extend([0, 0, 0, 0])

        return np.array(vec, dtype=np.float32)

    def _dummy_step(self):
        return (
            np.zeros(self.max_categories * 4),
            0.0,
            True,
            False,
            {}
        )