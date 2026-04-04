import random
import numpy as np
from uuid import uuid4

from openenv.core.env_server import Environment
from models import State, Category, ColorBlindType
from .utils import compute_delta_e, hex_to_rgb, rgb_to_hex
from .config import TASK_CONFIGS


class CBAEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task="easy"):
        super().__init__()
        self.config = TASK_CONFIGS[task]
        self._state = None
        self.reset()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def reset(self):
        self._state = State(
            categories={
                f"C{i}": Category(
                    hex=self.random_hex(),
                    points=[(random.random(), random.random()) for _ in range(self.config["n_points"])]
                )
                for i in range(self.config["n_categories"])
            },
            colorblind_types=[ColorBlindType.DEUTERANOPIA],
            step_count=0,
            max_steps=self.config["max_steps"]
        )

        return self._build_observation()

    def _build_observation(self):
        # For gym/internal use, return numpy. For server, return CBAObservation.
        # This is a bit of a hack to support both.
        from models import CBAObservation
        
        hex_map = {k: v.hex for k, v in self._state.categories.items()}
        
        return CBAObservation(
            vector=self._get_obs().tolist(),
            hex_code_per_category=hex_map,
            colorblind_types=self._state.colorblind_types,
            is_done=self._state.done,
            reward=self._state.reward
        )

    def random_hex(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # 🔥 APPLY CONTINUOUS ACTION
    def apply_action(self, action):
        keys = list(self.state.categories.keys())

        target_idx = int(np.clip(action[0], 0, len(keys)-1))
        target = keys[target_idx]

        delta = action[1:]  # ΔR, ΔG, ΔB

        r, g, b = hex_to_rgb(self.state.categories[target].hex)

        # normalize to [0,1]
        r /= 255
        g /= 255
        b /= 255

        # apply delta
        r = np.clip(r + delta[0], 0, 1)
        g = np.clip(g + delta[1], 0, 1)
        b = np.clip(b + delta[2], 0, 1)

        # back to 0–255
        r, g, b = int(r*255), int(g*255), int(b*255)

        self.state.categories[target].hex = rgb_to_hex(r, g, b)

    def step(self, action):
        from models import CBAAction
        
        if isinstance(action, CBAAction):
            if action.continuous_action:
                self.apply_action(action.continuous_action)
            else:
                # Handle old discrete actions if necessary
                pass
        else:
            # Assume it's a raw numpy array from gym
            self.apply_action(action)

        reward = self.compute_reward()
        self._state.reward = reward
        self._state.step_count += 1

        done = reward > 0.95 or self._state.step_count >= self._state.max_steps
        self._state.done = done

        obs = self._build_observation()
        # Gym expects (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
        # But server expects Observation object.
        # We'll return based on context or just return both and let gym_wrapper handle it.
        return obs

    # 🔥 STRONGER REWARD (MIN DISTANCE)
    def compute_reward(self):
        keys = list(self.state.categories.keys())
        threshold = self.config["delta_E_threshold"]

        min_delta = float("inf")

        for i in range(len(keys)):
            for j in range(i+1, len(keys)):

                c1 = self.state.categories[keys[i]].hex
                c2 = self.state.categories[keys[j]].hex

                for cb in self.state.colorblind_types:
                    delta = compute_delta_e(c1, c2, cb.value)
                    min_delta = min(min_delta, delta)

        # normalize
        reward = min(min_delta / threshold, 1.0)

        # 🔥 strong penalty if below threshold
        if min_delta < threshold:
            reward -= 0.5

        return max(0.0, reward)

    def _get_obs(self):
        return np.array([
            int(cat.hex[1:], 16) / 0xFFFFFF
            for cat in self.state.categories.values()
        ], dtype=np.float32)