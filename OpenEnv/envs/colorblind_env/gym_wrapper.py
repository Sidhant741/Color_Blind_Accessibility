import gymnasium as gym
import numpy as np
from server.environment import CBAEnvironment as ColorEnv


class CBAGymEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.env = ColorEnv(task="easy")

        # 🔥 DYNAMIC CONTINUOUS ACTION SPACE
        n_cats = self.env.config["n_categories"]
        self.action_space = gym.spaces.Box(
            low=np.array([0, -0.2, -0.2, -0.2], dtype=np.float32),
            high=np.array([n_cats - 1, 0.2, 0.2, 0.2], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=np.zeros(n_cats, dtype=np.float32),
            high=np.ones(n_cats, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs_obj = self.env.reset()
        # Extract numpy vector
        obs = np.array(obs_obj.vector, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs_obj = self.env.step(action)
        
        obs = np.array(obs_obj.vector, dtype=np.float32)
        reward = obs_obj.reward
        done = obs_obj.is_done
        
        return obs, reward, done, False, {}