from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv
from callbacks import TrainingLogger
import matplotlib.pyplot as plt
import numpy as np

env = CBAGymEnv()

callback = TrainingLogger()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=50000, callback=callback)

# -----------------------------
# 📊 PLOT LEARNING CURVE
# -----------------------------
rewards = np.array(callback.episode_rewards)

window = 50
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.3, label="Episode Reward")
plt.plot(range(window-1, len(rewards)), smoothed, label="Smoothed", linewidth=2)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("RL Learning Curve (Correct)")
plt.legend()
plt.grid()
plt.savefig("learning_curve.png")
print("Learning curve saved to learning_curve.png")

model.save("color_rl_model")
print("Model saved to color_rl_model.zip")