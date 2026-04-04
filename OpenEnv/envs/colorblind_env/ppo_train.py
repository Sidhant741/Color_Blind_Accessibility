from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv

env = CBAGymEnv("ws://localhost:8000")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=10000)

# model.save("cba_agent")
model.save("ppo_cba_model")