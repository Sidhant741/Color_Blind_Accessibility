from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv, run_async

def run_inference(return_initial=False):
    env = CBAGymEnv()
    model = PPO.load("ppo_cba_model")

    obs, _ = env.reset()

    # SAVE ORIGINAL
    # original_categories = env.env.categories.copy()
    import copy
    original_categories = copy.deepcopy(env.env.categories)

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

    fixed_categories = env.env.categories

    if return_initial:
        return original_categories, fixed_categories

    return fixed_categories