import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client import CBAEnv
from models import CBAAction, FixType

import asyncio

async def main():
    env = CBAEnv("ws://localhost:8000")

    reset_result = await env.reset()
    obs = reset_result.observation

    print("Initial observation received")

    # Pick first category
    target = list(obs.hex_code_per_category.keys())[0]

    action = CBAAction(
        target=target,
        fix_type=FixType.RECOLOR,
        change_hex="#0077BB"
    )

    result = await env.step(action)
    obs, reward, done = result.observation, result.reward, result.done

    print("Step successful!")
    print("Reward:", reward)
    print("Done:", done)

if __name__ == "__main__":
    asyncio.run(main())