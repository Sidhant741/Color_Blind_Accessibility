"""
Inference Script for Color Blind Accessibility Environment
===========================================================

This script uses an LLM (via OpenAI client) to solve CBA tasks.

MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
import json

from client import CBAEnv
from models import CBAAction, FixType, Shape

IMAGE_NAME = os.getenv("colorblind-env")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = os.getenv("CBA_BENCHMARK", "colorblind_env")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.1

# Maps validator task IDs → env task names
TASKS = {
    "task_easy":   "easy",
    "task_medium": "medium",
    "task_hard":   "hard",
}

SYSTEM_PROMPT = """
You are fixing a colorblind scatter plot.
Choose one action.

Return ONLY valid JSON in this format:
{"target":"Class A","fix_type":"recolor","change_hex":"#0077BB"}

Rules:
- fix_type must be "recolor" or "reshape"
- If recolor: provide change_hex, leave change_shape null
- If reshape: provide change_shape (one of: o, ^, *, x, +, p, s), leave change_hex null
""".strip()


def parse_action(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return {}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_action_prompt(obs) -> str:
    cats = list(obs.hex_code_per_category.keys())
    return (
        f"The scatter plot has these categories: {cats}\n"
        f"Current colors: {obs.hex_code_per_category}\n"
        f"Current shapes: {obs.shape_per_category}\n"
        f"Colorblind types affected: {obs.colorblind_types}\n\n"
        f"Pick ONE category to fix. Reply ONLY with valid JSON, no explanation:\n"
        f'Example: {{"target": "{cats[0]}", "fix_type": "recolor", "change_hex": "#0077BB"}}'
    )


def get_model_message(client: OpenAI, prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request FAILED: {exc}", flush=True)
        return "hello"


async def run_episode(client: OpenAI, task_id: str, env_task: str) -> None:
    env = CBAEnv(base_url="ws://localhost:7860")
    await env.connect()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=env_task)
        last_obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            prompt = build_action_prompt(last_obs)
            raw = get_model_message(client, prompt)
            action_data = parse_action(raw)

            if not action_data or "target" not in action_data:
                target = next(iter(last_obs.hex_code_per_category.keys()))
                action = CBAAction(target=target, fix_type=FixType.RECOLOR, change_hex="#0077BB")
                action_str = f"recolor {target} #0077BB"
            else:
                target = action_data["target"]
                fix_type = FixType(action_data["fix_type"])
                if fix_type == FixType.RECOLOR:
                    action = CBAAction(target=target, fix_type=fix_type, change_hex=action_data["change_hex"])
                    action_str = f"recolor {target} {action_data['change_hex']}"
                else:
                    action = CBAAction(target=target, fix_type=fix_type, change_shape=Shape(action_data["change_shape"]))
                    action_str = f"reshape {target} {action_data['change_shape']}"

            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_obs = result.observation
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.5
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id, env_task in TASKS.items():
        await run_episode(client, task_id, env_task)


if __name__ == "__main__":
    asyncio.run(main())