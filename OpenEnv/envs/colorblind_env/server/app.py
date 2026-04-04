"""
FastAPI application for the Color Blind Accessibility Environment
"""

import os
import logging
from fastapi.middleware.cors import CORSMiddleware

from openenv.core.env_server import create_app

from models import CBAAction, CBAObservation
from .environment import CBAEnvironment


# ---------------------------------
# LOGGING
# ---------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------
# CONFIG
# ---------------------------------

VALID_TASKS = ["easy", "medium", "hard"]

task = os.getenv("CBA_TASK", "easy")
if task not in VALID_TASKS:
    raise ValueError(f"Invalid task '{task}'. Must be one of {VALID_TASKS}")

PORT = int(os.getenv("PORT", 8000))

logger.info(f"Starting CBA Environment with task={task}")


# ---------------------------------
# ENV FACTORY
# ---------------------------------

def create_cba_environment():
    return CBAEnvironment(task=task)


# ---------------------------------
# APP CREATION
# ---------------------------------

app = create_app(
    create_cba_environment,
    CBAAction,
    CBAObservation,
    env_name="cba_env_v1"
)


# ---------------------------------
# CORS (for frontend / demos)
# ---------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------
# HEALTH CHECK
# ---------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "cba_env", "task": task}


# ---------------------------------
# MAIN (LOCAL RUN)
# ---------------------------------

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()