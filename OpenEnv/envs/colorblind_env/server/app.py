"""
FastAPI application for the Color Blind Accessibility Environment.

This module creates an HTTP server that exposes CBA Environment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.atari_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.atari_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.atari_env.server.app
"""

import os
from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from ..models import CBAAction, CBAObservation
    from .environment import CBAEnvironment
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    # Standalone imports (when running via uvicorn server.app:app)
    from models import CBAAction, CBAObservation
    from server.environment import CBAEnvironment

# Get configuration from environment variables
task = os.getenv("CBA_TASK", "easy")

def create_cba_environment():
    return CBAEnvironment(task=task)

app = create_app(
    create_cba_environment, CBAAction, CBAObservation, env_name="cba_env"
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
