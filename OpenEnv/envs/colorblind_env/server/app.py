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
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openenv.core.env_server import create_app
import openenv.core.env_server.web_interface as web_interface

def custom_generate_placeholder(field_name: str, field_info: dict) -> str:
    name = field_name.lower()
    if "target" in name:
        return "Class A, Class B"
    elif "fix_type" in name:
        return "Recolor or Reshape"
    elif "change_hex" in name:
        return "e.g. #FF0000"
    elif "change_shape" in name:
        return "O, X, ^, +, s, p, *"
    return f"Enter {field_name.replace('_', ' ')}..."

web_interface._generate_placeholder = custom_generate_placeholder
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
