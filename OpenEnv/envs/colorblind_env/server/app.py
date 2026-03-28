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

Environment variables:
    ATARI_GAME: Game name to serve (default: "pong")
    ATARI_OBS_TYPE: Observation type (default: "rgb")
    ATARI_FULL_ACTION_SPACE: Use full action space (default: "false")
    ATARI_MODE: Game mode (optional)
    ATARI_DIFFICULTY: Game difficulty (optional)
    ATARI_REPEAT_ACTION_PROB: Sticky action probability (default: "0.0")
    ATARI_FRAMESKIP: Frameskip (default: "4")
"""