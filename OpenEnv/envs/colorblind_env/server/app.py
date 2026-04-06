"""
FastAPI application for the Color Blind Accessibility Environment.

This module creates an HTTP server that exposes CBA Environment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.atari_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
#     uvicorn envs.colorblind_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4
#
# Or run directly:
#     python -m envs.colorblind_env.server.app
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

import gradio as gr
import json
import base64
import io
from PIL import Image
import numpy as np

def build_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    import openenv.core.env_server.gradio_ui as gradio_ui
    readme_content = gradio_ui._readme_section(metadata)
    display_title = gradio_ui.get_gradio_display_title(metadata, fallback=title)
    
    with gr.Blocks(title=display_title) as demo:
        with gr.Row():
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

            with gr.Column(scale=2, elem_classes="col-right"):
                img_display = gr.Image(label="Scatter Plot", type="numpy", interactive=False)
                obs_display = gr.Markdown(value="# Playground\n\nClick **Reset** to start a new episode.")
                
                step_inputs = []
                with gr.Group():
                    for field in action_fields:
                        name = field["name"]
                        ph = field.get("placeholder", "")
                        inp = gr.Textbox(label=name.replace("_", " ").title(), placeholder=ph)
                        step_inputs.append(inp)
                        
                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")
                        state_btn = gr.Button("Get state", variant="secondary")

                    status = gr.Textbox(label="Status", interactive=False)
                    raw_json = gr.Code(label="Raw JSON response", language="json", interactive=False)

        def extract_img(data):
            try:
                obs = data.get("observation", {})
                b64 = obs.get("scatter_plot")
                if b64:
                    if isinstance(b64, str) and b64.startswith("data:image"):
                        b64 = b64.split(",", 1)[1]
                    img_data = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    return np.array(img)
            except Exception as e:
                import traceback
                traceback.print_exc()
            return None
            
        def clean_json(data):
            import copy
            c = copy.deepcopy(data)
            if "scatter_plot" in c.get("observation", {}):
                c["observation"]["scatter_plot"] = "<base64 image hidden for clarity>"
            return json.dumps(c, indent=2)

        async def do_reset():
            try:
                data = await web_manager.reset_environment()
                return extract_img(data), clean_json(data), "Environment reset successfully."
            except Exception as e:
                return None, "", f"Error: {e}"
                
        async def do_step(*args):
            try:
                action_data = {}
                for field, val in zip(action_fields, args):
                    if val is not None and val != "":
                        action_data[field["name"]] = val
                data = await web_manager.step_environment(action_data)
                return extract_img(data), clean_json(data), "Step complete."
            except Exception as e:
                return None, "", f"Error: {e}"

        def get_state_sync():
            try:
                return clean_json(web_manager.get_state())
            except Exception as e:
                return f"Error: {e}"

        reset_btn.click(do_reset, outputs=[img_display, raw_json, status])
        step_btn.click(do_step, inputs=step_inputs, outputs=[img_display, raw_json, status])
        state_btn.click(get_state_sync, outputs=[raw_json])
        
    return demo

# Override the default UI builder directly
web_interface.build_gradio_app = build_gradio_app

app = create_app(
    create_cba_environment, CBAAction, CBAObservation, env_name="cba_env"
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
