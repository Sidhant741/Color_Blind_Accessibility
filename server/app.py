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

repo_root = Path(__file__).resolve().parent.parent
src_dir = str(repo_root / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from openenv.core.env_server import create_web_interface_app
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
# task = os.getenv("CBA_TASK", "easy") # We will set task in reset() instead of here, to allow dynamic task selection per episode

def create_cba_environment():
    return CBAEnvironment()

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
                obs_display = gr.Markdown(value="# Playground\n\nChoose a difficulty and click **Reset** to start.")
                
                with gr.Group():
                    mode_radio = gr.Radio(
                        ["easy", "medium", "hard"], 
                        label="Difficulty Mode", 
                        value="easy"
                    )
                    
                    cb_type_easy = gr.Dropdown(
                        choices=["deutronopia", "protanopia", "titanopia"],
                        label="Type of Colorblind",
                        value="deutronopia",
                        visible=True
                    )
                    cb_type_medium = gr.Dropdown(
                        choices=[
                            "deutronopia and protanopia", 
                            "deutronopia and titanopia", 
                            "protanopia and tritanopia"
                        ],
                        label="Type of Colorblind",
                        value="deutronopia and protanopia",
                        visible=False
                    )
                    
                    gr.HTML("<hr>")
                    
                    inputs_dict = {}
                    for field in action_fields:
                        name = field["name"]
                        label = name.replace("_", " ").title()
                        ph = field.get("placeholder", "")
                        
                        if name == "fix_type":
                            inputs_dict[name] = gr.Dropdown(
                                choices=["recolor", "reshape"], 
                                label=label, 
                                value="recolor",
                                visible=True 
                            )
                        elif name == "change_shape":
                            inputs_dict[name] = gr.Textbox(
                                label="Shape", 
                                placeholder="O, X, ^, +, s, p, *",
                                visible=True
                            )
                        elif name == "change_hex":
                            inputs_dict[name] = gr.Textbox(label=label, placeholder=ph, visible=True)
                        else:
                            inputs_dict[name] = gr.Textbox(label=label, placeholder=ph)
                    
                    step_inputs = [inputs_dict[f["name"]] for f in action_fields]
                        
                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")
                        state_btn = gr.Button("Get state", variant="secondary")

                    status = gr.Textbox(label="Status", interactive=False)
                    raw_json = gr.Code(label="Raw JSON response", language="json", interactive=False)

        def update_ui_visibility(mode, fix_type):
            """Update visibility of fields based on mode and fix type."""
            updates = {}
            # Defaults for all modes
            updates[cb_type_easy] = gr.update(visible=False)
            updates[cb_type_medium] = gr.update(visible=False)
            
            updates[inputs_dict["fix_type"]] = gr.update(visible=True)
            if fix_type == "recolor":
                updates[inputs_dict["change_hex"]] = gr.update(visible=True)
                updates[inputs_dict["change_shape"]] = gr.update(visible=False)
            else:
                updates[inputs_dict["change_hex"]] = gr.update(visible=False)
                updates[inputs_dict["change_shape"]] = gr.update(visible=True)
            
            if mode == "easy":
                updates[inputs_dict["target"]] = gr.update(placeholder="Class A, Class B")
                updates[cb_type_easy] = gr.update(visible=True)
            elif mode == "medium":
                updates[inputs_dict["target"]] = gr.update(placeholder="Class A, Class B, Class C, Class D, Class E")
                updates[cb_type_medium] = gr.update(visible=True)
            elif mode == "hard":
                updates[inputs_dict["target"]] = gr.update(placeholder="Class A, Class B, Class C, Class D, Class E, Class F, Class G, Class H, Class I, Class J")
            
            return [
                updates[cb_type_easy],
                updates[cb_type_medium],
                updates[inputs_dict["fix_type"]], 
                updates[inputs_dict["change_hex"]], 
                updates[inputs_dict["change_shape"]],
                updates[inputs_dict["target"]]
            ]

        mode_radio.change(
            fn=update_ui_visibility,
            inputs=[mode_radio, inputs_dict["fix_type"]],
            outputs=[cb_type_easy, cb_type_medium, inputs_dict["fix_type"], inputs_dict["change_hex"], inputs_dict["change_shape"], inputs_dict["target"]]
        )
        
        inputs_dict["fix_type"].change(
            fn=update_ui_visibility,
            inputs=[mode_radio, inputs_dict["fix_type"]],
            outputs=[cb_type_easy, cb_type_medium, inputs_dict["fix_type"], inputs_dict["change_hex"], inputs_dict["change_shape"], inputs_dict["target"]]
        )

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

        async def do_reset(mode, cb_easy, cb_medium):
            cb_mapping = {
                "deutronopia": "deuteranopia",
                "protanopia": "protanopia",
                "titanopia": "tritanopia",
                "tritanopia": "tritanopia"
            }
            
            cb_types = None
            if mode == "easy":
                cb_types = [cb_mapping.get(cb_easy, cb_easy)]
            elif mode == "medium":
                parts = cb_medium.split(" and ")
                cb_types = [cb_mapping.get(p.strip(), p.strip()) for p in parts]
            
            try:
                data = await web_manager.reset_environment(task=mode, cb_types=cb_types)
                return extract_img(data), clean_json(data), f"Environment reset to {mode} mode with {cb_types} successfully."
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

        reset_btn.click(do_reset, inputs=[mode_radio, cb_type_easy, cb_type_medium], outputs=[img_display, raw_json, status])
        step_btn.click(do_step, inputs=step_inputs, outputs=[img_display, raw_json, status])
        state_btn.click(get_state_sync, outputs=[raw_json])
        
    return demo

# Override the default UI builder directly
web_interface.build_gradio_app = build_gradio_app

app = create_web_interface_app(
    create_cba_environment, CBAAction, CBAObservation,
    env_name="cba_env",
    max_concurrent_envs=10,
)

# Redirect root to the Gradio UI so HF Spaces shows the app
from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/web/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
