import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
import json
import os

from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv, run_async
from server.utils import simulate_cb_hex as simulate_cb
from models import ColorBlindType

# -----------------------------
# 🎯 Plot Function
# -----------------------------
def plot_categories(categories, title="Plot"):
    fig, ax = plt.subplots(figsize=(5, 5))
    if categories:
        for name, cat in categories.items():
            points = np.array(cat.points)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=cat.hex,
                marker=cat.shape,
                label=name,
                s=60
            )
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig

# -----------------------------
# 🎨 Simulate Colorblind View
# -----------------------------
def simulate_plot(categories, cb_type):
    simulated_categories = {}
    for name, cat in categories.items():
        sim_hex = simulate_cb(cat.hex, cb_type)
        simulated_categories[name] = type(cat)(
            hex=sim_hex,
            shape=cat.shape,
            points=cat.points
        )
    return simulated_categories

# -----------------------------
# 🤖 Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return PPO.load("ppo_cba_model")

# -----------------------------
# 🚀 RL Inference
# -----------------------------
def run_inference(preset_categories=None, colorblind_types=None):
    if preset_categories is not None:
        payload = {
            "categories": {
                k: {"hex": v.hex, "shape": v.shape.value if hasattr(v.shape, "value") else v.shape, "points": v.points}
                for k, v in preset_categories.items()
            }
        }
        if colorblind_types is not None:
            payload["colorblind_types"] = [t.value if hasattr(t, "value") else t for t in colorblind_types]
        
        with open("/tmp/preset_layout.json", "w") as f:
            json.dump(payload, f)

    env = CBAGymEnv()
    model = load_model()

    try:
        # Just normal reset, backend will handle picking up the file automatically
        obs, _ = env.reset()
        
        # After reset, we might want to manually clean the file if the server didn't
        if preset_categories is not None:
            try: os.remove("/tmp/preset_layout.json")
            except: pass

        # Fetch state through async wrapper
        initial_state = run_async(env.env.state())
        original_categories = initial_state.categories
        cb_types = initial_state.colorblind_types

        done = False
        step_idx = 0
        while not done and step_idx < 50:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            step_idx += 1

        final_state = run_async(env.env.state())
        fixed_categories = final_state.categories

        return original_categories, fixed_categories, cb_types
    finally:
        try: run_async(env.env.close())
        except: pass

# -----------------------------
# 🌐 Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="Colorblind AI", layout="wide")
st.title("🎨 Color Blind Accessibility using RL")
st.write("Fixing scatter plots for colorblind users using Reinforcement Learning")

# -----------------------------
# 🎛️ Controls
# -----------------------------
cb_type = st.selectbox(
    "Select Colorblind Type",
    ["deuteranopia", "protanopia", "tritanopia"]
)
cb_enum = ColorBlindType(cb_type)

# -----------------------------
# 🧠 Session State Init
# -----------------------------
if "original_layout" not in st.session_state:
    st.session_state["original_layout"] = None
if "fixed_layout" not in st.session_state:
    st.session_state["fixed_layout"] = None
if "current_cb_types" not in st.session_state:
    st.session_state["current_cb_types"] = None

# -----------------------------
# 🔘 Buttons
# -----------------------------
col_b1, col_b2 = st.columns(2)
with col_b1:
    run_new = st.button("🚀 Run Model (New Plot)")
with col_b2:
    run_again = st.button(
        "🔄 Regenerate Fix (Same Plot)",
        disabled=(st.session_state["original_layout"] is None)
    )

# -----------------------------
# 🚀 Execution Logic
# -----------------------------
if run_new or run_again:
    with st.spinner("Processing RL Inference..."):
        if run_new:
            st.session_state["original_layout"] = None
            st.session_state["current_cb_types"] = None
        
        preset = st.session_state["original_layout"]
        cb_presets = st.session_state["current_cb_types"]
        
        orig, fixed, cb_types = run_inference(preset_categories=preset, colorblind_types=cb_presets)
        
        # UI Stability preservation
        if run_again and preset is not None:
             st.session_state["original_layout"] = preset # keep exact user vision
        else:
             st.session_state["original_layout"] = orig
             
        st.session_state["fixed_layout"] = fixed
        st.session_state["current_cb_types"] = cb_types

# -----------------------------
# 📊 Visualization (Persistent)
# -----------------------------
if st.session_state["original_layout"] is not None:
    original = st.session_state["original_layout"]
    fixed = st.session_state["fixed_layout"]
    simulated = simulate_plot(original, cb_enum)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Plot")
        st.pyplot(plot_categories(original, "Original"))
    with col2:
        st.subheader("Simulated (Colorblind View)")
        st.pyplot(plot_categories(simulated, "Colorblind View"))
    with col3:
        st.subheader("RL Output (Fixed)")
        if fixed is not None:
            st.pyplot(plot_categories(fixed, "Fixed"))

    st.success("✅ Visualization complete!")