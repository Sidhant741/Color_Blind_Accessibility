# This is working fine , but not for regenerated functionality
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy

from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv
from server.utils import simulate_cb_hex as simulate_cb
from models import ColorBlindType


# -----------------------------
# 🎯 Plot Function
# -----------------------------
def plot_categories(categories, title="Plot"):
    fig, ax = plt.subplots(figsize=(5, 5))

    for name, cat in categories.items():
        points = np.array(cat.points)
        color = cat.hex
        shape = cat.shape

        ax.scatter(points[:, 0], points[:, 1],
                   c=color,
                   marker=shape,
                   label=name,
                   s=60)

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
# 🤖 Run RL Inference
# -----------------------------
@st.cache_resource
def load_model():
    return PPO.load("ppo_cba_model")


def run_inference(preset_categories=None, colorblind_types=None):
    if preset_categories is not None:
        import json
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

        if preset_categories is not None:
            import os
            try:
                os.remove("/tmp/preset_layout.json")
            except Exception:
                pass

        from gym_wrapper import run_async
        
        # 🔥 IMPORTANT: fetch state through async wrapper
        initial_state = run_async(env.env.state())
        original_categories = initial_state.categories
        colorblind_types = initial_state.colorblind_types

        done = False
        step_idx = 0
        while not done and step_idx < 50:
            # Use deterministic=False to yield different RL paths conditionally!
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env.step(action)
            step_idx += 1

        final_state = run_async(env.env.state())
        fixed_categories = final_state.categories

        return original_categories, fixed_categories, colorblind_types
    finally:
        # CRITICAL FIX for connection leaks
        from gym_wrapper import run_async
        try:
            run_async(env.env.close())
        except Exception:
            pass


# -----------------------------
# 🌐 Streamlit UI
# -----------------------------
st.set_page_config(page_title="Colorblind AI", layout="wide")

st.title("🎨 Color Blind Accessibility using RL")
st.write("Fixing scatter plots for colorblind users using Reinforcement Learning")

# Select CB type
cb_type = st.selectbox(
    "Select Colorblind Type",
    ["deuteranopia", "protanopia", "tritanopia"]
)

cb_enum = ColorBlindType(cb_type)

if "original_layout" not in st.session_state:
    st.session_state["original_layout"] = None

# Run button
col_b1, col_b2 = st.columns(2)

with col_b1:
    run_new = st.button("🚀 Run Model (New Plot)")

with col_b2:
    # Only active if we have mapped a plot loaded in RAM historically
    run_again = st.button("🔄 Regenerate Fix (Same Plot)", disabled=(st.session_state["original_layout"] is None))

if run_new or run_again:

    with st.spinner("Running RL model..."):
        if run_new:
            st.session_state["original_layout"] = None
            st.session_state["current_cb_types"] = None
        
        preset = st.session_state["original_layout"]
        cb_presets = st.session_state.get("current_cb_types")
        
        # Run RL
        original, fixed, cb_types = run_inference(preset_categories=preset, colorblind_types=cb_presets)
        
        # If it was a regeneration, we MUST trust our previous 'original' 
        # but the server should have returned the same thing anyway.
        if run_again and preset is not None:
            # Keep the one we had to ensure absolute UI stability 
            # (prevents tiny floating point shifts or race conditions)
            original = preset
            
        st.session_state["original_layout"] = original
        st.session_state["current_cb_types"] = cb_types
        # Save the CB types the server chose (or we provided) for future regenerations
        # We fetch them from the 'original' categories metadata or state if needed, 
        # but actually run_inference returns the categories. 
        # Let's make run_inference also return the cb_types if they changed.

    simulated = simulate_plot(original, cb_enum)

    # -----------------------------
    # 📊 Display Plots Side-by-Side
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Plot")
        st.pyplot(plot_categories(original, "Original"))

    with col2:
        st.subheader("Simulated (Colorblind)")
        st.pyplot(plot_categories(simulated, "Colorblind View"))

    with col3:
        st.subheader("Fixed Plot (RL Output)")
        st.pyplot(plot_categories(fixed, "Fixed"))

    st.success("Visualization complete!")