import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from stable_baselines3 import PPO
from gym_wrapper import CBAGymEnv
from server.utils import simulate_cb
from models import ColorBlindType


# -----------------------------
# Plot Function
# -----------------------------
def plot_categories(categories, title):
    fig, ax = plt.subplots(figsize=(5, 5))

    for name, cat in categories.items():
        pts = np.array(cat.points)

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=cat.hex,
            label=name,
            s=60
        )

    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    return fig


# -----------------------------
# Simulate Colorblind Plot
# -----------------------------
def simulate_plot(categories, cb_type):
    simulated = {}

    for name, cat in categories.items():
        sim_hex = simulate_cb(cat.hex, cb_type)

        simulated[name] = type(cat)(
            hex=sim_hex,
            points=cat.points
        )

    return simulated


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return PPO.load("color_rl_model")


# -----------------------------
# Run RL Episode
# -----------------------------
def run_rl():
    env = CBAGymEnv()
    model = load_model()

    obs, _ = env.reset()

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    return env.env.state.categories


# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("🎨 Colorblind Accessibility (RL)")

cb_type = st.selectbox(
    "Colorblind Type",
    ["deuteranopia", "protanopia"]
)

# -----------------------------
# Run Button
# -----------------------------
if st.button("🚀 Run RL"):

    env = CBAGymEnv()
    obs, _ = env.reset()

    # ORIGINAL
    original = copy.deepcopy(env.env.state.categories)

    # RUN RL
    model = load_model()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    fixed = env.env.state.categories

    # SIMULATED
    simulated = simulate_plot(original, cb_type)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original")
        st.pyplot(plot_categories(original, "Original"))

    with col2:
        st.subheader("Colorblind View")
        st.pyplot(plot_categories(simulated, "Simulated"))

    with col3:
        st.subheader("RL Fixed")
        st.pyplot(plot_categories(fixed, "Fixed"))

    st.success("Done!")

# -----------------------------
# 📊 Learning Curve
# -----------------------------
st.markdown("---")
st.subheader("📈 Learning Progress")

if os.path.exists("learning_curve.png"):
    st.image("learning_curve.png", caption="Model Learning Curve (Reward vs Time)")
else:
    st.info("No learning curve found. Train the model using ppo_train.py first.")