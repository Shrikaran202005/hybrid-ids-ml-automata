# pages/7_Network_Simulation.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# AUTOMATA RULE (fallback if utils unavailable)
# -----------------------------
try:
    from utils import automata_rule
except ImportError:
    def automata_rule(flow, threshold=2.8):
        """Simple Euclidean-distance rule: returns 1 if Attack, else 0"""
        if isinstance(flow, dict):
            values = np.array(list(flow.values()), dtype=float)
        else:
            values = np.array(flow, dtype=float)
        dist = np.linalg.norm(values - np.mean(values))
        return 1 if dist > threshold else 0

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Network Simulation", layout="wide")
st.title("🌐 Real-Time Network Traffic Simulation")
st.write("""
Simulate network traffic flows and visualize **ML model vs Automata baseline predictions** in real-time.
""")

# -----------------------------
# LOAD DATA
# -----------------------------
processed_file = "data/processed/preprocessed.csv"
model_file = "models/pytorch_model_2class.pth"

if not os.path.exists(processed_file):
    st.error("❌ Preprocessed dataset not found! Please generate `preprocessed.csv`.")
    st.stop()

df = pd.read_csv(processed_file)
if "label" not in df.columns:
    st.error("❌ Dataset must include a 'label' column.")
    st.stop()

# Features and numeric cleaning
features = df.drop("label", axis=1, errors='ignore').columns
df_clean = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

# Map labels to 0/1
label_mapping = {label: 0 if str(label).lower() == "benign" else 1 for label in df["label"].unique()}

# -----------------------------
# DEFINE PYTORCH MODEL
# -----------------------------
class IntrusionNet(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
input_size = len(features)
model = IntrusionNet(input_dim=input_size, num_classes=2)

if os.path.exists(model_file):
    state_dict = torch.load(model_file, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    st.success("✅ ML PyTorch model loaded successfully")
else:
    st.warning("⚠ Model file not found, using untrained model.")

model.eval()

# -----------------------------
# SIMULATION PARAMETERS
# -----------------------------
st.sidebar.subheader("⚙️ Simulation Settings")
num_flows = st.sidebar.slider("Number of flows to simulate", min_value=10, max_value=500, value=50, step=10)
delay = st.sidebar.slider("Delay between flows (seconds)", 0.1, 2.0, 0.5)

st.subheader("🚦 Simulation Dashboard")
ml_col, auto_col = st.columns(2)
ml_col.markdown("### 🤖 ML Model Predictions")
auto_col.markdown("### ⚙️ Automata Baseline Predictions")

ml_chart = ml_col.empty()
auto_chart = auto_col.empty()
ml_last_pred = ml_col.empty()
auto_last_pred = auto_col.empty()

# -----------------------------
# COUNTERS
# -----------------------------
ml_attack_count = 0
ml_benign_count = 0
auto_attack_count = 0
auto_benign_count = 0

results_df = pd.DataFrame(columns=["Flow", "Benign", "Attack", "Type"])

# -----------------------------
# RUN SIMULATION
# -----------------------------
st.write(f"Simulating **{num_flows} network flows**...")
progress_bar = st.progress(0)
status_text = st.empty()

for i in range(num_flows):
    # Random sample flow
    sample = df_clean.sample(1).iloc[0]
    flow = sample.to_dict()
    flow_df = pd.DataFrame([flow])

    # ----- ML Prediction -----
    try:
        flow_tensor = torch.tensor(flow_df.values, dtype=torch.float32)
        with torch.no_grad():
            y_pred = torch.argmax(model(flow_tensor), dim=1).item()
        ml_pred = y_pred  # 0=Benign, 1=Attack
    except Exception as e:
        st.warning(f"⚠ ML prediction failed at flow {i+1}: {e}")
        ml_pred = np.random.choice([0, 1])

    if ml_pred == 1:
        ml_attack_count += 1
        ml_status = "Attack"
    else:
        ml_benign_count += 1
        ml_status = "Benign"

    # ----- Automata Prediction -----
    try:
        auto_pred = int(automata_rule(flow))
    except Exception:
        auto_pred = np.random.choice([0, 1])

    if auto_pred == 1:
        auto_attack_count += 1
        auto_status = "Attack"
    else:
        auto_benign_count += 1
        auto_status = "Benign"

    # Save results
    results_df = pd.concat([
        results_df,
        pd.DataFrame([
            {"Flow": i+1, "Benign": ml_benign_count, "Attack": ml_attack_count, "Type": "ML"},
            {"Flow": i+1, "Benign": auto_benign_count, "Attack": auto_attack_count, "Type": "Automata"}
        ])
    ], ignore_index=True)

    # Update charts
    ml_chart.bar_chart(results_df[results_df["Type"]=="ML"][["Benign","Attack"]].tail(1))
    auto_chart.bar_chart(results_df[results_df["Type"]=="Automata"][["Benign","Attack"]].tail(1))

    # Show last predictions
    ml_last_pred.markdown(f"**Last ML Flow Prediction:** `{ml_status}`")
    auto_last_pred.markdown(f"**Last Automata Flow Prediction:** `{auto_status}`")

    progress_bar.progress((i + 1) / num_flows)
    status_text.text(f"Flow {i+1}/{num_flows} processed...")
    time.sleep(delay)

progress_bar.empty()
status_text.empty()

st.success("✅ Simulation completed!")
st.write("ML Model vs Automata baseline predictions were simulated in real-time.")
