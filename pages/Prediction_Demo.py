# pages/6_Prediction_Demo.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Prediction Demo", layout="wide")
st.title("🎯 Intrusion Detection: Prediction Demo")
st.write("""
Choose a model, enter sample network flow values, and compare predictions between the **ML model (PyTorch)** and the **Automata rule-based system**.
""")

# -----------------------------
# LOAD DATA
# -----------------------------
processed_file = "data/processed/preprocessed.csv"
if not os.path.exists(processed_file):
    st.error("❌ Preprocessed dataset not found! Run Preprocessing first.")
    st.stop()

df = pd.read_csv(processed_file)
features = df.drop("label", axis=1).columns.tolist()

# -----------------------------
# DEFINE PyTorch MODEL
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
# MODEL SELECTOR
# -----------------------------
st.subheader("🧠 Select ML Model")

available_models = {
    "2-Class Model": ("models/pytorch_model_2class.pth", 2),
    "27-Class Model": ("models/pytorch_model.pth", 27),
}

selected_model_name = st.selectbox("Choose a PyTorch model", list(available_models.keys()))
model_file, num_classes = available_models[selected_model_name]

if not os.path.exists(model_file):
    st.error(f"❌ Selected model file not found: {model_file}")
    st.stop()

# -----------------------------
# LOAD MODEL
# -----------------------------
input_size = len(features)
model = IntrusionNet(input_dim=input_size, num_classes=num_classes)

state_dict = torch.load(model_file, map_location=torch.device("cpu"))

# If last layer mismatched, skip it
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
if missing_keys or unexpected_keys:
    st.warning(f"⚠ Some layers were skipped due to size mismatch: {missing_keys + unexpected_keys}")

model.eval()
st.success(f"✅ Loaded: {selected_model_name}")

# -----------------------------
# AUTOMATA RULE
# -----------------------------
def automata_rule(row):
    """Simple Automata baseline using threshold rules."""
    if "FlowBytes" in row and row["FlowBytes"] > 1_000_000:
        return 1
    if "DestinationPort" in row and row["DestinationPort"] not in [80, 443, 22, 21, 25]:
        return 1
    return 0

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("🖊️ Enter Network Flow Features")
user_input = {}

num_demo_features = min(10, len(features))
for feature in features[:num_demo_features]:
    val = st.number_input(f"{feature}", value=float(df[feature].mean()))
    user_input[feature] = val

# Fill remaining features with dataset mean
for feature in features[num_demo_features:]:
    user_input[feature] = float(df[feature].mean())

user_df = pd.DataFrame([user_input], columns=features)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Predict"):
    user_df_clean = user_df.replace([np.inf, -np.inf], np.nan).fillna(df.mean())
    X_input = torch.tensor(user_df_clean.values, dtype=torch.float32)

    # ML Prediction
    with torch.no_grad():
        logits = model(X_input)
        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob = torch.softmax(logits, dim=1).max().item()

    # Determine class meaning
    if num_classes == 2:
        ml_status = "BENIGN" if pred_class == 1 else "ATTACK"
    else:
        ml_status = f"Class {pred_class} (multi-class model)"

    # Automata Prediction
    auto_pred = automata_rule(user_input)
    auto_status = "ATTACK" if auto_pred == 1 else "BENIGN"

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🤖 ML Model Prediction (PyTorch)**")
        if "ATTACK" in ml_status:
            st.error(f"🚨 Prediction: {ml_status}")
        else:
            st.success(f"✅ Prediction: {ml_status}")
        st.info(f"Confidence: {pred_prob:.2f}")

    with col2:
        st.markdown("**⚙️ Automata Baseline Prediction**")
        if auto_status == "ATTACK":
            st.error(f"🚨 Prediction: {auto_status}")
        else:
            st.success(f"✅ Prediction: {auto_status}")

    st.info("ℹ️ ML model prediction depends on training and scaling. Automata uses fixed threshold rules.")
