import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Sklearn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Intrusion Detection Model", layout="wide")
st.title("🔮 Intrusion Detection Using ML")
st.caption("Train, evaluate and predict attacks on the CICIDS2017 dataset")

# -----------------------------
# FILE PATHS
# -----------------------------
MODEL_PATH = "models/pytorch_model_2class.pth"
SCALER_PATH = "models/scaler_2class.pkl"
LABELMAP_PATH = "models/label_map_2class.pkl"
DATA_PATH = "data/processed/preprocessed.csv"

os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL CLASS
# -----------------------------
class IntrusionNet(nn.Module):
    def __init__(self, input_dim, num_classes):
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
# DATA LOADING & PREPARATION
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error("❌ Preprocessed dataset not found. Please upload or generate `preprocessed.csv`.")
    st.stop()

df = pd.read_csv(DATA_PATH, low_memory=False)
if "label" not in df.columns:
    st.error("❌ Dataset missing 'label' column.")
    st.stop()

# ✅ Collapse all attacks into "ATTACK" and keep "BENIGN" as is
df["label"] = df["label"].astype(str).fillna("BENIGN")
df["label"] = df["label"].apply(lambda x: "BENIGN" if x.strip().upper() == "BENIGN" else "ATTACK")

# Encode labels into 0 (BENIGN) and 1 (ATTACK)
le = LabelEncoder()
y_encoded = le.fit_transform(df["label"])
label_map = {i: cls for i, cls in enumerate(le.classes_)}

X = df.drop("label", axis=1).select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean()).astype(np.float32)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and label map
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_map, LABELMAP_PATH)

# -----------------------------
# TRAINING / LOADING MODEL
# -----------------------------
if os.path.exists(MODEL_PATH):
    st.success("✅ Found pretrained 2-class model! Loading it...")
    model = IntrusionNet(input_dim=X.shape[1], num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
else:
    st.warning("⚙ No model found. Training a new 2-class model...")

    # Split and oversample
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    st.write(f"After SMOTE balancing: {Counter(y_train_res)}")

    # Convert to tensors
    train_data = TensorDataset(
        torch.tensor(X_train_res, dtype=torch.float32),
        torch.tensor(y_train_res, dtype=torch.long)
    )
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Train model
    model = IntrusionNet(input_dim=X_train_res.shape[1], num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    st.subheader("🚀 Training Model")
    progress = st.progress(0)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        progress.progress((epoch + 1) / epochs)
    st.success("✅ Model trained successfully!")
    torch.save(model.state_dict(), MODEL_PATH)

# -----------------------------
# EVALUATION
# -----------------------------
st.header("📊 Model Evaluation")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=128, shuffle=False)

all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for Xb, yb in test_loader:
        out = model(Xb)
        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1 Score", f"{f1:.3f}")

st.markdown("### 🧾 Classification Report")
st.text(classification_report(all_labels, all_preds, target_names=list(label_map.values()), zero_division=0))

st.markdown("### 🔥 Confusion Matrix")
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="YlGnBu",
            xticklabels=list(label_map.values()),
            yticklabels=list(label_map.values()),
            fmt="d", ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
st.pyplot(fig)

# -----------------------------
# PREDICTION UI
# -----------------------------
st.header("🎯 Real-Time Attack Prediction")
attack_types = sorted(list(label_map.values()))
selected_class = st.selectbox("Choose sample class to test:", attack_types)

sample = df[df["label"] == selected_class].sample(1, random_state=42).drop("label", axis=1)
st.info(f"Modify feature values for sample of: **{selected_class}**")

user_inputs = {}
with st.expander("🧩 Adjust Feature Inputs"):
    for feature in sample.columns[:25]:  # limit to first 25
        val = float(sample[feature].iloc[0])
        user_inputs[feature] = st.number_input(feature, value=val)
    for feature in sample.columns[25:]:
        user_inputs[feature] = float(sample[feature].iloc[0])

user_df = pd.DataFrame([user_inputs])
user_scaled = scaler.transform(user_df)
user_tensor = torch.tensor(user_scaled, dtype=torch.float32).to(device)

if st.button("🔍 Predict Attack"):
    model.eval()
    with torch.no_grad():
        output = model(user_tensor)
        pred_code = torch.argmax(output, dim=1).item()
    pred_label = label_map[pred_code].strip().upper()

    if pred_label == "ATTACK":
        st.markdown(f"<h2 style='color:red;'>⚠️ ATTACK DETECTED — {pred_label}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color:green;'>✅ BENIGN TRAFFIC DETECTED</h2>", unsafe_allow_html=True)

st.caption("Developed with ❤️ using Streamlit + PyTorch (2-Class Model)")
