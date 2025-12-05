import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ML vs Automata IDS (78 Features)", layout="wide")
st.title("⚔️ ML Model vs Automata Intrusion Detection System — 78 Features")
st.markdown("---")

# -----------------------------
# LOAD DATASET
# -----------------------------
with st.spinner("Loading dataset..."):
    df = pd.read_csv("data/processed/preprocessed.csv")

st.success(f"✅ Dataset loaded successfully — {len(df):,} rows")

# -----------------------------
# FEATURE SELECTION
# -----------------------------
# Keep all numeric features (excluding labels and strings)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')

selected_features = numeric_cols
st.write(f"📊 Using {len(selected_features)} numeric features for ML model.")

# -----------------------------
# ENCODE LABEL
# -----------------------------
df['label'] = df['label'].apply(lambda x: 0 if x != 'BENIGN' else 1)
X = df[selected_features].values
y = df['label'].values

# -----------------------------
# STANDARDIZE
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# SMOTE BALANCING
# -----------------------------
smote_path = "data/processed/smote_data_78.csv"
if os.path.exists(smote_path):
    st.info("📂 SMOTE file found — loading existing balanced dataset...")
    smote_df = pd.read_csv(smote_path)
    X_res = smote_df[selected_features].values
    y_res = smote_df["label"].values
else:
    st.info("🧮 Applying SMOTE to balance dataset (first-time only)...")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    smote_df = pd.DataFrame(X_res, columns=selected_features)
    smote_df["label"] = y_res
    os.makedirs("data/processed", exist_ok=True)
    smote_df.to_csv(smote_path, index=False)
    st.success(f"✅ SMOTE complete and saved to {smote_path}")

st.write(f"Class distribution after SMOTE: {np.bincount(y_res)}")

# -----------------------------
# DEFINE INTRUSIONNET MODEL
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
# LOAD TRAINED 78-FEATURE MODEL
# -----------------------------
input_size = X_res.shape[1]
model_path = "models/pytorch_model_2class.pth"
model = IntrusionNet(input_dim=input_size, num_classes=2)

if os.path.exists(model_path):
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        st.success("✅ ML model loaded successfully (78-feature IntrusionNet).")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.warning("⚠ No pretrained model found — using untrained model for comparison.")

model.eval()

# -----------------------------
# AUTOMATA MODEL
# -----------------------------
class AutomataIDS:
    def __init__(self, threshold=2.8):
        self.threshold = threshold
        self.states = None

    def fit(self, X, y):
        benign = X[y == 1]
        self.states = np.mean(benign, axis=0)

    def predict(self, X):
        preds = []
        for x in X:
            dist = np.linalg.norm(x - self.states)
            preds.append(1 if dist < self.threshold else 0)
        return np.array(preds)

automata = AutomataIDS(threshold=2.8)
automata.fit(X_res, y_res)

# -----------------------------
# RUN PREDICTIONS
# -----------------------------
st.info("🚀 Running ML and Automata predictions...")

X_tensor = torch.tensor(X_res, dtype=torch.float32)
with torch.no_grad():
    try:
        y_pred_ml = torch.argmax(model(X_tensor), dim=1).numpy()
    except Exception as e:
        st.warning(f"⚠ ML model prediction failed: {e}")
        y_pred_ml = np.random.choice([0, 1], size=len(y_res))

y_pred_auto = automata.predict(X_res)

# -----------------------------
# EVALUATION
# -----------------------------
def get_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "Accuracy": report["accuracy"] * 100,
        "Precision": report["weighted avg"]["precision"] * 100,
        "Recall": report["weighted avg"]["recall"] * 100,
        "F1": report["weighted avg"]["f1-score"] * 100
    }

metrics_ml = get_metrics(y_res, y_pred_ml)
metrics_auto = get_metrics(y_res, y_pred_auto)

# -----------------------------
# RESULTS TABLE
# -----------------------------
st.header("📊 Model Comparison Summary (78 Features)")
comparison_df = pd.DataFrame([metrics_ml, metrics_auto], index=["ML Model", "Automata"])
st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

# -----------------------------
# CONFUSION MATRICES
# -----------------------------
st.subheader("🔹 Confusion Matrices")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(confusion_matrix(y_res, y_pred_ml), annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("ML Model")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_res, y_pred_auto), annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
axes[1].set_title("Automata")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

st.pyplot(fig)

# -----------------------------
# BAR CHART
# -----------------------------
st.subheader("📈 Metric Comparison (78 Features)")
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, list(metrics_ml.values()), width, label="ML Model")
ax.bar(x + width/2, list(metrics_auto.values()), width, label="Automata")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_title("ML vs Automata Performance (78 Features)")
st.pyplot(fig)

# -----------------------------
# END MESSAGE
# -----------------------------
st.success("✅ ML vs Automata Comparison Complete (78-Feature Model)")
st.caption("Developed with ❤️ Streamlit + PyTorch (78-Feature IntrusionNet)")
