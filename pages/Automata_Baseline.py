import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Automata Baseline IDS", layout="wide")
st.title("🤖 Automata Baseline Intrusion Detection System")
st.markdown("---")

# -----------------------------
# LOAD DATASET
# -----------------------------
with st.spinner("Loading dataset..."):
    df = pd.read_csv("data/processed/preprocessed.csv")

st.success(f"✅ Dataset loaded successfully — {len(df):,} rows")

# -----------------------------
# SELECT FEATURES
# -----------------------------
selected_features = [
    'destination port', 'flow duration', 'total fwd packets', 'total backward packets',
    'total length of fwd packets', 'total length of bwd packets',
    'fwd packet length max', 'fwd packet length mean', 'bwd packet length mean',
    'flow bytes/s', 'flow packets/s', 'flow iat mean', 'flow iat std',
    'fwd iat mean', 'bwd iat mean', 'bwd iat std', 'min packet length',
    'max packet length', 'packet length mean', 'packet length std',
    'fin flag count', 'syn flag count', 'rst flag count', 'ack flag count',
    'average packet size', 'active mean', 'idle mean'
]
df = df[selected_features + ['label']]
df['label'] = df['label'].apply(lambda x: 0 if 'Attack' in x or x != 'BENIGN' else 1)

X = df[selected_features].values
y = df['label'].values

# -----------------------------
# NORMALIZE & BALANCE DATA
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

benign = X_scaled[y == 1]
attack = X_scaled[y == 0]

attack_down = resample(attack,
                       replace=False,
                       n_samples=min(50000, len(attack)),
                       random_state=42)
combined = np.vstack((attack_down, benign))
combined_y = np.array([0]*len(attack_down) + [1]*len(benign))

sm = SMOTE(random_state=42, k_neighbors=3)
X_res, y_res = sm.fit_resample(combined, combined_y)

# -----------------------------
# AUTOMATA CLASS
# -----------------------------
class AutomataIDS:
    def __init__(self, threshold=2.8):
        self.threshold = threshold
        self.states = []

    def fit(self, X, y):
        benign = X[y == 1]
        self.states = np.mean(benign, axis=0)

    def predict(self, X):
        preds = []
        for x in X:
            dist = np.linalg.norm(x - self.states)
            preds.append(1 if dist < self.threshold else 0)
        return np.array(preds)

# -----------------------------
# TRAIN & EVALUATE
# -----------------------------
automata = AutomataIDS(threshold=2.8)
automata.fit(X_res, y_res)
y_pred = automata.predict(X_res)
cm = confusion_matrix(y_res, y_pred)
report = classification_report(y_res, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
st.header("📊 Classification Report")
st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))

accuracy = report["accuracy"] * 100
f1_attack = report["0"]["f1-score"] * 100
f1_benign = report["1"]["f1-score"] * 100

st.markdown("### 🎯 Key Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="background-color:#1E1E1E;padding:20px;border-radius:10px;text-align:center;">
    <h3 style='color:#00FFAA;'>Accuracy</h3>
    <h2 style='color:white;'>{accuracy:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background-color:#1E1E1E;padding:20px;border-radius:10px;text-align:center;">
    <h3 style='color:#00C8FF;'>F1 (Attack)</h3>
    <h2 style='color:white;'>{f1_attack:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="background-color:#1E1E1E;padding:20px;border-radius:10px;text-align:center;">
    <h3 style='color:#FFB800;'>F1 (Benign)</h3>
    <h2 style='color:white;'>{f1_benign:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# CONFUSION MATRIX HEATMAP
# -----------------------------
st.markdown("### 🔥 Confusion Matrix Heatmap")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix", fontsize=14)
st.pyplot(fig)

st.success("✅ Automata Baseline Evaluation Complete")
