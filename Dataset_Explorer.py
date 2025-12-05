# pages/1_Dataset_Explorer.py
from utils import load_dataset, load_model, automata_rule, compute_metrics, plot_confusion_matrix

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Dataset Explorer", layout="wide")

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title("📊 Dataset Explorer - CICIDS2017")
st.write("""
This section lets you **upload and explore the CICIDS2017 dataset**.
You can:
- Upload a sample CSV file.
- View dataset preview & statistics.
- Visualize class distributions (Normal vs Attacks).
""")

# -------------------------------
# FILE UPLOAD SECTION
# -------------------------------
uploaded_file = st.file_uploader("Upload a CICIDS2017 CSV file", type=["csv"])

if uploaded_file:
    # Load uploaded dataset
    df = pd.read_csv(uploaded_file)

    # Show shape
    st.subheader("📐 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Show first 5 rows
    st.subheader("🔎 Preview of Dataset")
    st.dataframe(df.head())

    # Show column types
    st.subheader("🧾 Column Information")
    st.write(df.dtypes)

    # Show missing values
    st.subheader("⚠️ Missing Values")
    st.write(df.isnull().sum())

    # -------------------------------
    # TARGET DISTRIBUTION
    # -------------------------------
    if "Label" in df.columns:
        st.subheader("📌 Target Variable Distribution")

        label_counts = df["Label"].value_counts()

        # --- Count BENIGN vs ATTACK ---
        benign_count = label_counts.get("BENIGN", 0)
        attack_count = df.shape[0] - benign_count
        total = df.shape[0]

        benign_pct = (benign_count / total) * 100
        attack_pct = (attack_count / total) * 100

        st.markdown(f"""
        **🟢 BENIGN Samples:** {benign_count:,} ({benign_pct:.2f}%)
        <br>**🔴 ATTACK Samples:** {attack_count:,} ({attack_pct:.2f}%)
        """, unsafe_allow_html=True)

        # Bar chart
        st.bar_chart(label_counts)

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    # -------------------------------
    # CORRELATION HEATMAP
    # -------------------------------
    st.subheader("📈 Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric features available for correlation heatmap.")

else:
    st.info("Please upload a CSV file to explore the dataset.")
