import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from imblearn.over_sampling import SMOTE

st.title("🔎 CICIDS2017 Preprocessing")

# --------------------------------------------------------------
# FILE SELECTION FROM LOCAL DIRECTORY
# --------------------------------------------------------------
data_dir = "data/raw"
available_files = [f for f in os.listdir(data_dir) if f.endswith((".csv", ".xls", ".xlsx", ".parquet"))]

if not available_files:
    st.error("❌ No dataset files found in 'data/raw/'. Please place your CICIDS2017 files there.")
    st.stop()

selected_file = st.selectbox("📂 Select a CICIDS2017 dataset file:", available_files)
file_path = os.path.join(data_dir, selected_file)
st.write(f"✅ Selected file: `{selected_file}`")

# Load file
if selected_file.endswith(".csv"):
    df = pd.read_csv(file_path, low_memory=False)
elif selected_file.endswith((".xls", ".xlsx")):
    df = pd.read_excel(file_path, engine="openpyxl")
elif selected_file.endswith(".parquet"):
    df = pd.read_parquet(file_path)
else:
    st.error("❌ Unsupported file type")
    st.stop()

# --------------------------------------------------------------
# SHOW DATA
# --------------------------------------------------------------
st.subheader("📊 Preview of Raw Data")
st.write(df.head())
st.write(f"📏 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# --------------------------------------------------------------
# PREPROCESSING STEPS
# --------------------------------------------------------------
st.subheader("🧹 Preprocessing Steps")

def downcast_numeric(df_):
    for col in df_.select_dtypes(include=["int64", "float64"]).columns:
        if pd.api.types.is_integer_dtype(df_[col].dropna()):
            df_[col] = pd.to_numeric(df_[col], downcast="integer")
        else:
            df_[col] = pd.to_numeric(df_[col], downcast="float")
    return df_

df.columns = df.columns.str.strip().str.lower()
orig_rows = df.shape[0]

# 1️⃣ Drop duplicates
before_dups = df.shape[0]
df = df.drop_duplicates()
st.write(f"✔️ Removed duplicates → {before_dups - df.shape[0]} rows removed; now {df.shape[0]} rows remain")

# 2️⃣ Replace infinities, handle missing values
df = df.replace([np.inf, -np.inf], np.nan)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

if numeric_cols:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    lower, upper = df[numeric_cols].quantile(0.01), df[numeric_cols].quantile(0.99)
    df[numeric_cols] = df[numeric_cols].clip(lower=lower, upper=upper, axis=1)

for col in non_numeric_cols:
    df[col] = df[col].fillna("unknown")

st.write(f"✔️ Missing/infinite values handled (median for numeric, 'unknown' for non-numeric).")

# 3️⃣ Standardize label column
possible_labels = ["label", "labels", "class", "attack_cat", "attack_category"]
label_column = next((c for c in df.columns if c.lower() in possible_labels), None)
if label_column is None:
    st.error(f"❌ No label column found. Available columns: {df.columns.tolist()}")
    st.stop()

if label_column != "label":
    df.rename(columns={label_column: "label"}, inplace=True)

df["label"] = df["label"].astype(str).str.strip()
df["label_norm"] = df["label"].str.upper().str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)

attack_map = {
    "BENIGN": "BENIGN",
    "DDOS": "DDoS",
    "DOS HULK": "DoS Hulk",
    "DOS SLOWHTTPTEST": "DoS Slowhttptest",
    "DOS GOLDENEYE": "DoS GoldenEye",
    "DOS SLOWLORIS": "DoS slowloris",
    "FTP-PATATOR": "FTP-Patator",
    "SSH-PATATOR": "SSH-Patator",
    "BOT": "Bot",
    "INFILTRATION": "Infiltration",
    "PORTSCAN": "PortScan",
    "WEB ATTACK - BRUTE FORCE": "Web Brute Force",
    "WEB ATTACK - XSS": "Web XSS",
    "WEB ATTACK - SQL INJECTION": "Web SQL Injection",
    "HEARTBLEED": "Heartbleed"
}
df["label"] = df["label_norm"].apply(lambda x: attack_map.get(x, x.title() if isinstance(x, str) else x))
df.drop(columns=["label_norm"], inplace=True)

st.subheader("📌 Label Distribution After Cleaning")
st.write(df["label"].value_counts())

df = downcast_numeric(df)
st.write("✔️ Numeric columns downcasted to smaller dtypes.")

# --------------------------------------------------------------
# SAVE CLEANED DATA (APPEND MODE) WITH DUPLICATE FILE CHECK
# --------------------------------------------------------------
processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)
processed_path = os.path.join(processed_dir, "preprocessed.csv")
processed_files_path = os.path.join(processed_dir, "processed_files.txt")

# Load already processed files
if os.path.exists(processed_files_path):
    with open(processed_files_path, "r") as f:
        processed_files = f.read().splitlines()
else:
    processed_files = []

# Check if this file was already processed
if selected_file in processed_files:
    st.warning(f"⚠️ File `{selected_file}` was already processed. Skipping append to avoid duplicates.")
    if os.path.exists(processed_path):
        combined_df = pd.read_csv(processed_path, low_memory=False)
    else:
        combined_df = df
else:
    # Combine with existing preprocessed dataset
    if os.path.exists(processed_path):
        existing_df = pd.read_csv(processed_path, low_memory=False)
        combined_df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
        st.write(f"✔️ Combined with existing preprocessed file.")
    else:
        combined_df = df

    # Save combined preprocessed dataset
    combined_df.to_csv(processed_path, index=False)
    st.success(f"✅ Preprocessed dataset updated and saved to `{processed_path}`")
    st.write(f"📏 Total rows in combined dataset: {combined_df.shape[0]}")

    # Update processed files list
    processed_files.append(selected_file)
    with open(processed_files_path, "w") as f:
        f.write("\n".join(processed_files))

# --------------------------------------------------------------
# 🔸 OPTIONAL BALANCING & PCA
# --------------------------------------------------------------
st.subheader("⚖️ Optional Post-Processing")

apply_smote = st.checkbox("Apply SMOTE (Balance Classes)", value=False)
apply_pca = st.checkbox("Apply PCA Dimensionality Reduction", value=False)

if apply_smote or apply_pca:
    with st.spinner("Processing..."):
        X = combined_df.drop(columns=["label"])
        y = combined_df["label"]

        # Use only numeric columns
        X = X.select_dtypes(include=[np.number])

        # 🧠 SMOTE Balancing
        if apply_smote:
            st.write("🔁 Applying SMOTE balancing (this may take a while)...")
            smote = SMOTE(random_state=42, sampling_strategy="auto", n_jobs=-1)
            X, y = smote.fit_resample(X, y)
            st.success("✅ SMOTE applied successfully — dataset balanced!")

        # 📉 PCA Reduction
        if apply_pca:
            st.write("📉 Performing PCA for dimensionality reduction...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = IncrementalPCA(n_components=0.95)
            X_pca = pca.fit_transform(X_scaled)
            X = pd.DataFrame(X_pca)
            st.success(f"✅ PCA completed — reduced to {X.shape[1]} components retaining 95% variance")

        # Merge back labels
        processed_final = pd.concat([X, pd.Series(y, name="label")], axis=1)

        # Save outputs
        out_path = os.path.join(processed_dir, "balanced_pca.csv" if (apply_smote and apply_pca)
                                else "balanced.csv" if apply_smote
                                else "pca.csv")

        processed_final.to_csv(out_path, index=False)
        st.success(f"💾 Final processed file saved to `{out_path}`")
        st.write(f"📏 Final dataset shape: {processed_final.shape}")
