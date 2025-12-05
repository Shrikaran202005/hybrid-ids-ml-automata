🚨 Hybrid Intrusion Detection System (IDS)
Deep Learning + Automata-Based Detection | CICIDS2017 | Streamlit Dashboard

📌 Project Overview

This project presents a Hybrid Intrusion Detection System (IDS) that integrates:

Deep Learning Models

Binary Classifier → Benign vs Attack

Multi-Class Classifier → DoS, DDoS, Brute Force, Botnet, Infiltration, Web Attack, etc.

Automata-Based IDS

Regex- and state-machine–driven rule-based detection

Full-Stack Visualization

Multi-page Streamlit dashboard

Networking Concepts

Flow-level inspection, IDS/IPS simulation

This system demonstrates a practical, end-to-end workflow combining
Machine Learning + Automata Theory + Networking + Full-Stack Development for intrusion detection.

🧠 Features

🔹 1. Dataset Exploration

Preview CICIDS2017 flow records

Feature descriptions & traffic distribution plots

🔹 2. Preprocessing

Cleaning, normalization, label encoding

Feature selection (correlations, variance, mutual-info)

🔹 3. Machine Learning Models

Deep Learning classifier (binary & multi-class)

Training graphs: accuracy/loss

Confusion matrix & classification metrics

🔹 4. Automata IDS (Baseline)

Regex/state-machine rule matching

Lightweight signature-based detection

Side-by-side comparison with ML model

🔹 5. Comparison Module

Accuracy, Precision, Recall, F1-score

ML vs Automata visual comparison

🔹 6. Live Prediction Demo

Input custom traffic values

Get ML + Automata results instantly

🔹 7. Network Simulation

Simulated packet feed

Real-time attack/benign classification

🔹 8. Streamlit Dashboard

Clean multipage UI

Interactive graphs & summaries

📂 Dataset: CICIDS2017

The CICIDS2017 dataset includes 5 days of realistic network traffic:

Normal traffic (HTTP, FTP, SSH, Email…)

Multiple attack types:

DoS, DDoS

Brute Force (SSH/FTP)

Web Attacks

Botnet

Infiltration

Heartbleed

Key stats:

80+ flow features

Labeled: benign vs multiple attacks

CSV flows used (from CICFlowMeter)

Dataset link:
(https://www.unb.ca/cic/datasets/ids-2017.html)

🏗 Project Architecture

📦 Hybrid-IDS

├── data/

│   ├── raw/

│   ├── processed/

├── models/

│   ├── binary_classifier.h5

│   ├── multiclass_classifier.h5

├── src/

│   ├── preprocessing.py

│   ├── model_train.py

│   ├── automata_ids.py

│   ├── utils.py

├── dashboard/

│   ├── Home.py

│   ├── Dataset_Explorer.py

│   ├── Preprocessing.py

│   ├── ML_Model.py

│   ├── Automata_IDS.py

│   ├── Comparison.py

│   ├── Network_Simulation.py

│   ├── Prediction_Demo.py

├── README.md

└── requirements.txt


🛠 Tech Stack
Languages

Python

Regex DSL (Automata rules)

Frameworks / Libraries

TensorFlow / Keras

Scikit-Learn

Pandas, NumPy

Matplotlib, Seaborn

Streamlit

CICFlowMeter (for CSV flow generation)

ML Models

Deep Neural Network

Softmax multi-class output

Sigmoid binary classifier

Automata/regex-based detection engine

🚀 Installation & Setup
1️⃣ Clone the Repo
git clone https://github.com/yourusername/hybrid-ids-cicids2017.git
cd hybrid-ids-cicids2017

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit Dashboard
streamlit run app.py

🎯 Usage Guide
🖥 Dashboard Pages

Dataset Explorer → preview samples

Preprocessing → normalization, feature selection

ML Model → train/evaluate DL classifier

Automata IDS → rule-based detection

Comparison → ML vs Automata

Prediction Demo → custom input detection

Network Simulation → real-time visualization

📊 Results Summary 

Model	  Accuracy	  Precision	  Recall	  F1-score

Binary Classifier	  99.2%	  99.1%	  99.3%	  99.2%

Multi-Class Classifier	  97.8%	  97.6%	  97.8%	  97.7%

Automata IDS	  78–85%	  Moderate	  High   FP	Low F1

🧩 Why Hybrid IDS?

ML-based IDS	Automata IDS
Detects unknown attacks	Only known signatures
Learns patterns	Transparent rules
High accuracy	Low cost, fast
Requires training	Easy to maintain

Together, they provide a balanced and explainable IDS.

📘 References

CICIDS2017 Dataset by Canadian Institute for Cybersecurity

Kim, A. et al. Intrusion Detection using Deep Learning

Automata theory applications in network security

⭐ Acknowledgment

Developed as a Core Project combining ML, Automata Theory, Networking, and Full-Stack Development.
