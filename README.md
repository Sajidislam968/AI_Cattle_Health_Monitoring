# 🐄 AI-Powered Multi-Modal Cattle Health Monitoring System

Early Detection of Cattle Diseases using Computer Vision, Synthetic IoT Collar Sensors, Explainable AI (XAI), and Decision Fusion.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLO11](https://img.shields.io/badge/YOLO-v11-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Overview

This project presents a complete **AI-powered multi-modal cattle health monitoring system** capable of detecting potential health conditions by combining:

- 🐄 Computer Vision
- 📡 Synthetic Smart Collar Sensor Data
- 🤖 Machine Learning (XGBoost)
- 🔍 Explainable AI (SHAP)
- 🧠 Decision Fusion
- 📊 Interactive Dashboard

Unlike traditional approaches that rely on only image analysis or only sensor data, this system integrates both modalities to provide more reliable health assessment and explainable predictions.


# Key Features

- YOLO11-based cattle behavior recognition
- Multi-object tracking using ByteTrack
- Five behavior detection classes
  - Standing
  - Walking
  - Eating
  - Lying
  - Rumination
- Synthetic ESP32 Collar Sensor Simulation
- Rule-Based Risk Score Engine
- XGBoost Disease Prediction
- SHAP Explainable AI
- Recommendation Engine
- Decision Fusion Engine
- Streamlit Dashboard
- FastAPI Backend


Detected Behaviors

- Standing
- Walking
- Eating
- Lying
- Rumination

---

# Synthetic Smart Collar Sensors

This project simulates IoT smart collar data using literature-based physiological ranges.

Generated Features

- Heart Rate
- Body Temperature
- Eating Duration
- Rumination Duration
- Walking Duration
- Lying Duration
- Ambient Temperature
- Humidity
- THI
- Activity Index

---

# Machine Learning Pipeline

```
Sensor Features
        │
              ▼
Risk Score Engine
        │
              ▼
XGBoost Classifier
        │
              ▼
    Healthy
    At Risk
    Diseased
```

---

# Explainable AI

The prediction model is interpreted using SHAP.

Generated outputs include

- Feature Importance
- SHAP Summary Plot
- Individual Prediction Explanation

---

# Decision Fusion

The Decision Fusion Engine combines

- Vision Behavior
- Sensor Information
- Risk Score
- XGBoost Prediction

to produce the final cattle health status.

Final Classes

- Healthy
- At Risk
- Diseased

---

# Recommendation Engine

The system automatically generates recommendations such as

- Continue Monitoring
- Increase Observation
- Veterinary Inspection Recommended
- Immediate Medical Attention Required

---

# Dashboard

The Streamlit dashboard includes
<img width="1848" height="1007" alt="Screenshot From 2026-08-21 14-23-08" src="https://github.com/user-attachments/assets/0d81ce95-1efd-4e14-86af-01f6d3178672" />


- Live Inference
  <img width="1848" height="1007" alt="Screenshot From 2026-08-21 14-10-37" src="https://github.com/user-attachments/assets/94fbcb72-e7f4-477a-9eaa-4161c2844943" />

- Cow Health Analytics
  <img width="1770" height="816" alt="Screenshot From 2026-08-21 14-35-29" src="https://github.com/user-attachments/assets/9329a781-385e-4c71-9bb3-9bf2ea74d9f1" />

- Explainable AI
  <img width="1474" height="936" alt="Screenshot From 2026-08-21 15-01-08" src="https://github.com/user-attachments/assets/5f122afd-b5de-4933-9a63-79d3a72518c1" />

- System Information
- Sensor Monitoring
- Final Health Prediction

---


# Installation

Clone the repository

```bash
git clone https://github.com/sajid968/AI_Cattle_Health_Monitoring.git

cd AI_Cattle_Health_Monitoring
```

Create virtual environment

```bash
python -m venv myenv

source myenv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

# ▶ Run Vision Pipeline

```bash
python realtime_inference.py
```

---

# Model Comparison

The project evaluates multiple machine learning models.

- Logistic Regression
- Decision Tree
- KNN
- Random Forest
- **XGBoost (Selected Model)**

XGBoost was selected because it provides:

- Better generalization
- Efficient gradient boosting
- Excellent compatibility with SHAP Explainable AI
- Smooth integration with the proposed multimodal framework

---

# Research Contribution

✔ Multi-modal health monitoring

✔ Synthetic smart collar simulation

✔ YOLO11 behavior recognition

✔ Explainable AI using SHAP

✔ Rule-based decision fusion

✔ Automatic recommendation engine

✔ Interactive dashboard

---

# Future Work

- Real ESP32 smart collar deployment
- MQTT communication
- Thermal camera integration
- LSTM-based temporal behavior analysis
- Transformer-based multimodal fusion
- Mobile application

---

# Citation


AI-Powered Multi-Modal Cattle Health Monitoring System
Computer Vision + Synthetic IoT + Explainable AI + Decision Fusion
```


# License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Sajid Islam**

Department of Computer Science & Engineering

Research Interests

- Artificial Intelligence
- Computer Vision
- IoT
- Explainable AI
- Precision Livestock Farming
