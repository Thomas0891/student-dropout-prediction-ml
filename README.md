# 🎓 Student Dropout Prediction System  
_A Machine Learning Web App for Early Dropout Detection_

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-99.23%25-brightgreen)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-1.0-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview
The **Student Dropout Prediction System** is a fully built end-to-end machine learning solution designed to identify students at risk of dropping out.  
The system uses academic, behavioural, demographic, and family-related attributes to predict dropout risk with **99.23% accuracy** and **1.0 ROC-AUC**.

The project includes:
- Machine Learning Model (Random Forest Tuned)
- EDA Dashboard
- SHAP Explainability
- Batch Predictions
- PDF Report Generation
- Modern Streamlit UI

---

## 📥 Dataset Source (Kaggle)

The dataset used in this project is from Kaggle:

🔗 **Student Dropout Analysis & Prediction Dataset**  
https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset

---

## 🧠 Objective
To build a reliable AI system that predicts whether a student is likely to **Drop Out** or **Continue**, enabling early intervention and academic support.

---

## 🧩 Project Architecture

        ┌────────────────┐
        │   Dataset      │
        └───────┬────────┘
                │
                ▼
    ┌────────────────────────┐
    │ Data Preprocessing     │
    │ - Cleaning             │
    │ - Encoding             │
    │ - Feature Engineering  │
    └─────────┬──────────────┘
              │
              ▼
    ┌────────────────────────┐
    │  Model Training        │
    │  (RF, SVM, XGB, etc.)  │
    └─────────┬──────────────┘
              │
              ▼
  ┌──────────────────────────────┐
  │ Tuned Random Forest Model    │
  │ - Accuracy: 99.23%           │
  │ - ROC AUC: 1.0               │
  └─────────┬────────────────────┘
            │
            ▼
 ┌─────────────────────────────┐
 │   Streamlit Web App         │
 │   - Single Prediction       │
 │   - Batch CSV Prediction    │
 │   - EDA Dashboard           │
 │   - SHAP Explainability     │
 │   - PDF Reports             │
 └─────────────────────────────┘

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 92% | 0.94 |
| Decision Tree | 93% | 0.95 |
| SVM | 96% | 0.97 |
| KNN | 94% | 0.95 |
| XGBoost | 98% | 0.99 |
| **Random Forest (Final Model)** | ⭐ **99.23%** | ⭐ **1.0** |

---

## 🔍 Feature Engineering

Created new features:
- `Grade_Avg`
- `High_Absence`
- `Total_Alcohol`

Applied:
- Scaling  
- One-Hot Encoding  
- Label Encoding  

---

## 🎛 Streamlit App Features

### **1. Single Student Prediction**
Predict dropout risk instantly.

### **2. Batch CSV Prediction**
Upload multiple students at once.

### **3. EDA Dashboard**
Explore data with:
- Histograms
- Heatmaps
- Correlation maps
- Distribution plots

### **4. SHAP Explainability**
Understand **why** the prediction was made.



---

## 🖥 How to Run Locally

```bash
git clone <your-repo-link>
cd student-dropout-prediction
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
shap
fpdf
