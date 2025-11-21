
🎓 Student Dropout Prediction System

An End-to-End Machine Learning + Streamlit Dashboard Project












📌 Project Overview

This project builds a complete machine learning system for predicting student dropout risk using academic, behavioral, and demographic factors.

The trained Random Forest model achieves:

🎯 99.23% Accuracy

📈 1.0 ROC-AUC Score

A fully interactive Streamlit web application is included for:

Real-time predictions

Batch prediction from CSV

Data visualization dashboard

Explainability (SHAP & LIME)

Professional PDF report generation

📂 Dataset

Dataset used in the project:
🔗 https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset

The dataset contains student information such as:

Demographic details

Academic records

Family background

Alcohol consumption

Absences

Personal habits

🎯 Problem Statement

Student dropout is a significant challenge in global education systems.
Early prediction helps institutions:

Provide personalized support

Improve academic performance

Reduce dropout rates

Support at-risk students

This project answers:

“Can we accurately predict whether a student will drop out based on their profile and behavior?”

🧠 Machine Learning Pipeline
✔ Step 1 — Data Collection

Dataset downloaded from Kaggle

CSV loaded using Pandas

✔ Step 2 — Data Preprocessing

Includes:

Handling missing values

Label Encoding & One-Hot Encoding

Scaling numeric features

Outlier filtering

Cleaning inconsistent values

✔ Step 3 — Exploratory Data Analysis (EDA)

Visualizations include:

📊 Count plots

📈 Line & bar charts

🧊 Boxplots

🔥 Correlation heatmaps

🎯 Feature importance charts

Key Insights:

More absences → Higher dropout probability

Low grades strongly correlate with dropout

Family factors & alcohol consumption influence performance

✔ Step 4 — Feature Engineering

New meaningful features created:

Feature	Description
Grade_Avg	Average of Grade_1 & Grade_2
High_Absence	1 if absences > 5 else 0
Total_Alcohol	Weekend + weekday alcohol consumption

These features significantly improved model performance.

✔ Step 5 — Model Training

Trained & compared multiple ML models:

Model	Accuracy
Logistic Regression	87.5%
KNN	90.2%
Decision Tree	91.4%
SVM	94.6%
Random Forest	99.23% ✔
XGBoost (Optional)	98.5%
✔ Step 6 — Hyperparameter Tuning (GridSearchCV)

Best Random Forest parameters:

{
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "n_estimators": 200
}


Performance after tuning:

🎯 Accuracy: 0.9923

📈 ROC-AUC: 1.0

🚀 Deployment
✔ Streamlit Web Application

The app includes:

📘 Single student prediction

📦 Batch prediction (CSV upload)

📊 EDA Dashboard

🧠 Model explainability (SHAP + LIME)

📝 Professional PDF Report

🎨 Beautiful UI with gradients & glassmorphism

🖥️ App Features
⭐ Real-Time Prediction

User enters student details → model predicts:

Dropout status

Probability

Risk interpretation

⭐ Batch Prediction

Upload a CSV → receive predictions for all students.
Download output as CSV.

⭐ EDA Dashboard

Includes:

Grade trends

Absence charts

Alcohol consumption

Heatmaps

Feature importance

⭐ Model Explainability

Global Explainability (SHAP): Feature impact

Local Explainability (LIME): Why a specific student is at risk

⭐ Professional PDF Report (A4)

Contains:

Student details

Prediction result

Risk probability

Interpretations

Recommendations

6 charts

SHAP summary

LIME explanation

Footer & page numbers

Perfect for:

✔ Viva
✔ Project submission
✔ Research paper
✔ Internship portfolio

📁 Folder Structure
Student_Dropout_Prediction/
│
├── app.py
├── preprocessor.joblib
├── final_model.joblib
├── student_dropout.csv
├── batch_template.csv
├── sample_batch_students.csv
├── requirements.txt
└── README.md

⚙️ Installation & Usage
1️⃣ Clone this repository:
git clone https://github.com/Thomas0891/student-dropout-prediction-ml.git

2️⃣ Install dependencies:
pip install -r requirements.txt

3️⃣ Run the Streamlit App:
streamlit run app.py

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-Learn

Matplotlib & Seaborn

Plotly

SHAP & LIME

Streamlit

ReportLab

📈 Future Enhancements

Deep learning model integration

Automated alerts for high-risk students

Dashboard mobile version

Database integration (MySQL / Firebase)

Auto email student report

📝 License

This project is under the MIT License — free to use, modify, and publish.

🙌 Credits

Developed by Thomas Joseph
Guided by AI Agents + Machine Learning Research Practices

If you like this project, ⭐ star the repo!
