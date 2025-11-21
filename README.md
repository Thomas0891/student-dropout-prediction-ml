##🎓 Student Dropout Prediction System
An End-to-End Machine Learning + Streamlit Dashboard Project












##⭐ Overview

This project builds a complete machine learning system to predict student dropout risk using academic, behavioral, and demographic data.

The final Random Forest model achieves:

🎯 99.23% Accuracy

📈 1.0 ROC-AUC Score

A beautiful, interactive Streamlit web application is included with:

🔮 Real-time Predictions

📦 Batch Prediction (CSV Upload)

📊 EDA Dashboard

🧠 Explainability (SHAP & LIME)

📝 Professional PDF Report Generation

##📂 Dataset

Dataset used in the project:
🔗 https://www.kaggle.com/datasets/abdullah0a/student-dropout-analysis-and-prediction-dataset

The dataset contains information such as:

Demographic details

Academic performance

Family background

Alcohol consumption

Attendance

Personal habits

All features were analyzed and refined to improve model performance.

##🧠 Machine Learning Pipeline
✔ Step 1: Data Collection

Data loaded from Kaggle dataset

Validated column types & formatting

Handled missing values

✔ Step 2: Data Preprocessing

Includes:

Encoding categorical variables

Scaling numeric variables

Handling missing entries

Removing inconsistencies

Outlier treatment

✔ Step 3: Exploratory Data Analysis (EDA)

Visualizations include:

Count plots

Histograms

Boxplots

Line & bar charts

Correlation heatmap

Clustered relationships

Insights:

High absences strongly correlate with dropout

Low grades predict dropout risk

Alcohol consumption affects grades

Family status influences performance

✔ Step 4: Feature Engineering
Engineered Feature	Description
Grade_Avg	Average of Grade_1 and Grade_2
High_Absence	Flag for absences > 5
Total_Alcohol	Weekend + weekday alcohol

These features significantly improved the model.

✔ Step 5: Model Training

The following ML models were trained and compared:

Logistic Regression

KNN

SVM

Decision Tree

Random Forest

XGBoost (optional)

Best Model:
🔥 Random Forest Classifier

✔ Step 6: Hyperparameter Tuning

Best parameters (via GridSearchCV):

{
  "max_depth": 10,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "n_estimators": 200
}


##Final Performance:

🎯 Accuracy: 0.9923

📈 ROC-AUC: 1.0

🎨 Streamlit Web Application

The app includes:

🧍 Single Student Prediction

User inputs data → model predicts dropout risk + explanation.

📦 Batch Prediction

Upload CSV → predicts risk for all students.
Output can be downloaded.

📊 EDA Dashboard

Grade trends

Absence distribution

Alcohol consumption charts

Correlation heatmap

Target distribution

🧠 Explainability

SHAP Summary Plot (global)

LIME Explanation (local)

📝 PDF Report (A4, Professional)

Generated report includes:

Student details

Prediction & probability

6 charts

Recommendations

SHAP/LIME explanation

Professional formatting

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
