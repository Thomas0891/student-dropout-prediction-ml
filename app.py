import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================
# 1. PAGE CONFIG & STYLING
# ==========================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        color: #e5e7eb;
    }
    .stApp {
        background-color: transparent;
    }
    .big-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e5e7eb;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #9ca3af;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.8rem;
        background: #020617;
        border: 1px solid #1f2937;
    }
    .footer {
        font-size: 0.8rem;
        color: #6b7280;
        padding-top: 1rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 2. LOAD MODEL & PREPROCESSOR
# ==========================

@st.cache_resource
def load_artifacts():
    try:
        preprocessor = joblib.load("preprocessor.joblib")
        model = joblib.load("final_model.joblib")
        return preprocessor, model
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

preprocessor, model = load_artifacts()

# For mapping prediction 0/1 back to labels
LABEL_MAP = {0: "No Dropout", 1: "Dropped Out"}  # adjust if your encoding is different

# These features must match what you used in training
SELECTED_FEATURES = [
    # Numeric
    "Age", "Mother_Education", "Father_Education",
    "Number_of_Absences", "Grade_1", "Grade_2", "Final_Grade",
    "Grade_Avg", "High_Absence", "Total_Alcohol",
    # Categorical
    "School", "Gender", "Address", "Family_Size", "Parental_Status",
    "Mother_Job", "Father_Job", "Reason_for_Choosing_School"
]

# ==========================
# 3. SIDEBAR
# ==========================

st.sidebar.title("⚙️ App Controls")
mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Student Prediction", "Batch Prediction (CSV)", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.write("📂 **Model files expected:**")
st.sidebar.code("preprocessor.joblib\nfinal_model.joblib", language="bash")
st.sidebar.markdown("---")
st.sidebar.write("👨‍🎓 **Project:** Student Dropout Prediction")
st.sidebar.write("🧠 ML Models used in training: Logistic, DT, RF, SVM, KNN, XGBoost")

# ==========================
# 4. HEADER
# ==========================

st.markdown('<div class="big-title">🎓 Student Dropout Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An intelligent machine learning web app to predict whether a student is likely to drop out, based on academic and demographic features.</div>',
    unsafe_allow_html=True
)
st.markdown("")

# Small info banner
if preprocessor is None or model is None:
    st.warning("⚠️ Model not loaded. Please ensure `preprocessor.joblib` and `final_model.joblib` are in the same folder as `app.py`.")
else:
    st.success("✅ Model and preprocessor loaded successfully.")


# ==========================
# 5. HELPER: MAKE INPUT DF
# ==========================

def build_single_input_df(
    school, gender, age, address, family_size, parental_status,
    mother_edu, father_edu, mother_job, father_job,
    reason_school,
    absences, g1, g2, final_grade,
    weekend_alcohol, weekday_alcohol
):
    """
    Build a single-row DataFrame with the same columns used in training.
    """
    # Derived features
    grade_avg = (g1 + g2) / 2.0
    high_absence = int(absences > 0)  # or use median threshold in training if you prefer
    total_alcohol = weekend_alcohol + weekday_alcohol

    data = {
        # Numeric
        "Age": [age],
        "Mother_Education": [mother_edu],
        "Father_Education": [father_edu],
        "Number_of_Absences": [absences],
        "Grade_1": [g1],
        "Grade_2": [g2],
        "Final_Grade": [final_grade],
        "Grade_Avg": [grade_avg],
        "High_Absence": [high_absence],
        "Total_Alcohol": [total_alcohol],
        # Categorical
        "School": [school],
        "Gender": [gender],
        "Address": [address],
        "Family_Size": [family_size],
        "Parental_Status": [parental_status],
        "Mother_Job": [mother_job],
        "Father_Job": [father_job],
        "Reason_for_Choosing_School": [reason_school]
    }

    df_input = pd.DataFrame(data, columns=SELECTED_FEATURES)
    return df_input


def predict_from_df(df_raw):
    """
    Predict using raw dataframe (columns = SELECTED_FEATURES)
    """
    X_pre = preprocessor.transform(df_raw)
    proba = model.predict_proba(X_pre)[:, 1]
    pred = model.predict(X_pre)
    return pred, proba


# ==========================
# 6. MODE: SINGLE STUDENT
# ==========================

if mode == "Single Student Prediction":

    st.markdown("### 👤 Single Student Prediction")

    col1, col2, col3 = st.columns(3)

    # Basic info
    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", min_value=10, max_value=30, value=16)

    with col2:
        address = st.selectbox("Address", ["U", "R"], help="U = Urban, R = Rural")
        family_size = st.selectbox("Family Size", ["LE3", "GT3"], help="LE3 = ≤3, GT3 = >3")
        parental_status = st.selectbox("Parental Status", ["T", "A"], help="T = Together, A = Apart")

    with col3:
        mother_edu = st.slider("Mother Education (0–4)", min_value=0, max_value=4, value=2)
        father_edu = st.slider("Father Education (0–4)", min_value=0, max_value=4, value=2)

    st.markdown("---")
    st.markdown("#### 👨‍👩‍👧 Family & Background")

    col4, col5 = st.columns(2)
    with col4:
        mother_job = st.selectbox("Mother Job", ["at_home","teacher","health","services","other"])
        father_job = st.selectbox("Father Job", ["teacher","health","services","other","at_home"])
    with col5:
        reason_school = st.selectbox("Reason for Choosing School", ["course","home","reputation","other"])

    st.markdown("---")
    st.markdown("#### 📚 Academic Performance & Behaviour")

    col6, col7, col8 = st.columns(3)
    with col6:
        absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=2)
    with col7:
        g1 = st.number_input("Grade 1", min_value=0, max_value=20, value=10)
        g2 = st.number_input("Grade 2", min_value=0, max_value=20, value=12)
    with col8:
        final_grade = st.number_input("Final Grade", min_value=0, max_value=20, value=12)
        weekend_alcohol = st.slider("Weekend Alcohol (1–5)", min_value=1, max_value=5, value=1)
        weekday_alcohol = st.slider("Weekday Alcohol (1–5)", min_value=1, max_value=5, value=1)

    st.markdown("---")

    if st.button("🔮 Predict Dropout Risk", type="primary"):

        if preprocessor is None or model is None:
            st.error("Model not loaded. Cannot predict.")
        else:
            df_input = build_single_input_df(
                school, gender, age, address, family_size, parental_status,
                mother_edu, father_edu, mother_job, father_job,
                reason_school,
                absences, g1, g2, final_grade,
                weekend_alcohol, weekday_alcohol
            )

            pred, proba = predict_from_df(df_input)
            label = LABEL_MAP.get(int(pred[0]), str(pred[0]))
            prob_dropout = proba[0]

            colA, colB = st.columns(2)
            with colA:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Predicted Status", label)
                st.markdown("</div>", unsafe_allow_html=True)
            with colB:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Dropout Probability", f"{prob_dropout*100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            st.progress(min(max(prob_dropout, 0.0), 1.0))
            st.markdown("**Interpretation:**")
            if prob_dropout > 0.7:
                st.write("🔴 High risk of dropout – needs strong academic & personal support.")
            elif prob_dropout > 0.4:
                st.write("🟠 Medium risk of dropout – monitor regularly and support.")
            else:
                st.write("🟢 Low risk of dropout – currently stable, keep supporting performance.")


# ==========================
# 7. MODE: BATCH PREDICTION
# ==========================

elif mode == "Batch Prediction (CSV)":

    st.markdown("### 📦 Batch Prediction from CSV")
    st.write("Upload a CSV file with the **same feature columns** used during training:")

    st.code("\n".join(SELECTED_FEATURES), language="text")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_batch.head())

        # Ensure columns exist
        missing = [c for c in SELECTED_FEATURES if c not in df_batch.columns]
        if missing:
            st.error(f"The following required columns are missing in your file: {missing}")
        else:
            if st.button("🚀 Run Batch Prediction"):
                if preprocessor is None or model is None:
                    st.error("Model not loaded. Cannot predict.")
                else:
                    df_input = df_batch[SELECTED_FEATURES].copy()
                    pred, proba = predict_from_df(df_input)

                    labels = [LABEL_MAP.get(int(p), str(p)) for p in pred]
                    df_batch["Predicted_Label"] = labels
                    df_batch["Dropout_Probability"] = proba

                    st.success("Batch prediction completed!")
                    st.dataframe(df_batch.head())

                    # Download result
                    csv_out = df_batch.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_out,
                        file_name="student_dropout_predictions.csv",
                        mime="text/csv"
                    )


# ==========================
# 8. MODE: ABOUT PROJECT
# ==========================

elif mode == "About Project":

    st.markdown("### ℹ️ About This Project")

    st.write("""
    This Student Dropout Prediction system is built using **Supervised Machine Learning**
    on a real-world educational dataset.

    **Pipeline includes:**
    - Data Collection & Loading from CSV
    - Data Preprocessing & Cleaning (missing values, encoding)
    - Exploratory Data Analysis (EDA) with plots
    - Feature Engineering (Grade_Avg, High_Absence, Total_Alcohol, etc.)
    - Feature Selection (manual + model-based)
    - Model Training:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - SVM (RBF)
        - KNN
        - XGBoost (optional)
    - Model Evaluation (Accuracy, ROC AUC, Confusion Matrix)
    - Hyperparameter Tuning (Random Forest GridSearch)
    - Final Deployment using **Streamlit**
    """)

    st.markdown("#### 🏗 Technical Stack")
    st.write("""
    - **Language:** Python  
    - **Libraries:** pandas, numpy, scikit-learn, (xgboost), streamlit  
    - **ML Type:** Binary classification (Dropout vs No Dropout)  
    - **Deployment:** Streamlit app (can be hosted on Streamlit Cloud / Render)
    """)

    st.markdown("#### 🎯 How to Explain in Viva")
    st.write("""
    1. Explain the problem: predicting student dropout early.
    2. Explain the dataset: features like grades, absences, family, etc.
    3. Explain preprocessing: handling missing values, encoding categories, scaling numeric features.
    4. Explain models: you tried multiple ML models and selected the best based on Accuracy & ROC AUC.
    5. Explain evaluation: show confusion matrix, ROC curve, and discuss results.
    6. Explain deployment: the trained model is exported and loaded in this Streamlit app for real-time predictions.
    """)

    st.markdown('<div class="footer">Made for Major Project & Research – Student Dropout Prediction 🎓</div>', unsafe_allow_html=True)
