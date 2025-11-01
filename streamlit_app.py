#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load artifacts ---
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("final_score_model.pkl")
    return preprocessor, model

preprocessor, model = load_artifacts()

st.set_page_config(page_title="Student Final Score Predictor", layout="centered")

st.title("🎓 Student Final Score Predictor")
st.markdown(
    "Use the form below to enter a student's attributes and subject scores. The app will predict the `final_score` using your trained model. Make sure `preprocessor.pkl` and `final_score_model.pkl` are in the same folder as this script."
)

with st.form("input_form"):
    st.header("Student Info & Scores")

    # Subject scores (numeric inputs)
    math_score = st.number_input("Math score", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    history_score = st.number_input("History score", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
    physics_score = st.number_input("Physics score", min_value=0.0, max_value=100.0, value=72.0, step=0.5)
    chemistry_score = st.number_input("Chemistry score", min_value=0.0, max_value=100.0, value=68.0, step=0.5)
    biology_score = st.number_input("Biology score", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
    english_score = st.number_input("English score", min_value=0.0, max_value=100.0, value=78.0, step=0.5)
    geography_score = st.number_input("Geography score", min_value=0.0, max_value=100.0, value=66.0, step=0.5)

    # Categorical inputs
    st.markdown("---")
    st.subheader("Categorical attributes")

    st.markdown(
        "**Important:** The dropdown values must match the *original values* in your training data for `gender` and `career_aspiration`. If your dataset used different labels (for example 'M'/'F' instead of 'Male'/'Female'), change the options here accordingly."
    )

    gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
    career_aspiration = st.selectbox(
        "Career aspiration",
        options=["Engineering", "Medicine", "Arts", "Commerce", "Others"],
        index=0,
    )

    # Boolean inputs
    part_time_job = st.checkbox("Has a part-time job")
    extracurricular_activities = st.checkbox("Participates in extracurricular activities")

    submitted = st.form_submit_button("Predict final score")

if submitted:
    # Build input DataFrame in the same column names as used during training
    input_dict = {
        "math_score": [math_score],
        "history_score": [history_score],
        "physics_score": [physics_score],
        "chemistry_score": [chemistry_score],
        "biology_score": [biology_score],
        "english_score": [english_score],
        "geography_score": [geography_score],
        "part_time_job": [int(part_time_job)],
        "extracurricular_activities": [int(extracurricular_activities)],
        "gender": [gender],
        "career_aspiration": [career_aspiration],
    }

    X_user = pd.DataFrame.from_dict(input_dict)

    try:
        X_proc = preprocessor.transform(X_user)
    except Exception as e:
        st.error(
            "Preprocessing failed. This usually means the input column names or categorical labels do not match the training data.\nError: {}".format(e)
        )
    else:
        pred = model.predict(X_proc)
        pred_value = float(pred[0])

        st.success("✅ Prediction complete")
        st.metric(label="Predicted final_score", value=round(pred_value, 3))

        # Extra: show simple explanation
        st.markdown("---")
        st.subheader("Prediction details")
        st.write("Predicted final score (rounded):", round(pred_value, 3))

        # Display the raw input and preprocessed vector for debugging
        with st.expander("Show input data (raw)"):
            st.write(X_user)
        with st.expander("Show preprocessed vector"):
            st.write(X_proc)

st.markdown("---")
st.caption("Tip: If the model throws an error about unseen categories, update the dropdown options to match your training labels or retrain the preprocessor with more categories.")

# Footer
st.write("Built with ❤️ using Streamlit. Need this deployed? Ask me to help deploy it to Streamlit Cloud or Docker.")


# In[ ]:




