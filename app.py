import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("fake_job_model.joblib")

model = load_model()

st.title("Fake Job Posting Detection 🕵️‍♂️")
st.markdown("This application predicts whether a job posting is fraudulent or legitimate based on its description and other characteristics.")

st.sidebar.header("Job Details")

# Initialize session state for default values
if 'title' not in st.session_state:
    st.session_state.title = "Software Engineer"
if 'company_profile' not in st.session_state:
    st.session_state.company_profile = "We are a great company."
if 'description' not in st.session_state:
    st.session_state.description = "Looking for a software engineer to build great products."
if 'requirements' not in st.session_state:
    st.session_state.requirements = "Python, SQL, Machine Learning"
if 'benefits' not in st.session_state:
    st.session_state.benefits = "Health insurance, 401k"
if 'telecommuting' not in st.session_state:
    st.session_state.telecommuting = False
if 'has_company_logo' not in st.session_state:
    st.session_state.has_company_logo = True
if 'has_questions' not in st.session_state:
    st.session_state.has_questions = True

# Sample Buttons
st.sidebar.markdown("### Quick Tests")
col1, col2 = st.sidebar.columns(2)
if col1.button("Legitimate Sample"):
    st.session_state.title = "Senior Data Scientist"
    st.session_state.company_profile = "DataCorp is a leading analytics provider with a global footprint."
    st.session_state.description = "We are looking for an experienced Data Scientist to lead our machine learning initiatives."
    st.session_state.requirements = "MS/PhD in Computer Science. 5+ years of Python and Scikit-Learn."
    st.session_state.benefits = "Competitive salary, equity, full health coverage."
    st.session_state.telecommuting = False
    st.session_state.has_company_logo = True
    st.session_state.has_questions = True
    st.rerun()

if col2.button("Scam Sample"):
    st.session_state.title = "WORK FROM HOME DATA ENTRY - EARN $500/DAY"
    st.session_state.company_profile = ""
    st.session_state.description = "URGENT HIRING! Work from the comfort of your home. No experience needed. Just need a laptop."
    st.session_state.requirements = "Must be able to type fast."
    st.session_state.benefits = "Make your own hours. Huge money."
    st.session_state.telecommuting = True
    st.session_state.has_company_logo = False
    st.session_state.has_questions = False
    st.rerun()
    
st.sidebar.markdown("---")

# Inputs
title = st.sidebar.text_input("Job Title", st.session_state.title)
company_profile = st.sidebar.text_area("Company Profile", st.session_state.company_profile)
description = st.sidebar.text_area("Job Description", st.session_state.description)
requirements = st.sidebar.text_area("Requirements", st.session_state.requirements)
benefits = st.sidebar.text_area("Benefits", st.session_state.benefits)

# Categorical inputs
employment_type = st.sidebar.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"])
required_experience = st.sidebar.selectbox("Required Experience", ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable", "Unknown"])
required_education = st.sidebar.selectbox("Required Education", ["Unspecified", "High School or equivalent", "Some College Coursework Completed", "Bachelor's Degree", "Master's Degree", "Doctorate", "Professional", "Unknown"])

# Binary inputs
telecommuting = st.sidebar.checkbox("Telecommuting (Remote)", st.session_state.telecommuting)
has_company_logo = st.sidebar.checkbox("Has Company Logo", st.session_state.has_company_logo)
has_questions = st.sidebar.checkbox("Has Screening Questions", st.session_state.has_questions)

if st.button("Predict Fraud", type="primary"):
    # Combine text the same way it was done during training
    combined_text = title + " " + company_profile + " " + description + " " + requirements + " " + benefits
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'combined_text': [combined_text],
        'employment_type': [employment_type],
        'required_experience': [required_experience],
        'required_education': [required_education],
        'telecommuting': [int(telecommuting)],
        'has_company_logo': [int(has_company_logo)],
        'has_questions': [int(has_questions)]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result")
    
    # Try to get probability
    try:
        fraud_prob = model.predict_proba(input_data)[0][1]
        has_prob = True
    except AttributeError:
        fraud_prob = 1.0 if prediction == 1 else 0.0
        has_prob = False

    if prediction == 1:
        st.error("🚨 **FRAUDULENT JOB POSTING** 🚨")
        if has_prob:
            st.metric(label="Fraud Risk Score", value=f"{fraud_prob * 100:.2f}%")
            progress_value = max(0.0, min(1.0, float(fraud_prob)))
            st.progress(progress_value)
            
            # Red Flag Explainer
            st.markdown("### 🚩 Key Red Flags Detected")
            st.markdown("Based on common patterns in fake job postings, here are potential issues with this listing:")
            
            flags = []
            if not company_profile.strip():
                flags.append("• **Missing Company Profile**: Fraudulent postings often lack details about the hiring company.")
            if not has_company_logo:
                flags.append("• **No Company Logo**: Genuine companies usually provide a logo.")
            if not has_questions:
                flags.append("• **No Screening Questions**: Lack of standard application questions can be suspicious.")
            if telecommuting and not requirements.strip():
                flags.append("• **Remote Work with No Requirements**: 'Work from home' offers with zero entry barriers are classic scam setups.")
            
            if flags:
                for flag in flags:
                    st.markdown(flag)
            else:
                st.markdown("• The model identified suspicious linguistic patterns in the job description or title.")
    else:
        st.success("✅ **LEGITIMATE JOB POSTING** ✅")
        if has_prob:
            st.metric(label="Legitimacy Confidence", value=f"{(1 - fraud_prob) * 100:.2f}%")
            progress_value = max(0.0, min(1.0, float(1 - fraud_prob)))
            st.progress(progress_value)
            st.caption(f"Estimated Fraud Risk: {fraud_prob * 100:.2f}%")
            
st.markdown("---")
st.markdown("*Disclaimer: This tool provides predictions based on machine learning models and may contain false positives or false negatives.*")
