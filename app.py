import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import re

# Set page config
st.set_page_config(
    page_title="HR Audit Tool | Job Fraud Detector",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 1. Custom Hero Header & Glassmorphism Cards (via CSS Injection)
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Card Container Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Red Flag Pill Badges */
    .badge-red {
        background-color: #ff4b4b22;
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
    
    .badge-green {
        background-color: #00c85322;
        color: #00c853;
        border: 1px solid #00c853;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("fake_job_model.joblib")

model = load_model()

# Header
st.title("🏢 Fake Job Posting Detection")
st.markdown("### Intelligent Job Verification & Trust Platform")

# Session State Initialization
if 'title' not in st.session_state:
    st.session_state.title = ""
if 'company_profile' not in st.session_state:
    st.session_state.company_profile = ""
if 'description' not in st.session_state:
    st.session_state.description = ""
if 'requirements' not in st.session_state:
    st.session_state.requirements = ""
if 'benefits' not in st.session_state:
    st.session_state.benefits = ""
if 'telecommuting' not in st.session_state:
    st.session_state.telecommuting = False
if 'has_company_logo' not in st.session_state:
    st.session_state.has_company_logo = False
if 'has_questions' not in st.session_state:
    st.session_state.has_questions = False

# 4. Tabbed Form Layout (Clean Up Main Screen)
tab1, tab2, tab3 = st.tabs(["📝 Job Content", "🏢 Company Metadata", "🚀 Quick Presets"])

with tab1:
    title = st.text_input("Job Title", value=st.session_state.title)
    company_profile = st.text_area("Company Profile", value=st.session_state.company_profile, height=80)
    description = st.text_area("Job Description", value=st.session_state.description, height=150)
    requirements = st.text_area("Requirements", value=st.session_state.requirements, height=100)
    benefits = st.text_area("Benefits", value=st.session_state.benefits, height=80)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"])
        has_company_logo = st.checkbox("Has Company Logo?", value=st.session_state.has_company_logo)
        has_questions = st.checkbox("Applicant Screening Questions?", value=st.session_state.has_questions)
    with col2:
        req_exp = st.selectbox("Required Experience", ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable", "Unknown"], index=7)
        req_edu = st.selectbox("Required Education", ["Unspecified", "High School or equivalent", "Some College Coursework Completed", "Bachelor's Degree", "Master's Degree", "Doctorate", "Professional", "Unknown"], index=7)
        telecommuting = st.checkbox("Remote / Telecommuting?", value=st.session_state.telecommuting)

with tab3:
    st.info("Click a sample to quickly populate the fields:")
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Load Legitimate Job", use_container_width=True):
        st.session_state.title = "Senior Data Scientist"
        st.session_state.company_profile = "DataCorp is a leading analytics provider with a global footprint."
        st.session_state.description = "We are looking for an experienced Data Scientist to lead our machine learning initiatives."
        st.session_state.requirements = "MS/PhD in Computer Science. 5+ years of Python and Scikit-Learn."
        st.session_state.benefits = "Competitive salary, equity, full health coverage."
        st.session_state.telecommuting = False
        st.session_state.has_company_logo = True
        st.session_state.has_questions = True
        st.rerun()

    if col_b.button("Load Scam Example", use_container_width=True):
        st.session_state.title = "WORK FROM HOME DATA ENTRY - EARN $500/DAY"
        st.session_state.company_profile = ""
        st.session_state.description = "URGENT HIRING! Work from the comfort of your home. No experience needed. Just need a laptop to earn big money."
        st.session_state.requirements = "Must be able to type fast."
        st.session_state.benefits = "Make your own hours. Huge money."
        st.session_state.telecommuting = True
        st.session_state.has_company_logo = False
        st.session_state.has_questions = False
        st.rerun()
        
    if col_c.button("Clear Form", use_container_width=True):
        st.session_state.title = ""
        st.session_state.company_profile = ""
        st.session_state.description = ""
        st.session_state.requirements = ""
        st.session_state.benefits = ""
        st.session_state.telecommuting = False
        st.session_state.has_company_logo = False
        st.session_state.has_questions = False
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
run_audit = st.button("Run Diagnostic Scan", type="primary", use_container_width=True)

# 2. Replace st.progress with an Interactive Gauge Meter
def create_risk_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        number = {'suffix': "%", 'font': {'size': 36, 'color': '#FAFAFA'}},
        title = {'text': "Fraud Risk Level", 'font': {'size': 18, 'color': '#FAFAFA'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#FAFAFA"},
            'bar': {'color': "#ff4b4b" if risk_score > 0.5 else "#00c853"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 200, 83, 0.15)'},
                {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(255, 75, 75, 0.15)'}
            ],
        }
    ))
    fig.update_layout(
        height=280, 
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FAFAFA"}
    )
    return fig

# 5. Real-Time Text Analyzer (Explainable AI Highlight Box)
def highlight_suspicious_words(text, suspicious_terms):
    highlighted_text = text
    for term in suspicious_terms:
        # Case insensitive replacement using regex
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted_text = pattern.sub(
            lambda m: f'<mark style="background-color: #ff4b4b88; color: white; padding: 2px 4px; border-radius: 4px;">{m.group(0)}</mark>',
            highlighted_text
        )
    return highlighted_text

st.markdown("---")

if run_audit:
    combined_text = title + " " + company_profile + " " + description + " " + requirements + " " + benefits
    
    input_data = pd.DataFrame({
        'combined_text': [combined_text],
        'employment_type': [emp_type],
        'required_experience': [req_exp],
        'required_education': [req_edu],
        'telecommuting': [int(telecommuting)],
        'has_company_logo': [int(has_company_logo)],
        'has_questions': [int(has_questions)]
    })
    
    with st.spinner("Analyzing parameters and linguistic patterns..."):
        prediction = model.predict(input_data)[0]
        try:
            fraud_prob = float(model.predict_proba(input_data)[0][1])
        except AttributeError:
            fraud_prob = 1.0 if prediction == 1 else 0.0
            
        st.markdown("## Diagnostic Output")
        
        col_res1, col_res2 = st.columns([1, 1.2])
        
        with col_res1:
            st.plotly_chart(create_risk_gauge(fraud_prob), use_container_width=True)
            
        with col_res2:
            st.markdown("### 🚩 Detected Threat Indicators")
            
            # 3. Replace Bullet Points with Tag Badges
            flags = []
            if not company_profile.strip():
                flags.append("Missing Company Profile")
            if not has_company_logo:
                flags.append("No Official Company Logo")
            if not has_questions:
                flags.append("No Applicant Screening Questions")
            if telecommuting and not requirements.strip():
                flags.append("Remote Work without Skill Requirements")
                
            if fraud_prob > 0.8:
                flags.append("High-Risk Linguistic Signatures")
                
            if flags:
                badge_html = "".join([f'<span class="badge-red">⚠️ {flag}</span>' for flag in flags])
            else:
                badge_html = '<span class="badge-green">✅ Verification Passed (No Flags)</span>'
                
            st.markdown(f'<div style="margin-top: 10px; margin-bottom: 20px;">{badge_html}</div>', unsafe_allow_html=True)
            
            # 5. Highlight Suspicious Text
            st.markdown("#### Explainable AI Analysis")
            suspicious_terms = ["work from home", "urgent", "no experience", "big money", "earn", "$500/day", "huge money", "cash", "typing"]
            
            full_posting_text = f"**{title}**\n\n{description}\n\n{requirements}\n\n{benefits}"
            
            if fraud_prob > 0.5:
                highlighted_html = highlight_suspicious_words(full_posting_text, suspicious_terms)
                st.markdown(f'<div style="background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; font-size: 0.9em; max-height: 200px; overflow-y: auto;">{highlighted_html.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            else:
                st.info("No highly suspicious linguistic patterns were detected in this payload.")
