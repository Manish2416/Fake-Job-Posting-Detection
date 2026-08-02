import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import re

# Set page config
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Vibrant Glassmorphism, Background Image, and Custom Font CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&display=swap');
    
    /* Hide Streamlit Header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Background Image & Main Font */
    .stApp {
        background-image: linear-gradient(rgba(10, 15, 25, 0.8), rgba(10, 15, 25, 0.8)), url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #FAFAFA;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Global Text Colors to pop against dark background */
    h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] p {
        color: #FAFAFA !important;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Vibrant Glassmorphism Card Container */
    .mac-card {
        background: rgba(15, 23, 42, 0.65); /* Deep slate with high transparency */
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .mac-card:hover {
        box-shadow: 0 8px 32px 0 rgba(0, 229, 255, 0.15);
    }
    
    /* Input Fields Styling (Glass effect) */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"] {
        background-color: transparent !important;
    }

    .stTextInput > div > div, 
    .stTextArea > div > div, 
    .stSelectbox > div > div {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .stTextInput input, 
    .stTextArea textarea {
        color: #00E5FF !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }
    
    .stTextInput > div > div:focus-within, 
    .stTextArea > div > div:focus-within {
        border-color: #00E5FF !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2), 0 0 0 2px rgba(0, 229, 255, 0.3) !important;
    }
    
    /* Force Input Labels */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stCheckbox label {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        font-size: 16px !important;
    }
    
    /* Primary Action Button (Gradient) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00E5FF, #0055FF) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.4);
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 18px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.6);
    }
    
    /* Secondary Action Button (Glass) */
    div.stButton > button[kind="secondary"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #FAFAFA !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px;
        font-weight: 600;
        backdrop-filter: blur(4px);
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 16px !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border-color: #00E5FF !important;
        color: #00E5FF !important;
    }
    
    /* Pill Badges */
    .badge-red {
        background: linear-gradient(135deg, #FF0055, #FF3300);
        color: #FFFFFF;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
        box-shadow: 0 4px 10px rgba(255, 0, 85, 0.4);
    }
    
    .badge-green {
        background: linear-gradient(135deg, #00E5FF, #00C853);
        color: #FFFFFF;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
        box-shadow: 0 4px 10px rgba(0, 229, 255, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab"] p {
        color: #94A3B8 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #00E5FF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #00E5FF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("fake_job_model.joblib")

model = load_model()

# Top Header (Glassmorphism)
st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255,255,255,0.1); padding: 20px 40px; margin: -3rem -3rem 2rem -3rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
        <div>
            <h1 style="margin: 0; font-size: 36px; font-weight: 700; background: -webkit-linear-gradient(45deg, #00E5FF, #0055FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI FRAUD DETECTION ENGINE</h1>
            <p style="margin: 0; color: #94A3B8; font-size: 16px; letter-spacing: 1px; text-transform: uppercase;">Intelligent Threat Intelligence Platform</p>
        </div>
        <div>
            <span style="background: rgba(0, 229, 255, 0.1); border: 1px solid #00E5FF; color: #00E5FF; padding: 8px 18px; border-radius: 30px; font-weight: 700; font-size: 14px; text-shadow: 0 0 8px rgba(0,229,255,0.5); box-shadow: 0 0 15px rgba(0,229,255,0.2); text-transform: uppercase; letter-spacing: 1px;">XGBoost Neural Core Active</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Application Modes
app_mode = st.radio("", ["Single Analysis", "Batch Analysis"], horizontal=True, label_visibility="collapsed")

if app_mode == "Single Analysis":
    # Session State Initialization
    if 'title' not in st.session_state: st.session_state.title = ""
    if 'company_profile' not in st.session_state: st.session_state.company_profile = ""
    if 'description' not in st.session_state: st.session_state.description = ""
    if 'requirements' not in st.session_state: st.session_state.requirements = ""
    if 'benefits' not in st.session_state: st.session_state.benefits = ""
    if 'telecommuting' not in st.session_state: st.session_state.telecommuting = False
    if 'has_company_logo' not in st.session_state: st.session_state.has_company_logo = False
    if 'has_questions' not in st.session_state: st.session_state.has_questions = False

    # 2-Column Split Screen for Single Analysis
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown('<div class="mac-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="margin-bottom: 20px;">PAYLOAD PARAMETERS</h3>', unsafe_allow_html=True)
        
        tab_text, tab_meta, tab_presets = st.tabs(["Content Fields", "Metadata Flags", "Quick Presets"])
        
        with tab_text:
            title = st.text_input("Job Title", value=st.session_state.title)
            description = st.text_area("Job Description", value=st.session_state.description, height=140)
            company_profile = st.text_area("Company Profile", value=st.session_state.company_profile, height=80)
            requirements = st.text_area("Requirements", value=st.session_state.requirements, height=80)
            benefits = st.text_area("Benefits", value=st.session_state.benefits, height=80)
        
        with tab_meta:
            col_a, col_b = st.columns(2)
            has_company_logo = col_a.checkbox("Company Logo Present", value=st.session_state.has_company_logo)
            telecommuting = col_b.checkbox("Remote / Telecommute", value=st.session_state.telecommuting)
            has_questions = col_a.checkbox("Screening Questions Included", value=st.session_state.has_questions)
            
            emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"])
            req_exp = st.selectbox("Required Experience", ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable", "Unknown"], index=7)
            req_edu = st.selectbox("Required Education", ["Unspecified", "High School or equivalent", "Some College Coursework Completed", "Bachelor's Degree", "Master's Degree", "Doctorate", "Professional", "Unknown"], index=7)

        with tab_presets:
            st.info("Inject a sample payload into the engine for testing:")
            col_p1, col_p2, col_p3 = st.columns(3)
            if col_p1.button("Standard Job", use_container_width=True):
                st.session_state.title = "Senior Data Scientist"
                st.session_state.company_profile = "DataCorp is a leading analytics provider with a global footprint."
                st.session_state.description = "We are looking for an experienced Data Scientist to lead our machine learning initiatives."
                st.session_state.requirements = "MS/PhD in Computer Science. 5+ years of Python and Scikit-Learn."
                st.session_state.benefits = "Competitive salary, equity, full health coverage."
                st.session_state.telecommuting = False
                st.session_state.has_company_logo = True
                st.session_state.has_questions = True
                st.rerun()

            if col_p2.button("Scam Specimen", use_container_width=True):
                st.session_state.title = "WORK FROM HOME DATA ENTRY - EARN $500/DAY"
                st.session_state.company_profile = ""
                st.session_state.description = "URGENT HIRING! Work from the comfort of your home. No experience needed. Just need a laptop to earn big money."
                st.session_state.requirements = "Must be able to type fast."
                st.session_state.benefits = "Make your own hours. Huge money."
                st.session_state.telecommuting = True
                st.session_state.has_company_logo = False
                st.session_state.has_questions = False
                st.rerun()
                
            if col_p3.button("Clear Form", use_container_width=True):
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
        scan_button = st.button("INITIATE THREAT SCAN", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        
        if scan_button:
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
            
            with st.spinner("Analyzing neural linguistic patterns..."):
                prediction = model.predict(input_data)[0]
                try:
                    fraud_prob = float(model.predict_proba(input_data)[0][1])
                except AttributeError:
                    fraud_prob = 1.0 if prediction == 1 else 0.0
                
                # Risk Triggers Logic
                flags = []
                missing_meta_score = 0
                if not company_profile.strip():
                    flags.append("Missing Company Profile")
                    missing_meta_score += 33
                if not has_company_logo:
                    flags.append("Unverified Identity Logo")
                    missing_meta_score += 33
                if not has_questions:
                    flags.append("No Screening Criteria")
                    missing_meta_score += 34
                    
                req_score = 0
                if telecommuting and not requirements.strip():
                    flags.append("Suspicious Remote Listing")
                    req_score = 100
                elif req_exp == "Unknown" and req_edu == "Unknown":
                    req_score = 80
                    
                ling_score = int(fraud_prob * 100)
                if fraud_prob > 0.8:
                    flags.append("High-Risk Linguistics")
                    
                # Create Vibrant Radar Chart
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                      r=[ling_score, missing_meta_score, req_score],
                      theta=['Linguistic Risk', 'Missing Metadata', 'Suspicious Requirements'],
                      fill='toself',
                      fillcolor='rgba(255, 0, 85, 0.4)' if prediction == 1 else 'rgba(0, 229, 255, 0.4)',
                      line_color='#FF0055' if prediction == 1 else '#00E5FF',
                      line_width=3
                ))
                
                # FIX: Set polar bgcolor to transparent so it doesn't render as a solid white circle
                fig.update_layout(
                  polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 100], color="#FAFAFA", gridcolor="rgba(255,255,255,0.2)"),
                    angularaxis=dict(color="#00E5FF", gridcolor="rgba(255,255,255,0.2)")
                  ),
                  showlegend=False,
                  margin=dict(l=40, r=40, t=40, b=40),
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  font=dict(color='#FAFAFA', size=14, family="Rajdhani, sans-serif")
                )
                    
                # Render Result Card
                is_fraud = prediction == 1
                risk_color = "#FF0055" if is_fraud else "#00E5FF"
                badge_class = "badge-red" if is_fraud else "badge-green"
                badge_text = "CRITICAL THREAT" if is_fraud else "VERIFIED SAFE"
                percentage = fraud_prob if is_fraud else (1 - fraud_prob)
                title_text = "Calculated Probability of Fraudulent Payload" if is_fraud else "Calculated Confidence of Legitimate Payload"
                
                st.markdown(f"""
                    <div class="mac-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 15px; color: #00E5FF; text-transform: uppercase; letter-spacing: 1px;">SYSTEM DIAGNOSIS</span>
                            <span class="{badge_class}">{badge_text}</span>
                        </div>
                        <h1 style="font-size: 64px; margin: 12px 0; font-weight: 800; color: {risk_color}; text-shadow: 0 0 20px {risk_color}88;">{percentage * 100:.1f}%</h1>
                        <p style="color: #94A3B8; font-size: 16px; margin: 0; font-weight: 500;">{title_text}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="mac-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #00E5FF; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">Radar Threat Matrix</h4>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<h4 style="color: #00E5FF; text-transform: uppercase; margin-top: 20px; letter-spacing: 1px; font-weight: 700;">Detected Anomalies</h4>', unsafe_allow_html=True)
                if flags:
                    badge_html = "".join([f'<span class="badge-red">[!] {flag}</span>' for flag in flags])
                    st.markdown(f'<div style="margin-top: 15px; margin-bottom: 20px;">{badge_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="margin-top: 15px;"><span class="badge-green">[+] Verification Passed (Zero Anomalies)</span></div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Idle State
            st.markdown("""
                <div class="mac-card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; height: 500px;">
                    <p style="font-size: 64px; margin: 0; filter: drop-shadow(0 0 20px #00E5FF); color: #00E5FF; font-weight: 700;">[ ! ]</p>
                    <p style="color: #00E5FF; font-weight: 800; font-size: 24px; margin-top: 24px; text-transform: uppercase; letter-spacing: 2px;">System Standby</p>
                    <p style="color: #94A3B8; font-size: 16px; margin-top: 8px; max-width: 300px; font-weight: 500;">Awaiting payload injection. Fill out the parameters and initiate scan to commence neural analysis.</p>
                </div>
            """, unsafe_allow_html=True)

# ----------------- BATCH ANALYSIS FEATURE -----------------
elif app_mode == "Batch Analysis":
    st.markdown('<div class="mac-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #00E5FF; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;">Batch Neural Analysis</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 16px; font-weight: 500;'>Upload a <code>.csv</code> file containing multiple job payloads. The engine will process all vectors simultaneously and generate a downloadable threat report.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Inject CSV File", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully intercepted {len(batch_df)} records.")
            
            with st.expander("Preview Injected Data"):
                st.dataframe(batch_df.head(), use_container_width=True)
            
            if st.button("INITIALIZE BATCH SCAN", type="primary"):
                with st.spinner("Processing batch records via XGBoost Neural Core..."):
                    
                    required_text = ['title', 'company_profile', 'description', 'requirements', 'benefits']
                    for col in required_text:
                        if col not in batch_df.columns: batch_df[col] = ""
                        batch_df[col] = batch_df[col].fillna('')
                        
                    batch_df['combined_text'] = batch_df['title'] + " " + batch_df['company_profile'] + " " + batch_df['description'] + " " + batch_df['requirements'] + " " + batch_df['benefits']
                    
                    required_cat = ['employment_type', 'required_experience', 'required_education']
                    for col in required_cat:
                        if col not in batch_df.columns: batch_df[col] = "Unknown"
                        batch_df[col] = batch_df[col].fillna('Unknown')
                        
                    required_bin = ['telecommuting', 'has_company_logo', 'has_questions']
                    for col in required_bin:
                        if col not in batch_df.columns: batch_df[col] = 0
                        batch_df[col] = batch_df[col].fillna(0).astype(int)
                    
                    inference_df = batch_df[['combined_text', 'employment_type', 'required_experience', 'required_education', 'telecommuting', 'has_company_logo', 'has_questions']]
                    
                    predictions = model.predict(inference_df)
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(inference_df)[:, 1]
                    else:
                        probs = predictions
                    
                    result_df = pd.read_csv(uploaded_file) 
                    result_df['Threat_Assessment'] = ["CRITICAL (Fraud)" if p == 1 else "SAFE (Legit)" for p in predictions]
                    result_df['Fraud_Probability_Score'] = [f"{p*100:.2f}%" for p in probs]
                    
                    st.markdown('<h4 style="color: #00E5FF; text-transform: uppercase; margin-top: 30px; font-weight: 700; letter-spacing: 1px;">Scan Complete</h4>', unsafe_allow_html=True)
                    st.dataframe(result_df[['Threat_Assessment', 'Fraud_Probability_Score', 'title', 'company_profile']].head(50), use_container_width=True)
                    
                    csv_export = result_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="DOWNLOAD SECURE THREAT REPORT (CSV)",
                        data=csv_export,
                        file_name='batch_threat_report.csv',
                        mime='text/csv',
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f"System Error: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
