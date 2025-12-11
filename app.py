import streamlit as st
import pandas as pd
import joblib
import time

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

# Advanced CSS with animations and red/dark blue theme
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a0a14 25%, #0d1b2a 50%, #1c0b1a 75%, #0a1628 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Animated particles background */
    .stApp::before {
        content: '';
        position: fixed;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(220, 38, 38, 0.3), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(30, 58, 138, 0.3), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(220, 38, 38, 0.2), transparent),
            radial-gradient(1px 1px at 80% 10%, rgba(30, 58, 138, 0.2), transparent);
        background-size: 200% 200%;
        animation: particleFloat 20s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleFloat {
        0%, 100% { transform: translate(0, 0); }
        25% { transform: translate(10px, -10px); }
        50% { transform: translate(-5px, 5px); }
        75% { transform: translate(5px, 10px); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Title styling with glow effect */
    h1 {
        color: #ffffff;
        text-align: center;
        font-weight: 700;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(220, 38, 38, 0.5), 
                     0 0 40px rgba(220, 38, 38, 0.3);
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { text-shadow: 0 0 20px rgba(220, 38, 38, 0.5), 0 0 40px rgba(220, 38, 38, 0.3); }
        50% { text-shadow: 0 0 30px rgba(220, 38, 38, 0.8), 0 0 60px rgba(220, 38, 38, 0.5); }
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #93c5fd;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glass card effect for input sections */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(220, 38, 38, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: slideIn 0.6s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Input field styling */
    .stSelectbox, .stNumberInput, .stSlider {
        animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: rgba(30, 58, 138, 0.2) !important;
        border: 1px solid rgba(220, 38, 38, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {
        border-color: rgba(220, 38, 38, 0.6) !important;
        box-shadow: 0 0 15px rgba(220, 38, 38, 0.3);
        transform: translateY(-2px);
    }
    
    /* Label styling */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #93c5fd !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #1e3a8a 0%, #dc2626 100%) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #dc2626 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 10px rgba(220, 38, 38, 0.5);
    }
    
    /* Button styling with advanced animation */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 1rem 2rem;
        border: none;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(220, 38, 38, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    /* Success/Error message styling */
    .stAlert {
        animation: popIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        border-radius: 15px;
        border: none;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 1.5rem;
    }
    
    @keyframes popIn {
        0% {
            opacity: 0;
            transform: scale(0.5) translateY(50px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 0 30px rgba(5, 150, 105, 0.4);
    }
    
    .stError {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.4);
    }
    
    /* Column styling */
    .stColumn {
        animation: fadeInUp 0.8s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(30, 58, 138, 0.2);
        border-left: 4px solid #dc2626;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #93c5fd;
        animation: slideInLeft 0.8s ease;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        background: rgba(30, 58, 138, 0.3);
        transform: translateX(10px);
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.2);
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #dc2626, transparent);
        margin: 2rem 0;
        animation: pulse 2s ease infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(15, 23, 42, 0.4);
        padding: 1rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #93c5fd !important;
        font-weight: 600;
        background: rgba(30, 58, 138, 0.2);
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(220, 38, 38, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-top-color: #dc2626 !important;
        border-right-color: #1e3a8a !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #dc2626 0%, #1e3a8a 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #b91c1c 0%, #1e40af 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown("<h1>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Cardiac Risk ML Model by Salik Ahmad</p>", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class='info-box'>
    <strong>ℹ️ About This Tool:</strong> This advanced prediction system uses machine learning to assess heart disease risk based on various health parameters. Please provide accurate information for the best results.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2 = st.tabs(["📋 Patient Information", "ℹ️ Understanding Results"])

with tab1:
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Basic Information")
        age = st.slider("Age", 18, 100, 40, help="Patient's age in years")
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, 
                                     help="Blood pressure at rest")
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200,
                                      help="Serum cholesterol level")
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                                  help="0 = No, 1 = Yes")
    
    with col2:
        st.markdown("### 🫀 Cardiac Metrics")
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
                                 help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"],
                                   help="Resting electrocardiogram results")
        max_hr = st.slider("Max Heart Rate", 60, 220, 150,
                          help="Maximum heart rate achieved")
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"],
                                      help="Angina during exercise")
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0,
                           help="ST depression induced by exercise")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"],
                               help="Slope of peak exercise ST segment")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Predict Heart Disease Risk")

    if predict_button:
        # Show loading animation
        with st.spinner("Analyzing your data..."):
            time.sleep(1)  # Simulate processing time for better UX
            
            # Create a raw input dictionary
            raw_input = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_' + sex: 1,
                'ChestPainType_' + chest_pain: 1,
                'RestingECG_' + resting_ecg: 1,
                'ExerciseAngina_' + exercise_angina: 1,
                'ST_Slope_' + st_slope: 1
            }

            # Create input dataframe
            input_df = pd.DataFrame([raw_input])

            # Fill in missing columns with 0s
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder columns
            input_df = input_df[expected_columns]

            # Scale the input
            scaled_input = scaler.transform(input_df)

            # Make prediction
            prediction = model.predict(scaled_input)[0]

        # Show result with animation
        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.error("⚠️ **HIGH RISK OF HEART DISEASE DETECTED**")
            st.markdown("""
            <div class='info-box'>
                <strong>⚕️ Recommendation:</strong> Please consult with a cardiologist immediately for further evaluation and appropriate medical intervention.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ **LOW RISK OF HEART DISEASE**")
            st.markdown("""
            <div class='info-box'>
                <strong>💚 Recommendation:</strong> Maintain a healthy lifestyle with regular exercise, balanced diet, and routine check-ups.
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Understanding Your Results")
    
    st.markdown("""
    <div class='info-box'>
        <strong>🎯 Risk Factors:</strong><br>
        • <strong>Age:</strong> Risk increases with age<br>
        • <strong>Blood Pressure:</strong> Higher values indicate greater risk<br>
        • <strong>Cholesterol:</strong> Elevated levels are concerning<br>
        • <strong>Chest Pain:</strong> Different types indicate varying risk levels<br>
        • <strong>Max Heart Rate:</strong> Lower values during exercise may indicate problems<br>
        • <strong>ST Depression (Oldpeak):</strong> Higher values suggest cardiac stress
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <strong>⚠️ Important Disclaimer:</strong><br>
        This tool is for educational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for accurate diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #93c5fd; padding: 1rem;'>
    <p>Powered by Machine Learning | Built with ❤️ by Salik Ahmad</p>
</div>
""", unsafe_allow_html=True)