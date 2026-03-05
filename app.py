# ===================== IMPORTS =====================
import streamlit as st
import numpy as np
import joblib
import random

# ===================== LOAD MODEL =====================
model = joblib.load("breast_cancer_pipeline.pkl")

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .main {
        background: linear-gradient(45deg, #f5f5f5, #e8f5e8, #f0f8ff, #fffacd);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stNumberInput input {
        border-radius: 5px;
        transition: border-color 0.3s ease;
    }
    .stNumberInput input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .sidebar .sidebar-content {
        background-color: rgba(232, 245, 232, 0.9);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# ===================== RANDOM SAMPLE FUNCTION =====================
def random_sample():
    return {
        "radius": round(random.uniform(6, 30), 2),
        "texture": round(random.uniform(10, 40), 2),
        "perimeter": round(random.uniform(40, 200), 2),
        "area": round(random.uniform(200, 2500), 2),
        "smoothness": round(random.uniform(0.05, 0.25), 3),
        "compactness": round(random.uniform(0.02, 0.35), 3),
    }

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app predicts breast cancer risk based on mammography features using a machine learning model.
    
    **Features used:**
    - Radius
    - Texture  
    - Perimeter
    - Area
    - Smoothness
    - Compactness
    
    **Note:** This is for educational purposes only. Consult a medical professional for actual diagnosis.
    """)
    
    st.header("🔄 Random Sample")
    if st.button("🎲 Generate New Sample", key="random_btn"):
        st.session_state.sample = random_sample()
        st.rerun()

# ===================== MAIN CONTENT =====================
st.title("🩺 Breast Cancer Risk Prediction")
st.markdown("### Enter Mammography Features")
st.write("Adjust the values below or use the random sample button in the sidebar.")

if "sample" not in st.session_state:
    st.session_state.sample = random_sample()

# ===================== INPUTS (6 MAIN FEATURES) =====================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📏 Size Features")
    radius = st.number_input("Radius (mean)", value=st.session_state.sample["radius"], min_value=0.0, step=0.01)
    perimeter = st.number_input("Perimeter (mean)", value=st.session_state.sample["perimeter"], min_value=0.0, step=0.01)

with col2:
    st.subheader("🔍 Texture Features")
    texture = st.number_input("Texture (mean)", value=st.session_state.sample["texture"], min_value=0.0, step=0.01)
    smoothness = st.number_input("Smoothness (mean)", value=st.session_state.sample["smoothness"], min_value=0.0, step=0.001)

with col3:
    st.subheader("📐 Shape Features")
    area = st.number_input("Area (mean)", value=st.session_state.sample["area"], min_value=0.0, step=0.01)
    compactness = st.number_input("Compactness (mean)", value=st.session_state.sample["compactness"], min_value=0.0, step=0.001)

# ===================== PREDICT =====================
st.markdown("---")
col_pred, col_space = st.columns([1, 3])
with col_pred:
    predict_button = st.button("🔍 Predict Risk", use_container_width=True)

if predict_button:
    # Create 30-feature input
    input_30 = np.zeros(30)

    # Fill important features (based on sklearn dataset order)
    input_30[0] = radius
    input_30[1] = texture
    input_30[2] = perimeter
    input_30[3] = area
    input_30[4] = smoothness
    input_30[5] = compactness

    input_30 = input_30.reshape(1, -1)

    probability = model.predict_proba(input_30)[0][1].item()

    prediction = (
        "High Risk (Cancer Detected)" if probability > 0.7
        else "Moderate Risk" if probability > 0.4
        else "Low Risk (Benign)"
    )

    # ===================== RESULTS =====================
    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Results")
    
    if probability > 0.7:
        st.error(f"⚠️ **{prediction}**")
        st.write(f"**Cancer Probability:** {probability:.1%}")
        st.progress(probability)
        st.warning("**Recommendation:** Please consult a medical professional immediately for further evaluation.")
    elif probability > 0.4:
        st.warning(f"🟡 **{prediction}**")
        st.write(f"**Cancer Probability:** {probability:.1%}")
        st.progress(probability)
        st.info("**Recommendation:** Schedule a follow-up appointment with your doctor for additional tests.")
    else:
        st.success(f"✅ **{prediction}**")
        st.write(f"**Cancer Probability:** {probability:.1%}")
        st.progress(probability)
        st.info("**Note:** Regular check-ups are still important for breast health.")
    
    st.markdown('</div>', unsafe_allow_html=True)

