# =============================================================================
# MDM-271-MEC | AIML Assignment | SPPU 2024 Pattern
# Topic      : Motor Health Monitor using SVM (Support Vector Machine)
# Student    : Kalpesh K Ghodke — Second Year Mechanical Engineering — NMIET
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Motor Health Monitor", page_icon="🔌", layout="centered")

st.title("🔌 Motor Health Monitor")
st.markdown(
    "**Course:** MDM-271-MEC &nbsp;|&nbsp; **College:** NMIET, Pune &nbsp;|&nbsp; **SPPU 2024**  \n"
    "Uses **SVM (Support Vector Machine)** to classify motor health from sensor data."
)
st.divider()

# =============================================================================
# MODULE 1 — DATA GENERATION
# 4 features: Vibration, Current, Winding Temperature, Noise Level
# 2 classes : 0 = Healthy | 1 = Faulty
# =============================================================================

@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)
    n   = 300  # samples per class

    healthy = pd.DataFrame({
        "Vibration_mm_s":  rng.normal(1.5,  0.3, n),
        "Current_A":       rng.normal(8.0,  0.5, n),
        "Winding_Temp_C":  rng.normal(55.0, 5.0, n),
        "Noise_dB":        rng.normal(62.0, 3.0, n),
        "Status": 0
    })

    faulty = pd.DataFrame({
        "Vibration_mm_s":  rng.normal(7.5,  1.2, n),
        "Current_A":       rng.normal(14.5, 1.5, n),
        "Winding_Temp_C":  rng.normal(105.0,10.0, n),
        "Noise_dB":        rng.normal(82.0, 5.0, n),
        "Status": 1
    })

    df = pd.concat([healthy, faulty], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.clip(lower=0)
    return df

# =============================================================================
# MODULE 2 — SVM MODEL TRAINING
# Algorithm : SVM with RBF Kernel (Unit II — MDM-271-MEC)
# =============================================================================

@st.cache_resource
def train_svm(df):
    features = ["Vibration_mm_s", "Current_A", "Winding_Temp_C", "Noise_dB"]
    X = df[features]
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_sc, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_sc))
    return model, scaler, acc, features

# ── Train ─────────────────────────────────────────────────────────────────────
df = generate_data()
model, scaler, acc, features = train_svm(df)

# ── Model info ────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Algorithm",    "SVM — RBF Kernel")
c2.metric("Accuracy",     f"{acc * 100:.1f}%")
c3.metric("Total Samples","600 (300 per class)")
st.divider()

# ── Data preview ──────────────────────────────────────────────────────────────
with st.expander("📋 View Sample Training Data"):
    preview = df.head(10).copy()
    preview["Status"] = preview["Status"].map({0: "Healthy", 1: "Faulty"})
    st.dataframe(preview, use_container_width=True)

# =============================================================================
# MODULE 3 — SENSOR INPUT SLIDERS
# =============================================================================

st.subheader("🎛️ Enter Motor Sensor Readings")
st.markdown("Adjust sliders to simulate real-time motor sensor data:")

col_a, col_b = st.columns(2)

with col_a:
    vibration = st.slider(
        "Vibration (mm/s)", 0.5, 15.0, 1.5, 0.1,
        help="High vibration = bearing wear or rotor imbalance"
    )
    current = st.slider(
        "Current Draw (A)", 5.0, 20.0, 8.0, 0.1,
        help="High current = winding fault"
    )

with col_b:
    winding_temp = st.slider(
        "Winding Temperature (C)", 30.0, 130.0, 55.0, 1.0,
        help="High temp = insulation breakdown risk"
    )
    noise = st.slider(
        "Noise Level (dB)", 50.0, 100.0, 62.0, 0.5,
        help="High noise = bearing degradation"
    )

# =============================================================================
# MODULE 4 — PREDICTION & OUTPUT
# =============================================================================

st.divider()

if st.button("🔍 Check Motor Health", type="primary", use_container_width=True):

    inp    = np.array([[vibration, current, winding_temp, noise]])
    inp_sc = scaler.transform(inp)
    pred   = model.predict(inp_sc)[0]
    proba  = model.predict_proba(inp_sc)[0]
    conf   = proba[pred] * 100

    st.subheader("🩺 Diagnosis Result")

    if pred == 0:
        st.success(f"### ✅ Motor Status: HEALTHY")
        st.markdown(
            f"The SVM model classified this motor as **Healthy** with **{conf:.1f}% confidence**.  \n"
            f"All sensor readings are within normal range — "
            f"Vibration: {vibration} mm/s, Current: {current} A, "
            f"Temp: {winding_temp}°C, Noise: {noise} dB.  \n"
            "**Action:** No maintenance required. Monitor every 30 days."
        )
    else:
        st.error(f"### ⚠️ Motor Status: FAULTY")
        st.markdown(
            f"The SVM model classified this motor as **Faulty** with **{conf:.1f}% confidence**.  \n"
            f"One or more sensors exceed safe limits — "
            f"Vibration: {vibration} mm/s, Current: {current} A, "
            f"Temp: {winding_temp}°C, Noise: {noise} dB.  \n"
            "**Action:** 🛑 Inspect motor immediately. Schedule maintenance."
        )

    # Confidence bar chart
    st.markdown("**Prediction Confidence:**")
    fig, ax = plt.subplots(figsize=(5, 2))
    bars = ax.barh(
        ["Healthy", "Faulty"],
        [proba[0] * 100, proba[1] * 100],
        color=["#34d399", "#f87171"],
        edgecolor='none'
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# =============================================================================
# MODULE 5 — HOW SVM WORKS (Viva Reference)
# =============================================================================

st.divider()
with st.expander("📚 How SVM Works — Viva Reference (Unit II)"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **SVM — Support Vector Machine**

        - Finds the best **hyperplane** separating Healthy vs Faulty
        - Maximises the **margin** between both classes
        - **RBF Kernel** maps non-linear data to higher dimensions
        - Data points on the margin boundary = **Support Vectors**

        **Hyperparameters used:**
        | Parameter | Value | Reason |
        |-----------|-------|--------|
        | kernel | rbf | Non-linear data |
        | C | 1 | Balanced margin |
        | gamma | scale | Auto bandwidth |
        """)
    with col2:
        st.markdown("""
        **The 4 Sensor Features:**

        | Sensor | Healthy | Faulty |
        |--------|---------|--------|
        | Vibration | ~1.5 mm/s | ~7.5 mm/s |
        | Current | ~8 A | ~14.5 A |
        | Winding Temp | ~55°C | ~105°C |
        | Noise | ~62 dB | ~82 dB |

        **IoT Context:**
        In real deployment, these sensors mount on
        the motor and send live data to this app
        via MQTT / HTTP for 24x7 monitoring.
        """)

st.divider()
st.caption("MDM-271-MEC | NMIET, Pune | SPPU 2024 Pattern | Kalpesh K Ghodke")
