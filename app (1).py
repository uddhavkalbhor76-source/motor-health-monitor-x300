# =============================================================================
# MDM-271-MEC | AIML Assignment | SPPU 2024 Pattern
# Topic      : Motor Health Monitor using Random Forest Classifier
# Student    : Kalpesh K Ghodke — Second Year Mechanical Engineering — NMIET
# Description: Streamlit web app that classifies electric motor health into
#              four conditions: Healthy, Bearing Wear, Winding Fault,
#              and Rotor Imbalance — from synthetic sensor data.
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor Health Monitor",
    page_icon="🔌",
    layout="wide"
)

# ── Custom CSS (dark industrial theme) ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0b0f1a; }
    .stApp { background-color: #0b0f1a; }
    .healthy-badge {
        background: #064e3b; color: #34d399;
        border: 1px solid #34d399;
        border-radius: 20px; padding: 6px 18px;
        font-weight: bold; font-size: 15px;
    }
    .fault-badge {
        background: #7f1d1d; color: #f87171;
        border: 1px solid #f87171;
        border-radius: 20px; padding: 6px 18px;
        font-weight: bold; font-size: 15px;
    }
    .warn-badge {
        background: #78350f; color: #fbbf24;
        border: 1px solid #fbbf24;
        border-radius: 20px; padding: 6px 18px;
        font-weight: bold; font-size: 15px;
    }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stSlider > div > div > div { color: #60a5fa; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODULE 1 — SYNTHETIC DATA GENERATION
# Simulates real electric motor sensor readings for four health conditions.
# Features: Vibration (mm/s), Current Draw (A), Winding Temp (°C),
#           Noise Level (dB), RPM Deviation, Power Factor, Insulation Resistance (MΩ)
# Labels: 0=Healthy, 1=Bearing Wear, 2=Winding Fault, 3=Rotor Imbalance
# =============================================================================

@st.cache_data
def generate_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic electric motor health dataset.
    Each condition has distinct physical sensor signatures:
    Bearing Wear     → high vibration + noise
    Winding Fault    → high current + winding temp + low insulation resistance
    Rotor Imbalance  → high vibration + high RPM deviation + noise
    """
    rng = np.random.default_rng(random_state)
    n = n_samples // 4

    # ── Class 0: Healthy ─────────────────────────────────────────────────────
    healthy = pd.DataFrame({
        "Vibration_mm_s":          rng.normal(1.5,  0.3,  n),
        "Current_Draw_A":          rng.normal(8.0,  0.5,  n),
        "Winding_Temp_C":          rng.normal(55,   5,    n),
        "Noise_Level_dB":          rng.normal(62,   3,    n),
        "RPM_Deviation":           rng.normal(2,    1,    n),
        "Power_Factor":            rng.normal(0.92, 0.02, n),
        "Insulation_Resistance_MO":rng.normal(95,   5,    n),
        "Condition": 0
    })

    # ── Class 1: Bearing Wear ─────────────────────────────────────────────────
    bearing = pd.DataFrame({
        "Vibration_mm_s":          rng.normal(7.5,  1.2,  n),
        "Current_Draw_A":          rng.normal(9.5,  0.8,  n),
        "Winding_Temp_C":          rng.normal(68,   6,    n),
        "Noise_Level_dB":          rng.normal(82,   5,    n),
        "RPM_Deviation":           rng.normal(6,    2,    n),
        "Power_Factor":            rng.normal(0.84, 0.03, n),
        "Insulation_Resistance_MO":rng.normal(80,   8,    n),
        "Condition": 1
    })

    # ── Class 2: Winding Fault ────────────────────────────────────────────────
    winding = pd.DataFrame({
        "Vibration_mm_s":          rng.normal(2.8,  0.5,  n),
        "Current_Draw_A":          rng.normal(14.5, 1.5,  n),
        "Winding_Temp_C":          rng.normal(105,  10,   n),
        "Noise_Level_dB":          rng.normal(70,   4,    n),
        "RPM_Deviation":           rng.normal(4,    1.5,  n),
        "Power_Factor":            rng.normal(0.72, 0.04, n),
        "Insulation_Resistance_MO":rng.normal(18,   5,    n),
        "Condition": 2
    })

    # ── Class 3: Rotor Imbalance ──────────────────────────────────────────────
    rotor = pd.DataFrame({
        "Vibration_mm_s":          rng.normal(11.0, 1.8,  n),
        "Current_Draw_A":          rng.normal(10.5, 1.0,  n),
        "Winding_Temp_C":          rng.normal(72,   7,    n),
        "Noise_Level_dB":          rng.normal(88,   6,    n),
        "RPM_Deviation":           rng.normal(18,   3.5,  n),
        "Power_Factor":            rng.normal(0.80, 0.03, n),
        "Insulation_Resistance_MO":rng.normal(75,   7,    n),
        "Condition": 3
    })

    df = pd.concat([healthy, bearing, winding, rotor], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = df.clip(lower=0)
    return df


# =============================================================================
# MODULE 2 — RANDOM FOREST MODEL TRAINING
# Random Forest Classifier — Unit II of MDM-271-MEC syllabus.
# Ensemble of 150 decision trees using majority voting for classification.
# =============================================================================

@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Trains a Random Forest Classifier for multi-class motor fault detection.
    Random Forest is robust to noisy sensor data — ideal for motor diagnostics.
    Returns: model, scaler, accuracy, feature names, report, confusion matrix
    """
    feature_cols = [
        "Vibration_mm_s", "Current_Draw_A", "Winding_Temp_C",
        "Noise_Level_dB", "RPM_Deviation", "Power_Factor",
        "Insulation_Resistance_MO"
    ]
    X = df[feature_cols]
    y = df["Condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Random Forest — 150 trees, max depth 12
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    labels   = ["Healthy", "Bearing Wear", "Winding Fault", "Rotor Imbalance"]
    report   = classification_report(y_test, y_pred, target_names=labels)
    cm       = confusion_matrix(y_test, y_pred)

    return model, scaler, accuracy, feature_cols, report, cm, labels


# =============================================================================
# MODULE 3 — STREAMLIT UI
# =============================================================================

def badge(label, kind="healthy"):
    css = {"healthy": "healthy-badge", "fault": "fault-badge", "warn": "warn-badge"}
    return f'<span class="{css.get(kind, "fault-badge")}">{label}</span>'


def main():

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔌 Motor Health Monitor")
    st.markdown(
        """
        **Course:** MDM-271-MEC — AI & ML in Mechanical Engineering &nbsp;|&nbsp;
        **College:** NMIET, Pune &nbsp;|&nbsp; **Pattern:** SPPU 2024  
        This system uses a **Random Forest Classifier** to monitor electric motor
        health and classify it into four conditions from real-time sensor data.
        """
    )
    st.divider()

    # ── Load data & train ─────────────────────────────────────────────────────
    with st.spinner("🔄 Training Random Forest model..."):
        df = generate_dataset()
        model, scaler, accuracy, feature_cols, report, cm, labels = train_model(df)

    # ── Top KPI row ───────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🤖 Algorithm",       "Random Forest")
    k2.metric("🎯 Test Accuracy",   f"{accuracy*100:.2f}%")
    k3.metric("📊 Training Samples","960")
    k4.metric("🔢 Sensor Features", "7 sensors")
    st.divider()

    # ── Layout: Left (inputs) | Right (charts) ────────────────────────────────
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        # ── Data preview ─────────────────────────────────────────────────────
        with st.expander("📋 Sample Training Data (first 8 rows)"):
            cond_map = {
                0: "Healthy", 1: "Bearing Wear",
                2: "Winding Fault", 3: "Rotor Imbalance"
            }
            preview = df.head(8).copy()
            preview["Condition"] = preview["Condition"].map(cond_map)
            st.dataframe(preview, use_container_width=True)

        # ── Sensor input sliders ──────────────────────────────────────────────
        st.subheader("🎛️ Live Motor Sensor Input")
        st.markdown("Simulate real-time motor sensor readings:")

        vibration  = st.slider("Vibration (mm/s)",           0.5,  15.0, 1.5,  0.1,
                               help="Overall vibration level of motor housing")
        current    = st.slider("Current Draw (A)",            5.0,  20.0, 8.0,  0.1,
                               help="Phase current — winding faults cause spikes")
        winding    = st.slider("Winding Temperature (°C)",   30.0, 130.0, 55.0, 1.0,
                               help="Motor winding temperature — critical for insulation life")
        noise      = st.slider("Noise Level (dB)",           50.0, 100.0, 62.0, 0.5,
                               help="Acoustic noise from motor — bearing wear increases noise")
        rpm_dev    = st.slider("RPM Deviation",               0.0,  25.0,  2.0, 0.5,
                               help="Speed fluctuation from setpoint — rotor imbalance causes this")
        pf         = st.slider("Power Factor",                0.60,  1.00, 0.92, 0.01,
                               help="Ratio of real power to apparent power — drops with winding faults")
        insulation = st.slider("Insulation Resistance (MΩ)", 5.0,  110.0, 95.0, 1.0,
                               help="Low value = deteriorated winding insulation = fault risk")

        # ── Predict button ────────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔍 Analyse Motor Health", type="primary", use_container_width=True):

            inp    = np.array([[vibration, current, winding, noise,
                                rpm_dev, pf, insulation]])
            inp_sc = scaler.transform(inp)
            pred   = model.predict(inp_sc)[0]
            proba  = model.predict_proba(inp_sc)[0]
            conf   = proba[pred] * 100

            condition_info = {
                0: ("✅ MOTOR HEALTHY", "healthy",
                    "All sensor readings are within normal operating range. "
                    "Motor is functioning efficiently. No maintenance required. "
                    "Continue scheduled monitoring every 30 days."),
                1: ("⚠️ BEARING WEAR DETECTED", "warn",
                    f"High vibration ({vibration} mm/s) and noise ({noise} dB) indicate "
                    "progressive bearing degradation. "
                    "🛑 Schedule bearing inspection and lubrication within 72 hours. "
                    "Replace bearings if wear exceeds tolerance."),
                2: ("🔴 WINDING FAULT DETECTED", "fault",
                    f"Elevated current draw ({current} A) and winding temperature ({winding}°C) "
                    f"with low insulation resistance ({insulation} MΩ) confirm winding insulation breakdown. "
                    "🛑 Shut down motor immediately. Risk of electrical failure and fire."),
                3: ("🔴 ROTOR IMBALANCE DETECTED", "fault",
                    f"Excessive vibration ({vibration} mm/s) and RPM deviation ({rpm_dev} rpm) "
                    "confirm mechanical rotor imbalance. "
                    "🛑 Stop motor. Perform dynamic balancing before restarting. "
                    "Continued operation risks bearing and shaft damage.")
            }

            label, badge_kind, explanation = condition_info[pred]

            st.markdown("### 🩺 Diagnosis Result")
            st.markdown(badge(label, badge_kind), unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {conf:.1f}%")
            st.markdown(f"**Explanation:** {explanation}")

            st.markdown("**Confidence per class:**")
            for i, (lbl, p) in enumerate(zip(labels, proba)):
                st.progress(float(p), text=f"{lbl}: {p*100:.1f}%")

    with right:
        st.subheader("📊 Model Evaluation")
        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Class Report", "Feature Importance"])

        with tab1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor('#0b0f1a')
            ax.set_facecolor('#0b0f1a')
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='#1a1f2e', ax=ax,
                annot_kws={"size": 11, "weight": "bold"}
            )
            ax.set_xlabel("Predicted", color='#94a3b8', fontsize=10)
            ax.set_ylabel("Actual",    color='#94a3b8', fontsize=10)
            ax.set_title("Confusion Matrix — Random Forest", color='#e2e8f0',
                         fontsize=11, pad=12)
            ax.tick_params(colors='#94a3b8')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.subheader("Classification Report")
            st.code(report)
            st.caption("Precision, Recall, F1-Score per motor health condition.")

        with tab3:
            # Feature importance bar chart
            importances = model.feature_importances_
            feat_labels = [
                "Vibration", "Current Draw", "Winding Temp",
                "Noise Level", "RPM Deviation", "Power Factor", "Insulation Res."
            ]
            sorted_idx = np.argsort(importances)
            colors = ['#f87171' if i == sorted_idx[-1] else '#60a5fa'
                      for i in range(len(importances))]
            sorted_colors = [colors[i] for i in sorted_idx]

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#0b0f1a')
            ax2.set_facecolor('#0b0f1a')
            bars = ax2.barh(
                [feat_labels[i] for i in sorted_idx],
                importances[sorted_idx] * 100,
                color=sorted_colors, edgecolor='none'
            )
            ax2.bar_label(bars, fmt="%.1f%%", padding=3,
                          color='#e2e8f0', fontsize=9)
            ax2.set_xlabel("Importance (%)", color='#94a3b8')
            ax2.set_title("Feature Importance — Random Forest",
                          color='#e2e8f0', fontsize=11)
            ax2.tick_params(colors='#94a3b8')
            ax2.spines['bottom'].set_color('#2d3748')
            ax2.spines['left'].set_color('#2d3748')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            st.caption(
                "Red bar = most important feature. "
                "Determined by how much each sensor reduces impurity across all 150 trees."
            )

    # ── Viva Explainer ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📚 How does Random Forest work? (Viva Preparation — Unit II MDM-271-MEC)"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Random Forest** is an ensemble learning algorithm from **Unit II**.

            **Step-by-step:**
            1. Training data (960 samples) is split into random subsets
            2. A **Decision Tree** is trained on each subset — 150 trees total
            3. Each tree independently predicts the motor condition
            4. Final result = **majority vote** across all 150 trees

            **Why Random Forest for motor faults?**
            - Motor sensor data is noisy — RF handles noise well
            - Multiple fault types — RF handles multi-class naturally
            - Feature importance — tells engineers which sensor matters most
            - Less overfitting than a single Decision Tree

            **Key hyperparameters:**
            | Parameter | Value | Reason |
            |-----------|-------|--------|
            | `n_estimators` | 150 | More trees = more stable |
            | `max_depth` | 12 | Prevents overfitting |
            | `random_state` | 42 | Reproducibility |
            | `n_jobs` | -1 | Use all CPU cores |
            """)
        with col_b:
            st.markdown("""
            **The 7 Motor Sensor Features:**

            | Feature | Why It Matters |
            |---------|----------------|
            | Vibration | Bearing wear & rotor imbalance |
            | Current Draw | Winding faults cause current spikes |
            | Winding Temp | Overheating = insulation breakdown |
            | Noise Level | Bearing degradation = acoustic noise |
            | RPM Deviation | Rotor imbalance causes speed ripple |
            | Power Factor | Drops with winding faults |
            | Insulation Res. | Low value = imminent winding failure |

            **Industry 4.0 / IoT Context:**
            In real deployment, these 7 sensors are mounted on
            the motor and transmit data via IoT (MQTT/HTTP) to
            this Streamlit dashboard — enabling 24/7 automated
            motor health monitoring without human inspection.
            """)

    # ── Admin Panel Info ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("🗂️ Admin Panel Details (Project Portfolio)"):
        st.markdown("""
        | Field | Value |
        |-------|-------|
        | **Project Title** | Motor Health Monitor |
        | **Category** | AIML, Predictive Maintenance, IoT |
        | **Tech Stack** | Python, Streamlit, Scikit-learn, Pandas, NumPy, Seaborn |
        | **Author / Team** | Kalpesh K Ghodke — NMIET |
        | **Image URL** | https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800 |
        """)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
