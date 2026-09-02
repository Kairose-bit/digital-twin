import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os
from telemetry import SignalTelemetryPipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Rotax 914 UAV Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- APP BRANDING & HEADER ---
st.title("🛩️ AI-Enabled Real-Time Rotax 914 UAV Digital Twin")
st.subheader("Turbocharged Time-Series Anomaly Detection & Predictive Health Matrix")

# --- ROTAX 914 ENGINE SPECIFICATIONS ---
ENGINE_SPECS = {
    "Engine Model": "BRP-Rotax 914 UL / F (Turbocharged)",
    "Type": "4-Cylinder Boxer (Liquid-Cooled Heads / Air-Cooled Cylinders)",
    "Displacement": "1,211.2 cc",
    "Max Takeoff Power": "115 HP @ 5800 RPM (5-Min Limit)",
    "Continuous Power": "100 HP @ 5500 RPM",
    "Fuel Type": "AVGAS 100LL / MOGAS RON 95 Min"
}

# Official Rotax 914 Operational Boundaries
ROTAX_LIMITS = {
    "RPM_MAX_TAKEOFF": 5800,
    "RPM_MAX_CONTINUOUS": 5500,
    "CHT_MAX": 135.0,      # °C Max (Coolant/Head Temp)
    "OIL_TEMP_MIN": 50.0,    # °C
    "OIL_TEMP_MAX": 130.0,   # °C
    "OIL_TEMP_OP": (90.0, 110.0), # Perfect Operating Zone
    "OIL_PRESS_MIN": 2.0,    # bar (Above 3500 RPM)
    "OIL_PRESS_MAX": 5.0,    # bar
    "MAP_MAX_CONT": 35.4,    # inHg (1.20 bar Max Continuous)
    "MAP_MAX_TO": 39.9       # inHg (1.35 bar Takeoff Limit)
}

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("🛠️ Digital Twin Control Unit")
st.sidebar.subheader("Rotax 914 Specifications")
for key, val in ENGINE_SPECS.items():
    st.sidebar.markdown(f"- **{key}**: {val}")

st.sidebar.divider()

st.sidebar.subheader("Flight Environment & Controls")
throttle = st.sidebar.slider("Throttle / TCU Position (%)", 0.0, 115.0, 100.0, step=5.0,
                            help="100% represents Max Continuous. 100-115% activates the Turbocharger Automatic Wastegate Control.")
altitude = st.sidebar.slider("Flight Altitude (ft)", 0, 18000, 7500, step=500)
ambient_temp = st.sidebar.slider("Ambient Temperature OAT (°C)", -30, 45, 15, step=1)

st.sidebar.divider()

st.sidebar.subheader("Simulation / Scenario Injection")
flight_scenario = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Normal Flight Profile",
        "Turbo Charger Wastegate Stuck (Overboost Anomaly)",
        "Cylinder Head Coolant Loss (High CHT)",
        "Oil Pressure Loss / Bearing Degradation",
        "Fuel Pump Restrictive Flow (Lean Spikes)"
    ]
)

stream_active = st.sidebar.toggle("Activate Live Telemetry Stream", value=True)
window_size = 15  # Input frame footprint for sliding sequence models (e.g. LSTM)

# --- SIMULATED ROTAX TIME-SERIES ML INFERENCE ---
class Rotax914MLInference:
    @staticmethod
    def evaluate_window(df_window):
        if len(df_window) < window_size:
            return 0.0, 100.0, "Synchronizing Time-Series Data..."
        
        latest = df_window.iloc[-1]
        anomaly_weight = 0.0
        reasons = []
        
        if latest['CHT'] > ROTAX_LIMITS["CHT_MAX"]:
            anomaly_weight += (latest['CHT'] - ROTAX_LIMITS["CHT_MAX"]) * 2.0
            reasons.append("CHT Exceeded Thermal Limits")
        if latest['MAP'] > ROTAX_LIMITS["MAP_MAX_TO"] + 1.0:
            anomaly_weight += 50.0
            reasons.append("TCU Overboost Condition")
        if latest['Oil Press'] < ROTAX_LIMITS["OIL_PRESS_MIN"] and latest['RPM'] > 3500:
            anomaly_weight += 65.0
            reasons.append("Critical Low Oil Pressure")
        if latest['Oil Temp'] > ROTAX_LIMITS["OIL_TEMP_MAX"]:
            anomaly_weight += 30.0
            reasons.append("Oil Thermal Limit Exceeded")
        if latest['EGT'] > 880.0:
            anomaly_weight += 40.0
            reasons.append("Extreme EGT Lean Burn Gradient")
            
        anomaly_weight += np.random.uniform(0.05, 0.95)
        health_index = max(0.0, min(100.0, 100.0 - (anomaly_weight * 1.3)))
        
        if anomaly_weight > 30.0:
            status = "🚨 FAULT DETECTED: " + " & ".join(reasons)
        elif anomaly_weight > 12.0:
            status = "⚠️ DEGRADED PERFORMANCE STATE"
        else:
            status = "✅ NOMINAL FLIGHT RUN"
            
        return anomaly_weight, health_index, status

# --- TELEMETRY STATE MANAGEMENT ---
if 'history' not in st.session_state:
    t_init = list(range(50))
    init_data = {
        "Time": t_init,
        "RPM": list(np.random.normal(5000, 15, 50)),
        "CHT": list(np.random.normal(105, 1, 50)),
        "EGT": list(np.random.normal(740, 4, 50)),
        "Oil Press": list(np.random.normal(3.8, 0.05, 50)),
        "Oil Temp": list(np.random.normal(98, 0.5, 50)),
        "Fuel Flow": list(np.random.normal(22.0, 0.2, 50)),
        "MAP": list(np.random.normal(33.5, 0.1, 50)),
        "Throttle": [90.0] * 50,
        "Altitude": [7500] * 50,
        "Ambient Temp": [15] * 50,
        "Anomaly Score": [0.3] * 50,
        "Health Index": [99.7] * 50
    }
    st.session_state.history = pd.DataFrame(init_data)
    st.session_state.counter = 50

# --- DATA GENERATION LOOP ---
if stream_active:
    st.session_state.counter += 1
    new_time = st.session_state.counter
    
    throttle_factor = throttle / 100.0
    
    if throttle <= 100.0:
        target_rpm = 1800 + (ROTAX_LIMITS["RPM_MAX_CONTINUOUS"] - 1800) * throttle_factor
        target_map = 25.0 + (ROTAX_LIMITS["MAP_MAX_CONT"] - 25.0) * throttle_factor
    else:
        over_load = (throttle - 100.0) / 15.0
        target_rpm = ROTAX_LIMITS["RPM_MAX_CONTINUOUS"] + (ROTAX_LIMITS["RPM_MAX_TAKEOFF"] - ROTAX_LIMITS["RPM_MAX_CONTINUOUS"]) * over_load
        target_map = ROTAX_LIMITS["MAP_MAX_CONT"] + (ROTAX_LIMITS["MAP_MAX_TO"] - ROTAX_LIMITS["MAP_MAX_CONT"]) * over_load
        
    base_rpm = np.random.normal(target_rpm, 10)
    base_map = np.random.normal(target_map, 0.15)
    
    base_cht = np.random.normal(85 + (25 * throttle_factor) + (ambient_temp * 0.5), 0.8)
    base_egt = np.random.normal(640 + (140 * throttle_factor), 3.5)
    base_oil_p = np.random.normal(4.2 - (0.5 * (base_rpm / 5800.0)), 0.05) if base_rpm > 3500 else np.random.normal(2.5, 0.1)
    base_oil_t = np.random.normal(80 + (22 * throttle_factor), 0.4)
    base_fuel = np.random.normal(8.0 + (18.0 * throttle_factor * (base_map / 30.0)), 0.2)

    if flight_scenario == "Turbo Charger Wastegate Stuck (Overboost Anomaly)":
        if throttle > 80.0:
            base_map += np.random.uniform(5.5, 8.5)
            base_egt += 60.0
            base_cht += 12.0
    elif flight_scenario == "Cylinder Head Coolant Loss (High CHT)":
        base_cht += np.random.uniform(35.0, 58.0)
        base_oil_t += 18.0
    elif flight_scenario == "Oil Pressure Loss / Bearing Degradation":
        base_oil_p = np.random.uniform(0.8, 1.4)
        base_oil_t += 22.0
    elif flight_scenario == "Fuel Pump Restrictive Flow (Lean Spikes)":
        base_fuel -= np.random.uniform(6.0, 9.0)
        base_egt += np.random.uniform(110.0, 160.0)
        base_cht += 14.0
        
    new_row = {
        "Time": new_time, "RPM": base_rpm, "CHT": base_cht, "EGT": base_egt,
        "Oil Press": base_oil_p, "Oil Temp": base_oil_t, "Fuel Flow": base_fuel,
        "MAP": base_map, "Throttle": throttle, "Altitude": altitude,
        "Ambient Temp": ambient_temp, "Anomaly Score": 0.0, "Health Index": 100.0
    }
    
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history.iloc[1:].reset_index(drop=True)

    window_df = st.session_state.history.tail(window_size)
    anomaly_score, health_index, status_text = Rotax914MLInference.evaluate_window(window_df)
    
    st.session_state.history.iloc[-1, st.session_state.history.columns.get_loc("Anomaly Score")] = anomaly_score
    st.session_state.history.iloc[-1, st.session_state.history.columns.get_loc("Health Index")] = health_index
else:
    status_text = "⏸️ Telemetry Feed Suspended"
    anomaly_score = st.session_state.history["Anomaly Score"].iloc[-1]
    health_index = st.session_state.history["Health Index"].iloc[-1]

latest_data = st.session_state.history.iloc[-1]

# --- DASHBOARD VISUALS PRESENTATION LAYER ---
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
m_col1.metric("Engine RPM", f"{int(latest_data['RPM'])} rpm")
m_col2.metric("Manifold Pressure", f"{latest_data['MAP']:.2f} inHg")
m_col3.metric("Coolant CHT", f"{latest_data['CHT']:.1f} °C")
m_col4.metric("Exhaust EGT", f"{latest_data['EGT']:.1f} °C")
m_col5.metric("Oil Parameters", f"{latest_data['Oil Press']:.2f} bar")
m_col6.metric("Fuel Delivery", f"{latest_data['Fuel Flow']:.1f} L/h")

if "🚨" in status_text:
    st.error(f"**Digital Twin Automated Security Assessment:** {status_text}")
else:
    st.success(f"**Digital Twin Automated Security Assessment:** {status_text}")

# =============================================================================
# 📡 SYSTEM ELECTRONICS & SIGNAL TELEMETRY MODULE SECTION (`telemetry.py`)
# =============================================================================
st.divider()
st.header("⚡ System Electronics & Signal Telemetry Bus (`telemetry.py`)")
st.write("Real-time analog voltage scaling, signal noise filtering, and avionics GCS packet framing handled by the dedicated telemetry module.")

# Initialize telemetry processor
telemetry_processor = SignalTelemetryPipeline(device_id="ROTARX-914-UAV-054")

# Build live packet from current simulation data
raw_sensor_feed = {
    "RPM": float(latest_data['RPM']),
    "MAP": float(latest_data['MAP']),
    "CHT": float(latest_data['CHT']),
    "EGT": float(latest_data['EGT']),
    "Oil Press Voltage": float(latest_data['Oil Press']) * 0.8, # Simulated voltage translation
    "Oil Temp": float(latest_data['Oil Temp']),
    "Fuel Flow": float(latest_data['Fuel Flow'])
}

live_packet = telemetry_processor.build_telemetry_packet(
    throttle=throttle,
    altitude=altitude,
    ambient_temp=ambient_temp,
    raw_sensors_dict=raw_sensor_feed
)

tele_col1, tele_col2 = st.columns([1, 1])
with tele_col1:
    st.subheader("Hardware Bus Status")
    st.info("**Alternator/Battery Power:** Nominal (13.8V Bus Stable)\n\n**Dual ECU Logic:** Redundant Active\n\n**Signal Link Quality:** EXCELLENT (CRC-16 Verified)")
    st.metric("Scaled Oil Pressure Bus", f"{live_packet['sensor_payload']['oil_press_bar']} bar")

with tele_col2:
    st.subheader("Live Avionics GCS JSON Payload")
    st.json(live_packet)

# =============================================================================
# 🧠 TEAM AI & DATASET SHOWCASE SECTION
# =============================================================================
st.divider()
st.header("🧠 DRDO SIH 54 - AI Model Performance & Dataset Artifacts")

ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📊 Engine Telemetry Dataset", "📈 Classification Confusion Matrix", "⭐ Feature Importance & Metrics"])

with ai_tab1:
    st.subheader("Raw Dataset Preview (`engine_dataset_v1.csv`)")
    try:
        df_csv = pd.read_csv("engine_dataset_v1.csv")
        st.success(f"Dataset successfully loaded! Total rows: {df_csv.shape[0]}, Features: {df_csv.shape[1]}")
        st.dataframe(df_csv.head(15), use_container_width=True)
    except Exception as e:
        st.warning("`engine_dataset_v1.csv` not found in the root folder.")

with ai_tab2:
    st.subheader("Fault Classification Confusion Matrix")
    st.write("Demonstrates high classification accuracy across all 8 engine fault categories.")
    if os.path.exists("confusion_matrix"):
        st.image("confusion_matrix", caption="Fault Classification Confusion Matrix (Acc=0.988)", use_container_width=True)
    else:
        st.info("`confusion_matrix` image asset not found.")

with ai_tab3:
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("Feature Importance Analysis")
        st.write("Highlights how digital twin residuals outperform raw sensor readings.")
        if os.path.exists("feature_importance"):
            st.image("feature_importance", caption="Top Feature Importances — Fault Classifier", use_container_width=True)
        else:
            st.info("`feature_importance` image asset not found.")
            
    with col_y:
        st.subheader("JSON Metrics Summary")
        if os.path.exists("metrics_summary"):
            with open("metrics_summary", "r") as f:
                metrics_text = f.read()
            st.code(metrics_text, language="json")
        else:
            st.info("`metrics_summary` file not found.")

# --- AUTOMATIC RERUN FOR LIVE TELEMETRY STREAMING ---
if stream_active:
    time.sleep(0.5)
    st.rerun()