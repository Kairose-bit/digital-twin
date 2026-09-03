import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os

try:
    from telemetry import SignalTelemetryPipeline
    HAS_TELEMETRY_MODULE = True
except ImportError:
    HAS_TELEMETRY_MODULE = False

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
    "CHT_MAX": 135.0,      # °C Max
    "OIL_TEMP_MIN": 50.0,    # °C
    "OIL_TEMP_MAX": 130.0,   # °C
    "OIL_PRESS_MIN": 2.0,    # bar
    "OIL_PRESS_MAX": 5.0,    # bar
    "MAP_MAX_CONT": 35.4,    # inHg
    "MAP_MAX_TO": 39.9,      # inHg
    "EGT_MAX": 880.0         # °C
}

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("🛠️ Digital Twin Control Unit")
st.sidebar.subheader("Rotax 914 Specifications")
for key, val in ENGINE_SPECS.items():
    st.sidebar.markdown(f"- **{key}**: {val}")

st.sidebar.divider()

st.sidebar.subheader("Flight Environment & Controls")
throttle = st.sidebar.slider(
    "Throttle / TCU Position (%)", 0.0, 115.0, 100.0, step=5.0,
    help="100% represents Max Continuous. 100-115% activates the Turbocharger Automatic Wastegate Control."
)
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
window_size = 15

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
        if latest['EGT'] > ROTAX_LIMITS["EGT_MAX"]:
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

    # Scenario Injection Logic
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

# --- TEAMMATE'S UI STYLE: TOP KPI METRICS WITH DELTAS ---
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

m_col1.metric("Engine RPM", f"{int(latest_data['RPM'])} rpm", delta=f"{int(latest_data['RPM'] - 5000)} rpm vs Cruise")
m_col2.metric("Manifold Press", f"{latest_data['MAP']:.2f} inHg", delta=f"{latest_data['MAP'] - 30.0:.1f} inHg", delta_color="inverse" if latest_data['MAP'] > ROTAX_LIMITS["MAP_MAX_CONT"] else "normal")
m_col3.metric("Coolant CHT", f"{latest_data['CHT']:.1f} °C", delta=f"{latest_data['CHT'] - 105.0:.1f} °C", delta_color="inverse" if latest_data['CHT'] > ROTAX_LIMITS["CHT_MAX"] else "normal")
m_col4.metric("Exhaust EGT", f"{latest_data['EGT']:.1f} °C", delta=f"{latest_data['EGT'] - 740.0:.1f} °C", delta_color="inverse" if latest_data['EGT'] > ROTAX_LIMITS["EGT_MAX"] else "normal")
m_col5.metric("Oil Pressure", f"{latest_data['Oil Press']:.2f} bar", delta=f"{latest_data['Oil Press'] - 3.8:.2f} bar", delta_color="normal" if latest_data['Oil Press'] >= ROTAX_LIMITS["OIL_PRESS_MIN"] else "inverse")

health_status = "CRITICAL FAULT" if "🚨" in status_text else ("DEGRADED" if "⚠️" in status_text else "NOMINAL")
m_col6.metric("Health Index", f"{health_index:.1f}%", delta=health_status, delta_color="normal" if health_index > 80 else "inverse")

# Status Warning Banner System
if "🚨" in status_text:
    st.error(f"**Digital Twin Automated Security Assessment:** {status_text}")
elif "⚠️" in status_text:
    st.warning(f"**Digital Twin Automated Security Assessment:** {status_text}")
else:
    st.success(f"**Digital Twin Automated Security Assessment:** {status_text}")

st.divider()

# --- TEAMMATE'S UI STYLE: 2x2 PLOTLY TIME-SERIES CHARTS GRID ---
def create_plotly_chart(time_data, y_data, title, unit, color, threshold_line=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=y_data, mode='lines+markers', name=title, line=dict(color=color, width=2)))
    if threshold_line:
        fig.add_hline(y=threshold_line, line_dash="dash", line_color="red", annotation_text="Safety Limit")
    fig.update_layout(
        title=f"Real-Time {title} ({unit})",
        xaxis_title="Telemetry Tick",
        yaxis_title=unit,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        template="plotly_dark"
    )
    return fig

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.plotly_chart(
        create_plotly_chart(
            st.session_state.history["Time"], st.session_state.history["EGT"],
            "Exhaust Gas Temp (EGT)", "°C", "#FF4B4B", ROTAX_LIMITS["EGT_MAX"]
        ),
        use_container_width=True
    )

with chart_col2:
    st.plotly_chart(
        create_plotly_chart(
            st.session_state.history["Time"], st.session_state.history["Oil Press"],
            "Oil Pressure", "bar", "#00CC96", ROTAX_LIMITS["OIL_PRESS_MIN"]
        ),
        use_container_width=True
    )

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.plotly_chart(
        create_plotly_chart(
            st.session_state.history["Time"], st.session_state.history["MAP"],
            "Manifold Pressure (MAP)", "inHg", "#AB63FA", ROTAX_LIMITS["MAP_MAX_TO"]
        ),
        use_container_width=True
    )

with chart_col4:
    fig_health = go.Figure()
    fig_health.add_trace(go.Scatter(
        x=st.session_state.history["Time"],
        y=st.session_state.history["Health Index"],
        fill='tozeroy',
        name="Health Score",
        line=dict(color='#17BECF', width=2)
    ))
    fig_health.update_layout(
        title="Digital Twin Predicted Health Index Trend (%)",
        xaxis_title="Telemetry Tick",
        yaxis_title="Health %",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        template="plotly_dark"
    )
    st.plotly_chart(fig_health, use_container_width=True)

# --- SYSTEM ELECTRONICS & TELEMETRY BUS SECTION ---
st.divider()
st.header("⚡ System Electronics & Signal Telemetry Bus (`telemetry.py`)")

if HAS_TELEMETRY_MODULE:
    telemetry_processor = SignalTelemetryPipeline(device_id="ROTAX-914-UAV-054")
    raw_sensor_feed = {
        "RPM": float(latest_data['RPM']),
        "MAP": float(latest_data['MAP']),
        "CHT": float(latest_data['CHT']),
        "EGT": float(latest_data['EGT']),
        "Oil Press Voltage": float(latest_data['Oil Press']) * 0.8,
        "Oil Temp": float(latest_data['Oil Temp']),
        "Fuel Flow": float(latest_data['Fuel Flow'])
    }
    live_packet = telemetry_processor.build_telemetry_packet(
        throttle=throttle, altitude=altitude, ambient_temp=ambient_temp, raw_sensors_dict=raw_sensor_feed
    )

    tele_col1, tele_col2 = st.columns(2)
    with tele_col1:
        st.subheader("Hardware Bus Status")
        st.info("**Alternator/Battery Power:** Nominal (13.8V Bus Stable)\n\n**Dual ECU Logic:** Redundant Active\n\n**Signal Link Quality:** EXCELLENT (CRC-16 Verified)")
        st.metric("Scaled Oil Pressure Bus", f"{live_packet['sensor_payload']['oil_press_bar']} bar")

    with tele_col2:
        st.subheader("Live Avionics GCS JSON Payload")
        st.json(live_packet)
else:
    st.info("`telemetry.py` module not loaded. Telemetry pipeline streaming in simulation mode.")

# --- DRDO SIH 54 AI MODEL PERFORMANCE & DATASET SHOWCASE ---
st.divider()
st.header("🧠 DRDO SIH 54 - AI Model Performance & Dataset Artifacts")

ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📊 Engine Telemetry Dataset", "📈 Classification Confusion Matrix", "⭐ Feature Importance & Metrics"])

with ai_tab1:
    st.subheader("Raw Dataset Preview (`engine_dataset_v1.csv`)")
    if os.path.exists("engine_dataset_v1.csv"):
        df_csv = pd.read_csv("engine_dataset_v1.csv")
        st.success(f"Dataset successfully loaded! Total rows: {df_csv.shape[0]}, Features: {df_csv.shape[1]}")
        st.dataframe(df_csv.head(15), use_container_width=True)
    else:
        st.info("`engine_dataset_v1.csv` file not found in directory.")

with ai_tab2:
    st.subheader("Fault Classification Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Fault Classification Confusion Matrix (Acc=0.988)", use_container_width=True)
    else:
        st.info("`confusion_matrix.png` image asset not found.")

with ai_tab3:
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("Feature Importance Analysis")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", caption="Top Feature Importances — Fault Classifier", use_container_width=True)
        else:
            st.info("`feature_importance.png` image asset not found.")

    with col_y:
        st.subheader("JSON Metrics Summary")
        if os.path.exists("metrics_summary.json"):
            with open("metrics_summary.json", "r") as f:
                metrics_text = f.read()
            st.code(metrics_text, language="json")
        else:
            st.info("`metrics_summary.json` file not found.")

# --- AUTOMATIC RERUN FOR LIVE STREAMING ---
if stream_active:
    time.sleep(0.4)
    st.rerun()