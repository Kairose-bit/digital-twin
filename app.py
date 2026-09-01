<<<<<<< HEAD
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

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
    "CHT_MAX": 135.0,        # °C Max (Coolant/Head Temp)
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
    """
    Simulates a sequence-based Neural Network processing windows of 9 operational values:
    [RPM, CHT, EGT, Oil Press, Oil Temp, Fuel Flow, Throttle, Altitude, Ambient Temp]
    """
    @staticmethod
    def evaluate_window(df_window):
        if len(df_window) < window_size:
            return 0.0, 100.0, "Synchronizing Time-Series Data..."
        
        latest = df_window.iloc[-1]
        anomaly_weight = 0.0
        reasons = []
        
        # Real ML networks flag anomalies based on out-of-distribution vectors.
        # Below represents the logic boundaries the ML maps into reconstruction errors:
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
            
        # Background statistical variance
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
    
    # Thermodynamic & Aerodynamic scaling factors based on Rotax physics engine
    throttle_factor = throttle / 100.0
    alt_density_ratio = max(0.45, 1.0 - (altitude / 45000.0))
    
    # Normal physics mapping curves
    if throttle <= 100.0:
        target_rpm = 1800 + (ROTAX_LIMITS["RPM_MAX_CONTINUOUS"] - 1800) * throttle_factor
        target_map = 25.0 + (ROTAX_LIMITS["MAP_MAX_CONT"] - 25.0) * throttle_factor
    else:
        # Turbo boost envelope up to 115% throttle command
        over_load = (throttle - 100.0) / 15.0
        target_rpm = ROTAX_LIMITS["RPM_MAX_CONTINUOUS"] + (ROTAX_LIMITS["RPM_MAX_TAKEOFF"] - ROTAX_LIMITS["RPM_MAX_CONTINUOUS"]) * over_load
        target_map = ROTAX_LIMITS["MAP_MAX_CONT"] + (ROTAX_LIMITS["MAP_MAX_TO"] - ROTAX_LIMITS["MAP_MAX_CONT"]) * over_load
        
    base_rpm = np.random.normal(target_rpm, 10)
    base_map = np.random.normal(target_map, 0.15)
    
    # Multi-sensor dependency propagation
    base_cht = np.random.normal(85 + (25 * throttle_factor) + (ambient_temp * 0.5), 0.8)
    base_egt = np.random.normal(640 + (140 * throttle_factor), 3.5)
    base_oil_p = np.random.normal(4.2 - (0.5 * (base_rpm / 5800.0)), 0.05) if base_rpm > 3500 else np.random.normal(2.5, 0.1)
    base_oil_t = np.random.normal(80 + (22 * throttle_factor), 0.4)
    base_fuel = np.random.normal(8.0 + (18.0 * throttle_factor * (base_map / 30.0)), 0.2)

    # Scenario Injection Algorithms 
    if flight_scenario == "Turbo Charger Wastegate Stuck (Overboost Anomaly)":
        if throttle > 80.0:
            base_map += np.random.uniform(5.5, 8.5) # Wastegate fails closed forcing excessive boost
            base_egt += 60.0
            base_cht += 12.0
    elif flight_scenario == "Cylinder Head Coolant Loss (High CHT)":
        base_cht += np.random.uniform(35.0, 58.0) # Thermal divergence
        base_oil_t += 18.0
    elif flight_scenario == "Oil Pressure Loss / Bearing Degradation":
        base_oil_p = np.random.uniform(0.8, 1.4) # Violates 2.0 bar limit
        base_oil_t += 22.0
    elif flight_scenario == "Fuel Pump Restrictive Flow (Lean Spikes)":
        base_fuel -= np.random.uniform(6.0, 9.0)
        base_egt += np.random.uniform(110.0, 160.0) # Dangerous lean EGT spikes
        base_cht += 14.0
        
    # Build complete sensor structure matrix payload
    new_row = {
        "Time": new_time,
        "RPM": base_rpm,
        "CHT": base_cht,
        "EGT": base_egt,
        "Oil Press": base_oil_p,
        "Oil Temp": base_oil_t,
        "Fuel Flow": base_fuel,
        "MAP": base_map,
        "Throttle": throttle,
        "Altitude": altitude,
        "Ambient Temp": ambient_temp,
        "Anomaly Score": 0.0, 
        "Health Index": 100.0 
    }
    
    st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history.iloc[1:].reset_index(drop=True)

    # Pass the time series data frame block to ML Inference
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

# Row 1: KPI Telemetry Blocks
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
m_col1.metric("Engine RPM", f"{int(latest_data['RPM'])} rpm")
m_col2.metric("Manifold Pressure (MAP)", f"{latest_data['MAP']:.2f} inHg")
m_col3.metric("Coolant CHT", f"{latest_data['CHT']:.1f} °C")
m_col4.metric("Exhaust Gas EGT", f"{latest_data['EGT']:.1f} °C")
m_col5.metric("Oil Parameters", f"{latest_data['Oil Press']:.2f} bar / {latest_data['Oil Temp']:.1f}°C")
m_col6.metric("Fuel Delivery", f"{latest_data['Fuel Flow']:.1f} L/h")

# Status Warning Banner System
if "🚨" in status_text:
    st.error(f"**Digital Twin Automated Security Assessment:** {status_text}")

# --- AUTOMATIC RERUN FOR LIVE TELEMETRY STREAMING ---
if stream_active:
    time.sleep(0.5)
    st.rerun()
=======
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="UAV Aero Piston Engine Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STATIC ENGINE CONFIGURATION (DIGITAL TWIN SPECIFICATIONS) ---
ENGINE_SPECS = {
    "Engine Type": "Unmanned Aero Piston Engine (4-Stroke, EFI)",
    "Number of Cylinders": 4,
    "Displacement": "500 cc",
    "Rated Power": "55 hp",
    "Rated RPM": 5800,
    "Compression Ratio": "10.5:1",
    "Fuel Type": "AVGAS 100LL / Premium Auto Fuel"
}

# Thresholds for Anomaly Detection
THRESHOLDS = {
    "EGT_MAX": 850,       # °C
    "OIL_PRESS_MIN": 2.0, # bar
    "OIL_PRESS_MAX": 5.0, # bar
    "MAP_MAX": 45.0       # inHg (Boost)
}

# --- SIDEBAR: ENGINE SPECS & CONTROLS ---
st.sidebar.title("🛠️ UAV Engine Specifications")
for key, val in ENGINE_SPECS.items():
    st.sidebar.text_value = f"**{key}:** {val}"
    st.sidebar.markdown(f"- **{key}**: {val}")

st.sidebar.divider()
st.sidebar.title("✈️ Flight Control Simulation")
simulation_mode = st.sidebar.selectbox("Flight Phase / Scenario", ["Normal Cruise", "High-Power Climb", "Exhaust Leak (Anomaly)", "Oil Pressure Drop (Anomaly)"])
run_simulation = st.sidebar.toggle("Start Live Telemetry Stream", value=True)

# --- MAIN DASHBOARD INTERFACE ---
st.title("🤖 AI-Enabled Real-Time UAV Aero Piston Engine Digital Twin")
st.subheader("Time-Series Anomaly Detection & Health Predictive Analytics")

# Top Level Metrics Placeholders
metric_row = st.columns(4)
egt_metric = metric_row[0].empty()
oil_metric = metric_row[1].empty()
map_metric = metric_row[2].empty()
health_metric = metric_row[3].empty()

st.divider()

# Charts Placeholders
chart_row_1 = st.columns(2)
chart_egt = chart_row_1[0].empty()
chart_oil = chart_row_1[1].empty()

chart_row_2 = st.columns(2)
chart_map = chart_row_2[0].empty()
chart_health = chart_row_2[1].empty()

# --- SIMULATED DATA GENERATION & ML PREDICTION ---
# Initializing historical data arrays for real-time plotting
if 'time_steps' not in st.session_state:
    st.session_state.time_steps = list(range(50))
    st.session_state.egt_hist = list(np.random.normal(720, 10, 50))
    st.session_state.oil_hist = list(np.random.normal(3.8, 0.2, 50))
    st.session_state.map_hist = list(np.random.normal(29.9, 0.5, 50))
    st.session_state.health_hist = list(np.linspace(100, 98, 50))

# Live loop
while run_simulation:
    # 1. Update Time Step
    next_time = st.session_state.time_steps[-1] + 1
    st.session_state.time_steps.append(next_time)
    st.session_state.time_steps.pop(0)

    # 2. Simulate Physics & Fault Injector based on UI choices
    base_rpm = ENGINE_SPECS["Rated RPM"] if "Climb" in simulation_mode else 4500
    
    if simulation_mode == "Normal Cruise":
        current_egt = np.random.normal(730, 5)
        current_oil = np.random.normal(4.0, 0.1)
        current_map = np.random.normal(30.0, 0.3)
        base_health = 99.5
    elif simulation_mode == "High-Power Climb":
        current_egt = np.random.normal(810, 8)
        current_oil = np.random.normal(3.5, 0.15)
        current_map = np.random.normal(40.0, 0.6)
        base_health = 98.2
    elif simulation_mode == "Exhaust Leak (Anomaly)":
        # Multi-variable anomaly: EGT spikes aggressively, MAP drops slightly
        current_egt = st.session_state.egt_hist[-1] + np.random.uniform(5, 15)
        current_oil = np.random.normal(3.9, 0.1)
        current_map = np.random.normal(27.0, 0.5)
        base_health = max(40.0, st.session_state.health_hist[-1] - 1.5)
    elif simulation_mode == "Oil Pressure Drop (Anomaly)":
        # Structural anomaly: Rapid loss of oil pressure
        current_egt = np.random.normal(760, 5)
        current_oil = max(0.5, st.session_state.oil_hist[-1] - np.random.uniform(0.1, 0.3))
        current_map = np.random.normal(30.0, 0.3)
        base_health = max(20.0, st.session_state.health_hist[-1] - 2.5)

    # Update historical arrays
    st.session_state.egt_hist.append(current_egt)
    st.session_state.egt_hist.pop(0)
    st.session_state.oil_hist.append(current_oil)
    st.session_state.oil_hist.pop(0)
    st.session_state.map_hist.append(current_map)
    st.session_state.map_hist.pop(0)

    # 3. AI / Rule-Based Real-Time Inference
    # In production, swap this out with: internal_model.predict(window_data)
    is_anomaly = False
    status_msg = "HEALTHY"
    
    if current_egt > THRESHOLDS["EGT_MAX"] or current_oil < THRESHOLDS["OIL_PRESS_MIN"] or current_oil > THRESHOLDS["OIL_PRESS_MAX"]:
        is_anomaly = True
        status_msg = "CRITICAL FAULT DETECTED"
    
    # Calculate health index proxy based on variance and thresholds
    calculated_health = base_health if not is_anomaly else base_health * 0.6
    st.session_state.health_hist.append(calculated_health)
    st.session_state.health_hist.pop(0)

    # 4. Render Metrics UI
    egt_metric.metric("Exhaust Gas Temp (EGT)", f"{current_egt:.1f} °C", delta=f"{current_egt-730:.1f} °C vs Cruise", delta_color="inverse" if current_egt > 800 else "normal")
    oil_metric.metric("Oil Pressure", f"{current_oil:.2f} bar", delta=f"{current_oil-4.0:.2f} bar vs Cruise", delta_color="normal" if current_oil >= 2.0 else "inverse")
    map_metric.metric("Manifold Pressure (MAP)", f"{current_map:.1f} inHg", delta=f"{current_map-30.0:.1f} inHg")
    
    health_color = "normal" if calculated_health > 80 else "inverse"
    health_metric.metric("Engine Health Index", f"{calculated_health:.1f}%", delta=status_msg, delta_color=health_color)

    # 5. Render Time-Series Plots
    def create_plot(time_data, val_data, name, unit, color, threshold_line=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_data, y=val_data, mode='lines+markers', name=name, line=dict(color=color)))
        if threshold_line:
            fig.add_hline(y=threshold_line, line_dash="dash", line_color="red", annotation_text="Limit Threshold")
        fig.update_layout(title=f"Real-Time {name} ({unit})", xaxis_title="Telemetry Tick", yaxis_title=unit, margin=dict(l=20, r=20, t=40, b=20), height=300)
        return fig

    chart_egt.plotly_chart(create_plot(st.session_state.time_steps, st.session_state.egt_hist, "EGT", "°C", "#FF4B4B", THRESHOLDS["EGT_MAX"]), use_container_width=True)
    chart_oil.plotly_chart(create_plot(st.session_state.time_steps, st.session_state.oil_hist, "Oil Pressure", "bar", "#00CC96", THRESHOLDS["OIL_PRESS_MIN"]), use_container_width=True)
    chart_map.plotly_chart(create_plot(st.session_state.time_steps, st.session_state.map_hist, "MAP / Boost", "inHg", "#AB63FA", THRESHOLDS["MAP_MAX"]), use_container_width=True)
    
    # Engine Twin Health History Chart
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=st.session_state.time_steps, y=st.session_state.health_hist, fill='tozeroy', name="Health Score", line=dict(color='#17BECF')))
    fig_h.update_layout(title="Digital Twin Predicted Health Index Trend", xaxis_title="Telemetry Tick", yaxis_title="Health %", yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=40, b=20), height=300)
    chart_health.plotly_chart(fig_h, use_container_width=True)

    # Refresh pacing for real-time behavior (0.4 seconds)
    time.sleep(0.4)
>>>>>>> e90a731b3e8ccd68f2efbf6dfb8d57e28f775709
