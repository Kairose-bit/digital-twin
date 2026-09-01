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