import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import os

# --- OPTIONAL TELEMETRY MODULE IMPORT ---
try:
    from telemetry import SignalTelemetryPipeline
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NEXUS | Rotax 914 UAV Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR TARGET "NEXUS" DARK DASHBOARD UI ---
st.markdown("""
<style>
    /* Main Background and Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Custom Card Component Styling */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        line-height: 1.2;
    }
    .badge-pill {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
        margin-top: 6px;
    }
    .badge-green {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }
    .badge-red {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
    }
    
    /* Status Banner Styling */
    .status-banner-ok {
        background-color: rgba(46, 160, 67, 0.1);
        border: 1px solid rgba(46, 160, 67, 0.4);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 20px;
    }
    .status-banner-fault {
        background-color: rgba(248, 81, 73, 0.1);
        border: 1px solid rgba(248, 81, 73, 0.4);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 20px;
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADERS ---
@st.cache_data
def load_engine_dataset():
    if os.path.exists("engine_dataset_v1.csv"):
        return pd.read_csv("engine_dataset_v1.csv")
    return None

# --- ENGINE SPECIFICATIONS & LIMITS ---
ENGINE_SPECS = {
    "Engine Model": "BRP-Rotax 914 UL / F",
    "Type": "4-Cylinder Boxer Turbocharged",
    "Displacement": "1,211.2 cc",
    "Max Power": "115 HP @ 5800 RPM",
    "Continuous": "100 HP @ 5500 RPM",
    "Fuel Type": "AVGAS 100LL / MOGAS RON 95"
}

ROTAX_LIMITS = {
    "RPM_MAX_TAKEOFF": 5800,
    "RPM_MAX_CONTINUOUS": 5500,
    "CHT_MAX": 135.0,      # °C
    "OIL_TEMP_MIN": 50.0,  # °C
    "OIL_TEMP_MAX": 130.0, # °C
    "OIL_PRESS_MIN": 2.0,  # bar
    "OIL_PRESS_MAX": 5.0,  # bar
    "MAP_MAX_CONT": 35.4,  # inHg
    "MAP_MAX_TO": 39.9,    # inHg
    "EGT_MAX": 880.0       # °C
}

# --- SIDEBAR: NEXUS CONTROLS ---
st.sidebar.markdown("### 🛠️ **NEXUS**")
st.sidebar.caption("Rotax 914 Digital Twin - Prototype")
st.sidebar.divider()

st.sidebar.markdown("**Flight Controls**")
throttle = st.sidebar.slider("Throttle / TCU (%)", 0.0, 115.0, 90.0, step=5.0)
altitude = st.sidebar.slider("Altitude (m)", 0, 6000, 1000, step=100)
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", -30, 45, 25, step=1)

st.sidebar.divider()
st.sidebar.markdown("**Scenario Injection**")
flight_scenario = st.sidebar.selectbox(
    "Scenario",
    [
        "Normal Flight Profile",
        "Turbo Charger Wastegate Stuck (Overboost Anomaly)",
        "Cylinder Head Coolant Loss (High CHT)",
        "Oil Pressure Loss / Bearing Degradation",
        "Fuel Pump Restrictive Flow (Lean Spikes)"
    ]
)

stream_active = st.sidebar.toggle("Live Telemetry Stream", value=True)
window_size = 15

st.sidebar.divider()
st.sidebar.markdown("**Engine Specifications**")
for k, v in ENGINE_SPECS.items():
    st.sidebar.caption(f"**{k}:** {v}")

# --- ML INFERENCE ENGINE ---
class Rotax914MLInference:
    @staticmethod
    def evaluate_window(df_window):
        if len(df_window) < window_size:
            return 0.0, 100.0, "Synchronizing Telemetry..."

        latest = df_window.iloc[-1]
        anomaly_weight = 0.0
        reasons = []

        if latest['CHT'] > ROTAX_LIMITS["CHT_MAX"]:
            anomaly_weight += (latest['CHT'] - ROTAX_LIMITS["CHT_MAX"]) * 2.0
            reasons.append("CHT Thermal Limit Exceeded")
        if latest['MAP'] > ROTAX_LIMITS["MAP_MAX_TO"] + 0.5:
            anomaly_weight += 50.0
            reasons.append("TCU Overboost Condition")
        if latest['Oil Press'] < ROTAX_LIMITS["OIL_PRESS_MIN"] and latest['RPM'] > 3500:
            anomaly_weight += 65.0
            reasons.append("Critical Low Oil Pressure")
        if latest['EGT'] > ROTAX_LIMITS["EGT_MAX"]:
            anomaly_weight += 40.0
            reasons.append("Extreme EGT Thermal Peak")

        anomaly_weight += np.random.uniform(0.05, 0.85)
        health_index = max(0.0, min(100.0, 100.0 - (anomaly_weight * 1.3)))

        if anomaly_weight > 30.0:
            status = "CRITICAL FAULT DETECTED: " + " & ".join(reasons)
        elif anomaly_weight > 12.0:
            status = "DEGRADED PERFORMANCE STATE"
        else:
            status = "NOMINAL FLIGHT RUN"

        return anomaly_weight, health_index, status

# --- STATE MANAGEMENT ---
if 'history' not in st.session_state:
    t_init = list(range(40))
    init_data = {
        "Time": t_init,
        "RPM": list(np.random.normal(5150, 20, 40)),
        "CHT": list(np.random.normal(123, 1, 40)),
        "EGT": list(np.random.normal(772, 3, 40)),
        "Oil Press": list(np.random.normal(3.84, 0.03, 40)),
        "Oil Temp": list(np.random.normal(98, 0.5, 40)),
        "Fuel Flow": list(np.random.normal(22.0, 0.2, 40)),
        "MAP": list(np.random.normal(34.0, 0.2, 40)),
        "Throttle": [90.0] * 40,
        "Altitude": [1000] * 40,
        "Ambient Temp": [25] * 40,
        "Anomaly Score": [0.3] * 40,
        "Health Index": [100.0] * 40
    }
    st.session_state.history = pd.DataFrame(init_data)
    st.session_state.counter = 40

# --- SIMULATION LOOP ---
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

    base_rpm = np.random.normal(target_rpm, 12)
    base_map = np.random.normal(target_map, 0.15)
    base_cht = np.random.normal(95 + (25 * throttle_factor) + (ambient_temp * 0.4), 0.8)
    base_egt = np.random.normal(640 + (140 * throttle_factor), 3.5)
    base_oil_p = np.random.normal(4.2 - (0.5 * (base_rpm / 5800.0)), 0.05) if base_rpm > 3500 else np.random.normal(2.5, 0.1)
    base_oil_t = np.random.normal(80 + (22 * throttle_factor), 0.4)
    base_fuel = np.random.normal(8.0 + (18.0 * throttle_factor * (base_map / 30.0)), 0.2)

    # Inject Scenarios
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
    if len(st.session_state.history) > 40:
        st.session_state.history = st.session_state.history.iloc[1:].reset_index(drop=True)

    window_df = st.session_state.history.tail(window_size)
    anomaly_score, health_index, status_text = Rotax914MLInference.evaluate_window(window_df)

    st.session_state.history.iloc[-1, st.session_state.history.columns.get_loc("Anomaly Score")] = anomaly_score
    st.session_state.history.iloc[-1, st.session_state.history.columns.get_loc("Health Index")] = health_index
else:
    status_text = "TELEMETRY PAUSED"
    anomaly_score = st.session_state.history["Anomaly Score"].iloc[-1]
    health_index = st.session_state.history["Health Index"].iloc[-1]

latest = st.session_state.history.iloc[-1]

# --- UI HEADER (TARGET DESIGN) ---
st.markdown("""
<div style="margin-bottom: 12px;">
    <div style="color: #3fb950; font-size: 11px; font-weight: 700; letter-spacing: 1px;">NEXUS • SYSTEM 26054+</div>
    <h1 style="color: #ffffff; margin: 2px 0 0 0; font-size: 32px; font-weight: 800; display: flex; align-items: center; gap: 8px;">
        <span style="color: #f85149;">❌</span> Rotax 914 UAV Digital Twin
    </h1>
    <div style="color: #8b949e; font-size: 13px; margin-top: 4px;">Real-time telemetry • operational limits • predictive health prototype</div>
</div>
""", unsafe_allow_html=True)

# --- STATUS BANNER (TARGET DESIGN) ---
is_fault = "CRITICAL" in status_text or "FAULT" in status_text
banner_class = "status-banner-fault" if is_fault else "status-banner-ok"
status_icon = "🚨" if is_fault else "●"
status_color = "#f85149" if is_fault else "#3fb950"

st.markdown(f"""
<div class="{banner_class}">
    <div style="color: {status_color}; font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;">
        <span>{status_icon}</span> {status_text}
    </div>
    <div style="color: #8b949e; font-size: 12px; margin-top: 3px;">
        {'Operational anomaly active - immediate inspection required' if is_fault else 'All monitored prototype parameters within configured limits'}
    </div>
</div>
""", unsafe_allow_html=True)

# --- METRIC CARDS ROW (TARGET UI MATCH) ---
def render_metric_card(title, value_str, limit_str, is_danger=False):
    badge_class = "badge-red" if is_danger else "badge-green"
    arrow = "↑" if not is_danger else "⚠️"
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value_str}</div>
        <div class="badge-pill {badge_class}">{arrow} {limit_str}</div>
    </div>
    """

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(render_metric_card("ENGINE RPM", f"{int(latest['RPM'])}", "Limit 5800", latest['RPM'] > 5800), unsafe_allow_html=True)
with c2:
    st.markdown(render_metric_card("MAP", f"{latest['MAP']:.2f} inHg", "Limit 39.9", latest['MAP'] > 39.9), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric_card("CHT", f"{latest['CHT']:.1f} °C", "Limit 135", latest['CHT'] > 135.0), unsafe_allow_html=True)
with c4:
    st.markdown(render_metric_card("EGT", f"{latest['EGT']:.1f} °C", "Limit 880", latest['EGT'] > 880.0), unsafe_allow_html=True)
with c5:
    st.markdown(render_metric_card("OIL PRESS", f"{latest['Oil Press']:.2f} bar", "Min 2.0", latest['Oil Press'] < 2.0), unsafe_allow_html=True)
with c6:
    st.markdown(render_metric_card("HEALTH", f"{health_index:.1f}%", "NOMINAL" if health_index > 80 else "DEGRADED", health_index <= 80), unsafe_allow_html=True)

st.markdown("<h3 style='color: #ffffff; margin-top: 15px; font-weight: 700;'>Live telemetry</h3>", unsafe_allow_html=True)

# --- TELEMETRY PLOTS (DARK PLOTLY THEME MATCHING TARGET UI) ---
time_ticks = st.session_state.history["Time"].tolist()

def create_target_plot(y_data, title, unit, color, limit_val=None):
    fig = go.Figure()
    
    # Main signal line
    fig.add_trace(go.Scatter(
        x=time_ticks, y=y_data,
        mode='lines',
        line=dict(color=color, width=2),
        name=title
    ))
    
    # Horizontal Limit line
    if limit_val is not None:
        fig.add_hline(
            y=limit_val, line_dash="dash", line_color="#ef4444", line_width=1.5,
            annotation_text="LIMIT", annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=10)
        )
        
    fig.update_layout(
        title=dict(text=title, font=dict(color='#ffffff', size=14)),
        paper_bgcolor='rgba(22, 27, 34, 1.0)',
        plot_bgcolor='rgba(22, 27, 34, 1.0)',
        margin=dict(l=30, r=20, t=35, b=25),
        height=220,
        xaxis=dict(showgrid=True, gridcolor='#21262d', tickfont=dict(color='#8b949e')),
        yaxis=dict(showgrid=True, gridcolor='#21262d', tickfont=dict(color='#8b949e'), title=unit, title_font=dict(color='#8b949e', size=11))
    )
    return fig

# Plot Grid (Matches 2-column layout in Image 2)
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.plotly_chart(create_target_plot(st.session_state.history["RPM"].tolist(), "Engine RPM", "rpm", "#58a6ff", ROTAX_LIMITS["RPM_MAX_TAKEOFF"]), use_container_width=True)
    st.plotly_chart(create_target_plot(st.session_state.history["EGT"].tolist(), "Exhaust Gas Temperature (EGT)", "°C", "#f0883e", ROTAX_LIMITS["EGT_MAX"]), use_container_width=True)
    st.plotly_chart(create_target_plot(st.session_state.history["MAP"].tolist(), "Manifold Pressure (MAP)", "inHg", "#bc8cff", ROTAX_LIMITS["MAP_MAX_TO"]), use_container_width=True)

with p_col2:
    st.plotly_chart(create_target_plot(st.session_state.history["CHT"].tolist(), "Cylinder Head Temperature (CHT)", "°C", "#f85149", ROTAX_LIMITS["CHT_MAX"]), use_container_width=True)
    st.plotly_chart(create_target_plot(st.session_state.history["Oil Press"].tolist(), "Oil Pressure", "bar", "#3fb950", ROTAX_LIMITS["OIL_PRESS_MIN"]), use_container_width=True)
    
    # Health Index Area Chart
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=time_ticks, y=st.session_state.history["Health Index"].tolist(),
        fill='tozeroy', line=dict(color='#39c5cf', width=2), name="Health"
    ))
    fig_h.update_layout(
        title=dict(text="Engine Health Index Trend", font=dict(color='#ffffff', size=14)),
        paper_bgcolor='rgba(22, 27, 34, 1.0)',
        plot_bgcolor='rgba(22, 27, 34, 1.0)',
        margin=dict(l=30, r=20, t=35, b=25),
        height=220,
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor='#21262d', tickfont=dict(color='#8b949e')),
        xaxis=dict(showgrid=True, gridcolor='#21262d', tickfont=dict(color='#8b949e'))
    )
    st.plotly_chart(fig_h, use_container_width=True)

# --- RETAINED SYSTEM & DATASET ARTIFACTS SECTION ---
st.divider()

st.markdown("<h3 style='color: #ffffff;'>⚡ Hardware Telemetry & Model Analytics</h3>", unsafe_allow_html=True)

if HAS_TELEMETRY:
    telemetry_processor = SignalTelemetryPipeline(device_id="ROTARX-914-UAV-054")
    raw_sensor_feed = {
        "RPM": float(latest['RPM']), "MAP": float(latest['MAP']),
        "CHT": float(latest['CHT']), "EGT": float(latest['EGT']),
        "Oil Press Voltage": float(latest['Oil Press']) * 0.8,
        "Oil Temp": float(latest['Oil Temp']), "Fuel Flow": float(latest['Fuel Flow'])
    }
    live_packet = telemetry_processor.build_telemetry_packet(
        throttle=throttle, altitude=altitude, ambient_temp=ambient_temp, raw_sensors_dict=raw_sensor_feed
    )
    t_col1, t_col2 = st.columns([1, 1])
    with t_col1:
        st.info("**Bus Status:** 13.8V Nominal | **Dual ECU:** Active | **Signal:** CRC-16 Verified")
    with t_col2:
        with st.expander("View Live Avionics Packet (JSON)"):
            st.json(live_packet)

ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📊 Engine Dataset", "📈 Confusion Matrix", "⭐ Feature Importances"])

with ai_tab1:
    df_csv = load_engine_dataset()
    if df_csv is not None:
        st.dataframe(df_csv.head(10), use_container_width=True)
    else:
        st.caption("`engine_dataset_v1.csv` not loaded.")

with ai_tab2:
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", use_container_width=True)

with ai_tab3:
    if os.path.exists("feature_importance.png"):
        st.image("feature_importance.png", use_container_width=True)

# --- REFRESH PACING FOR LIVE TELEMETRY STREAM ---
if stream_active:
    time.sleep(1.2)
    st.rerun()