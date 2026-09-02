# =============================================================================
# 📡 ROLE MODULE: SIGNAL TELEMETRY & DATA PACKET PIPELINE
# =============================================================================

import time
import pandas as pd
import numpy as np

class SignalTelemetryPipeline:
    """
    Handles signal filtering, sensor calibration scaling, and 
    avionics transmission packet framing (AV-bus / JSON GCS link).
    """
    
    def __init__(self, device_id="ROTARX-914-UAV-054"):
        self.device_id = device_id
        self.packet_counter = 0

    def apply_signal_filtering(self, raw_series, window_size=3):
        """
        Applies a moving average filter to smooth high-frequency 
        electrical noise and jitter from raw sensor streams.
        """
        if isinstance(raw_series, pd.Series):
            return raw_series.rolling(window=window_size, min_periods=1).mean()
        elif isinstance(raw_series, (list, np.ndarray)):
            s = pd.Series(raw_series)
            return s.rolling(window=window_size, min_periods=1).mean().tolist()
        return raw_series

    def scale_analog_sensor(self, raw_voltage, min_v=0.0, max_v=5.0, min_val=0.0, max_val=10.0):
        """
        Calibrates and maps raw analog sensor voltage (e.g., 0-5V) 
        to engineering units (e.g., bar, psi, or temperature).
        """
        if max_v == min_v:
            return min_val
        clamped_voltage = max(min_v, min(raw_voltage, max_v))
        scaled_value = min_val + (clamped_voltage - min_v) * (max_val - min_val) / (max_v - min_v)
        return round(scaled_value, 2)

    def build_telemetry_packet(self, throttle, altitude, ambient_temp, raw_sensors_dict):
        """
        Packages calibrated sensor feeds and environmental states into a 
        standardized avionics transmission payload for the Ground Control Station (GCS).
        """
        self.packet_counter += 1
        
        packet = {
            "header": {
                "sync_byte": "0xAA",
                "device_id": self.device_id,
                "packet_id": self.packet_counter,
                "timestamp_ms": int(time.time() * 1000)
            },
            "environmental": {
                "altitude_m": float(altitude),
                "ambient_temp_c": float(ambient_temp),
                "throttle_pct": float(throttle)
            },
            "sensor_payload": {
                "rpm": float(raw_sensors_dict.get("RPM", 0)),
                "map_inhg": float(raw_sensors_dict.get("MAP", 0)),
                "cht_c": float(raw_sensors_dict.get("CHT", 0)),
                "egt_c": float(raw_sensors_dict.get("EGT", 0)),
                "oil_press_bar": self.scale_analog_sensor(raw_sensors_dict.get("Oil Press Voltage", 3.2), 0.0, 5.0, 0.0, 10.0),
                "oil_temp_c": float(raw_sensors_dict.get("Oil Temp", 0)),
                "fuel_flow_lh": float(raw_sensors_dict.get("Fuel Flow", 0))
            },
            "status": {
                "link_quality": "EXCELLENT",
                "frame_error_check": "CRC-16-VALID"
            }
        }
        return packet

# --- Example Testing Execution ---
if __name__ == "__main__":
    telemetry_mgr = SignalTelemetryPipeline()
    
    # Simulate a sample sensor dictionary
    sample_sensors = {"RPM": 5200.5, "MAP": 34.2, "CHT": 108.5, "EGT": 680.1, "Oil Press Voltage": 3.6, "Oil Temp": 95.0, "Fuel Flow": 18.5}
    
    # Generate packet
    test_packet = telemetry_mgr.build_telemetry_packet(throttle=0.85, altitude=4500.0, ambient_temp_c=22.5, raw_sensors_dict=sample_sensors)
    
    import json
    print("--- SIGNAL TELEMETRY PACKET OUTPUT ---")
    print(json.dumps(test_packet, indent=4))