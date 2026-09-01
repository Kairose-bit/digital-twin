import asyncio
import json
import math
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.ensemble import IsolationForest

app = FastAPI(title="UAV Digital Twin Telemetry API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fault_state = {
    "active": False,
    "fault_type": None
}

np.random.seed(42)
normal_egt = np.random.uniform(350, 700, 500)
normal_rpm = np.random.uniform(2000, 5500, 500)
normal_oil_press = np.random.uniform(35, 75, 500)
normal_map = np.random.uniform(15, 28, 500)

X_train = np.column_stack((normal_egt, normal_rpm, normal_oil_press, normal_map))
anomaly_model = IsolationForest(contamination=0.05, random_state=42)
anomaly_model.fit(X_train)


class TelemetrySimulator:
    def __init__(self):
        self.step = 0
        self.base_rul = 100.0

    def generate_metrics(self):
        self.step += 1
        t = self.step * 0.5

        egt = 550 + 40 * math.sin(t / 5) + random.uniform(-5, 5)
        rpm = 4500 + 300 * math.sin(t / 8) + random.uniform(-20, 20)
        oil_press = 55 + 5 * math.cos(t / 6) + random.uniform(-1, 1)
        map_press = 22 + 2 * math.sin(t / 4) + random.uniform(-0.5, 0.5)

        if fault_state["active"]:
            if fault_state["fault_type"] == "thermal_excursion":
                egt += random.uniform(220, 280)
                oil_press -= random.uniform(15, 25)
            elif fault_state["fault_type"] == "oil_leak":
                oil_press -= random.uniform(30, 40)
                egt += random.uniform(60, 100)
            elif fault_state["fault_type"] == "cylinder_misfire":
                rpm -= random.uniform(1200, 1800)
                egt += random.uniform(150, 200)

        feature_vector = np.array([[egt, rpm, oil_press, map_press]])
        prediction = anomaly_model.predict(feature_vector)[0]
        is_anomaly = bool(prediction == -1)

        if fault_state["active"]:
            self.base_rul = max(5.0, self.base_rul - random.uniform(0.8, 2.5))
        else:
            self.base_rul = max(10.0, self.base_rul - 0.01)

        contingency = "NOMINAL: Continue standard surveillance protocol."
        status_level = "GREEN"

        if egt > 780:
            status_level = "RED"
            contingency = "CRITICAL ALERT: Cylinder Thermal Excursion! Throttle back to 60% and initiate Return-To-Base (RTB)."
        elif oil_press < 25:
            status_level = "RED"
            contingency = "CRITICAL ALERT: Low Oil Pressure! Reduce engine load immediately. RTB advised."
        elif is_anomaly:
            status_level = "YELLOW"
            contingency = "WARNING: Unstable telemetry anomaly detected. Monitor EGT and vibration metrics."

        return {
            "timestamp": round(time.time(), 2),
            "telemetry": {
                "egt": round(egt, 2),
                "rpm": round(rpm, 2),
                "oil_pressure": round(oil_press, 2),
                "map": round(map_press, 2)
            },
            "analytics": {
                "rul_percentage": round(self.base_rul, 2),
                "is_anomaly": is_anomaly,
                "status_level": status_level,
                "contingency_recommendation": contingency,
                "active_fault": fault_state["fault_type"] if fault_state["active"] else "NONE"
            }
        }


simulator = TelemetrySimulator()


@app.get("/")
def read_root():
    return {"status": "UAV Engine Digital Twin Backend Online"}


@app.post("/fault/inject")
def inject_fault(fault_type: str = "thermal_excursion"):
    fault_state["active"] = True
    fault_state["fault_type"] = fault_type
    return {"status": "Fault Injected", "fault_type": fault_type}


@app.post("/fault/clear")
def clear_fault():
    fault_state["active"] = False
    fault_state["fault_type"] = None
    return {"status": "Fault Cleared", "active": False}


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = simulator.generate_metrics()
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket stream.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
