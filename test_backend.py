from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "UAV Engine Digital Twin Backend Online"}

def test_fault_injection_and_clear():
    inject_res = client.post("/fault/inject?fault_type=thermal_excursion")
    assert inject_res.status_code == 200
    assert inject_res.json()["fault_type"] == "thermal_excursion"

    clear_res = client.post("/fault/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["active"] is False

def test_websocket_stream():
    with client.websocket_connect("/ws/telemetry") as websocket:
        data = websocket.receive_json()
        assert "telemetry" in data
        assert "analytics" in data
        assert "egt" in data["telemetry"]
        assert "rul_percentage" in data["analytics"]
