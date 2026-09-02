import subprocess
import time
import sys

def launch_digital_twin():
    print("🚀 Launching UAV Digital Twin System [SIH26054]...\n")
    
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd="./digital-twin-backend"
    )
    print("✅ Backend Stream Server initialized at ws://localhost:8000/ws/telemetry")
    
    time.sleep(2)
    
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="./digital-twin-frontend",
        shell=True
    )
    print("✅ Frontend 3D Canvas dashboard online at http://localhost:5173")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down backend & frontend streams gracefully...")
        backend.terminate()
        frontend.terminate()

if __name__ == "__main__":
    launch_digital_twin()
