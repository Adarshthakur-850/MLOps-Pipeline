import requests
import json
import sys

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # 1. Health Check
    try:
        resp = requests.get(f"{base_url}/health")
        print(f"Health Response: {resp.json()}")
        if not resp.json().get("model_loaded"):
            print("ERROR: Model not loaded in API.")
            sys.exit(1)
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(1)

    # 2. Prediction
    payload = {
        "age": 0.038,
        "sex": 0.050,
        "bmi": 0.061,
        "bp": 0.021,
        "s1": -0.044,
        "s2": -0.034,
        "s3": -0.043,
        "s4": -0.002,
        "s5": 0.019,
        "s6": -0.017
    }
    
    try:
        resp = requests.post(f"{base_url}/predict", json=payload)
        if resp.status_code == 200:
            print(f"Prediction: {resp.json()}")
        else:
            print(f"Prediction failed ({resp.status_code}): {resp.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Prediction request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
