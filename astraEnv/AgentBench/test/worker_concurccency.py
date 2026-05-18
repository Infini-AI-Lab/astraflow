import requests
import json

url = "http://localhost:5000/api/episode/start"
headers = {"Content-Type": "application/json"}
payload = {"sample_id": 0, "config": {}}

for i in range(64):
    try:
        response = requests.post(url, json=payload, headers=headers)
        # print(f"Request {i+1}: Status {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request {i+1} Failed: {e}")