import requests

API = "http://127.0.0.1:8000"


def run_pipeline(video_path):

    response = requests.post(
        f"{API}/run",
        json={
            "video": video_path
        }
    )

    return response.json()


def get_status():

    response = requests.get(f"{API}/status")

    return response.json()


def get_prediction():

    response = requests.get(f"{API}/prediction")

    return response.json()