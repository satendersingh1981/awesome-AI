import requests
from .config import BASE_URL, ADZUNA_APP_ID, ADZUNA_APP_KEY


def fetch_jobs_from_adzuna(query: str, location: str = "India", page: int = 1):
    url = f"{BASE_URL}/{page}"

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": 50,
        "content-type": "application/json"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    return response.json()