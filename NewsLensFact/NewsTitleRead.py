
import requests
import os
from dotenv import load_dotenv
load_dotenv()
NEWS_ORG_API_KEY = os.getenv("NEWS_ORG_API_KEY")
url = "https://newsapi.org/v2/top-headlines"
params = {
    "country": "us",
    "apiKey": NEWS_ORG_API_KEY  # Get free key from https://newsapi.org
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise error for bad status codes
    data = response.json()
    #print(data)
    for article in data['articles'][0:2]:
        print(article['title'])
        print(article['source']['name'])
except requests.exceptions.JSONDecodeError:
    print("Error: Response is not valid JSON")
    print(f"Response text: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")