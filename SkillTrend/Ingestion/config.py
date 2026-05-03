import os
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
COUNTRY = os.getenv("COUNTRY", "in")

BASE_URL = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search"