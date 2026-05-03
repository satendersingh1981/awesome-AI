import json
import os
from datetime import datetime


def save_raw_data(data, prefix="jobs"):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"data/raw/{prefix}_{timestamp}.json"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename