from __future__ import annotations

import json
import sys
import time
from urllib.request import urlopen


def get_json(url: str):
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


for attempt in range(30):
    try:
        health = get_json("http://localhost:8000/health")
        print(json.dumps(health, indent=2))
        if health.get("status") == "ok":
            print("Backend verification passed.")
            sys.exit(0)
    except Exception as exc:
        print(f"Attempt {attempt + 1}: {exc}")
    time.sleep(2)
raise SystemExit("Backend did not become healthy")
