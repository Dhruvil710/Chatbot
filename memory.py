import json
import os

FILE = "memory.json"

def load_memory():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r") as f:
        return json.load(f)

def save_memory(data):
    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_preference(key, value):
    data = load_memory()

    if "preferences" not in data:
        data["preferences"] = {}

    data["preferences"][key] = value

    save_memory(data)
