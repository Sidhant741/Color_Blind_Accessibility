import json
import os

DATA_PATH = "feedback_dataset.json"


def load_dataset():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def save_sample(sample):
    data = load_dataset()
    data.append(sample)

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
def load_rewards():
    data = load_dataset()
    return [sample["reward"] for sample in data]