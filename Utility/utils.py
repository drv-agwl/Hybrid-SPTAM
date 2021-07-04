import json
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def convert_to_one_hot(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b
