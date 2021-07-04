import os as os
import json

with open('./preds_objs.json', 'r') as f:
    obj_data = json.load(f)

with open('./preds_alphabet.json', 'r') as f:
    alphabet_data = json.load(f)

for img, bbox in obj_data.items():
    bbox += alphabet_data[img]

with open('./1100.json', 'w') as f:
    json.dump(obj_data, f)