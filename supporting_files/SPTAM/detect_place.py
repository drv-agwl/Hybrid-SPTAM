import os
import json

with open("./place_image.json","r") as f:
  place_image = json.loads(f.read())
  f.close()

def get_place(path):
	image_name = os.path.split(path)[-1]

	return place_image[image_name]


