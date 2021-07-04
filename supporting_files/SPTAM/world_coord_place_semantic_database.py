import numpy as np
import json
import sys

PLACE_SEMANTIC_DATABASE_FILE = "./place_semantic.json"
with open(PLACE_SEMANTIC_DATABASE_FILE, "r") as f:
    place_semantic_db = json.loads(f.read())
    f.close()

characteristics = {"Place_0" : "000000_rgb.png",
"Place_1" : "000014_rgb.png",
"Place_2" : "000018_rgb.png",
"Place_3": "000023_rgb.png",
"Place_4" : "000028_rgb.png",
"Place_5" : "000032_rgb.png",
"Place_6" : "000051_rgb.png",
"Place_7" : "000055_rgb.png",
"Place_8" : "000056_rgb.png",
"Place_9" : "000064_rgb.png",
"Place_10" : "000035_rgb.png",
"Place_11" : "000039_rgb.png",
"Place_12" : "000043_rgb.png",
"Place_13" : "000011_rgb.png"
}


def get_world_coords_from_place_semantic_database(corner, place, semantic_type, corner_name):
    corner_index = int(corner_name[-1])-1
    semantic_dicts = place_semantic_db[characteristics[place]]
    minimum_distance_image = sys.maxsize
    minimum_distance_world_frame = None
    final_object_name = None
    for object_dicts in semantic_dicts:
        if(len(object_dicts) == 0):
            return None
        object_name = list(object_dicts.keys())[0]
        if not (corner_index >=0 and corner_index < len(object_dicts[object_name])):
            continue
        # corner_index = -1

        # for idx,corner_list in enumerate(object_dicts[object_name]):
        #     if corner_name == corner_list[-1]:
        #         corner_index = idx
        #         break

        # if corner_index == -1:
        #     continue
        
        # corner_uv = object_dicts[object_name][corner_index][0]
        corner_world_coord = object_dicts[object_name][corner_index][1:]
        corner_world_coord = [float(i) for i in corner_world_coord]
        if object_name == semantic_type:
            # semantic_coord_image = np.array(corner_uv).reshape(2, 1)
            # distance_image = np.linalg.norm(corner - semantic_coord_image)
            # if distance_image < minimum_distance_image:
                # minimum_distance_image = distance_image
            minimum_distance_world_frame = np.array(corner_world_coord).reshape(3, 1)
            final_object_name = object_name
            break

    if(minimum_distance_world_frame is None):
        return None
    print(final_object_name, semantic_type)
    return minimum_distance_world_frame/100

