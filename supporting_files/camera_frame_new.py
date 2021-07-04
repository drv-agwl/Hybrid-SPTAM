import copy
import glob
import json
import os

import cv2
import numpy as np
import glob
import json
from matplotlib import pyplot as plt

# images = glob.glob("./Oct_2020/sequence/images/*.png")
# images.sort()
# count = 0
# obj_det = []

# # FILE NAMES
# Hypothesis_no = 1

# obj_detect_coordinates = f"./Hypothesis_{Hypothesis_no}/annot_depths_hypoth{Hypothesis_no}.txt"
# output_file = f"./Hypothesis_{Hypothesis_no}/camera_frame_set_1_hypoth{Hypothesis_no}.txt"

# # with open(obj_detect_coordinates) as f:
# #     for line in f:
# #         obj_det.append(json.loads(line))

# # file1 = open(output_file, "w+")
# coord = {}

# print(len(images))

# corner_map = {1: "bottom_left", 2: "top_left", 3: "top_right", 4: "bottom_right"}
corner_map = {1: "Pos1", 2: "Pos2", 3: "Pos3", 4: "Pos4"}


def compute_coord_robot(imgLp, imgRp, j, corners):
    # print("Beohar output: ", j)

    id = j[0]
    name = j[1]

    imgL = cv2.imread(imgLp, 0)
    imgR = cv2.imread(imgRp, 0)

    kps_left = []
    kps_right = []
    stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    # plt.imshow(disparity,'gray')
    # plt.show()

    centreCol = 960.0
    centreRow = 540.0
    b = 1.20008
    f = 1399.56
    c = 605.254
    c1 = 898.46
    c2 = 875.82
    c = c1 - c2

    # extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32, use_orientation=False)

    x_arr = []
    y_arr = []
    z_arr = []
    image_coord = []

    corner_no = 1
    for corner in j[2:]:
        x, y = corner[0], corner[1]
        x = min(x, 1920)
        y = min(y, 1080)
        # print(j)
        # print(f"Corner:{x}, {y}")
        # print("Disparity: ", disparity[round(y), round(x)])

        x_coord, y_coord = int(x), int(y)

        # kps_left.append(cv2.KeyPoint(x_coord, y_coord, 3))
        # kps_right.append(cv2.KeyPoint(x_coord + disparity[y_coord, x_coord], y_coord, 3))

        # z = (b * f) / (disparity[y_coord, x_coord] + c)
        z = corners["Corner_"+str(corner_no)+"_depth"]
        # print(10*z)
        u = x_coord - centreCol
        v = -(y_coord - centreRow)
        x1 = u * z / f
        y1 = v * z / f
        x_arr.append(x1)
        y_arr.append(y1)
        z_arr.append(z)
        image_coord.append((x_coord, y_coord))
        corner_no += 1

    # kps_left, descs_left = extractor.compute(imgL, kps_left)
    # kps_right, descs_right = extractor.compute(imgR, kps_right)

    final_coord = []

    ret_kps_left, ret_kps_right, ret_desc_left, ret_desc_right = [], [], [], []

    # if len(kps_left) < 4 or len(kps_right) < 4:
    #     return None

    for i in range(len(z_arr)):
        # if 0 < z_arr[i] < 1:
        final_coord.append([corner_map[i + 1], x_arr[i], y_arr[i], z_arr[i]])
        # ret_kps_left.append(kps_left[i])
        # ret_kps_right.append(kps_right[i])
        # ret_desc_left.append(descs_left[i])
        # ret_desc_right.append(descs_right[i])

    if len(z_arr) < 3:
        print(f"ID:{id}")
        return

    coord["ID"] = id
    coord["Name"] = name
    coord["Mat"] = final_coord

    for i in range(len(image_coord)):
        coord[f"pos{i + 1}xy"] = image_coord[i]

    # print(coord)
    if not final_coord:
        # print("empty")
        return

    json_data = json.dumps(coord)
    file1.write(json_data + "\n")


# print("Z : " + str(z) + " X : " + str(x) + " Y :" + str(y))
# print(disparity[y_coord][x_coord])
# plt.imshow(disparity,'gray')
# plt.show()

# if __name__ == "__main__":
#     for index in range(len(images)):
#         print(images[index][27:])
#         for corners in obj_det:
#             # print(corners["ID"])
#             if corners["ID"] == images[index][27:]:
#                 # print(corners["ID"] == images[index][8:])
#                 name = corners["Name"]
#                 id = corners["ID"]
#                 x1, y1 = corners["Xmin"], corners["Ymin"]
#                 x2, y2 = corners["Xmax"], corners["Ymin"]
#                 x3, y3 = corners["Xmin"], corners["Ymax"]
#                 x4, y4 = corners["Xmax"], corners["Ymax"]
#                 left_image_path = images[index]
#                 right_image_path = "./Oct_2020/sequence/images_1/" + str(images[index][27:])
#                 corner_sequence = [id, name, [x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#                 compute_coord_robot(left_image_path, right_image_path, corner_sequence, corners)
