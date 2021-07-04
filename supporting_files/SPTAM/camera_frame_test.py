import numpy as np
import cv2
import os
import json
import glob
import copy
from config import dataset_dir
# from stereo2depth_mod2 import find_kps_depths
# from matcher import find_corner_depths


# from tracker import find_all_corners_from_bboxs


def compute_coord_robot(imgLp, imgRp, j_left,j_right, hypotheses_no=1):
	# print("Beohar output: ", j_left)

	# global depths
	name = j_left[0]

	x1_left, y1_left = j_left[1][0], j_left[1][1]
	x2_left, y2_left = j_left[2][0], j_left[2][1]
	x3_left, y3_left = j_left[3][0], j_left[3][1]
	x4_left, y4_left = j_left[4][0], j_left[4][1]
	
	x1_right, y1_right = j_right[1][0], j_right[1][1]
	x2_right, y2_right = j_right[2][0], j_right[2][1]
	x3_right, y3_right = j_right[3][0], j_right[3][1]
	x4_right, y4_right = j_right[4][0], j_right[4][1]

	
	imgL = cv2.imread(f"{dataset_dir}/image_0/"+imgLp, 0)
	imgR = cv2.imread(f"{dataset_dir}/image_1/"+imgRp, 0)

	x1_left = min(x1_left, 1279)
	x2_left = min(x2_left, 1279)
	x3_left = min(x3_left, 1279)
	x4_left = min(x4_left, 1279)

	y1_left = min(y1_left, 719)
	y2_left = min(y2_left, 719)
	y3_left = min(y3_left, 719)
	y4_left = min(y4_left, 719)
 
	x1_right = min(x1_right, 1279)
	x2_right = min(x2_right, 1279)
	x3_right = min(x3_right, 1279)
	x4_right = min(x4_right, 1279)

	y1_right = min(y1_right, 719)
	y2_right = min(y2_right, 719)
	y3_right = min(y3_right, 719)
	y4_right = min(y4_right, 719)

	# print(j_left)
	# print("Corners: ", [x1_left, y1_left, x2_left, y2_left, x3_left, y3_left, x4_left, y4_left])

	# print(imgLp)
	# print(imgRp)



	# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	# disparity = stereo.compute(imgL, imgR)

	# print("Disparity: ", [disparity[round(y1_left), round(x1_left)], disparity[round(y2_left), round(x2_left)], disparity[round(y3_left.0), round(x3_left.0)], disparity[round(y4_left), round(x4_left)]])

	x_coord_1_left, y_coord_1_left = int(x1_left), int(y1_left)
	x_coord_2_left, y_coord_2_left = int(x2_left), int(y2_left)
	x_coord_3_left, y_coord_3_left = int(x3_left), int(y3_left)
	x_coord_4_left, y_coord_4_left = int(x4_left), int(y4_left)
 
	x_coord_1_right, y_coord_1_right = int(x1_right), int(y1_right)
	x_coord_2_right, y_coord_2_right = int(x2_right), int(y2_right)
	x_coord_3_right, y_coord_3_right = int(x3_right), int(y3_right)
	x_coord_4_right, y_coord_4_right = int(x4_right), int(y4_right)

	# centreCol = 640.0
	# centreRow = 360.0
	# b = 0.120008
	# f = 699.014
	# c = 605.254
	# c1 = 612.0278930664062
	# c2 = 609.567138671875
	# c = c1 - c2

	extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
		bytes=32, use_orientation=False)

	kps_left = []
	kps_left.append(cv2.KeyPoint(x_coord_1_left, y_coord_1_left, 3))
	kps_left.append(cv2.KeyPoint(x_coord_2_left, y_coord_2_left, 3))
	kps_left.append(cv2.KeyPoint(x_coord_3_left, y_coord_3_left, 3))
	kps_left.append(cv2.KeyPoint(x_coord_4_left, y_coord_4_left, 3))

	kps_left, descs_left = extractor.compute(imgL, kps_left)

	kps_right = []
	kps_right.append(cv2.KeyPoint(x_coord_1_right, y_coord_1_right, 3))
	kps_right.append(cv2.KeyPoint(x_coord_2_right, y_coord_2_right, 3))
	kps_right.append(cv2.KeyPoint(x_coord_3_right, y_coord_3_right, 3))
	kps_right.append(cv2.KeyPoint(x_coord_4_right, y_coord_4_right, 3))
	kps_right, descs_right = extractor.compute(imgR, kps_right)

	# kps_l = [cv2.KeyPoint(x_coord_1,y_coord_1,32.0),cv2.KeyPoint(x_coord_2,y_coord_2,32.0), cv2.KeyPoint(x_coord_3,y_coord_3,32.0), cv2.KeyPoint(x_coord_4,y_coord_4,32.0)]

	# kps_r = [cv2.KeyPoint(x_coord_1 + disparity[y_coord_1, x_coord_1],y_coord_1,32.0),cv2.KeyPoint(x_coord_2 + disparity[y_coord_2, x_coord_2],y_coord_2,32.0), cv2.KeyPoint(x_coord_3 + disparity[y_coord_3, x_coord_3],y_coord_3,32.0), cv2.KeyPoint(x_coord_4 + disparity[y_coord_4, x_coord_4],y_coord_4,32.0)]

	# corners = [[x_coord_1, y_coord_1],
	# 		[x_coord_2, y_coord_2],
	# 		[x_coord_3, y_coord_3],
	# 		[x_coord_4, y_coord_4]]

	# if hypotheses_no == 1 or hypotheses_no == 2:
	# 	combined_image = combine(imgLp, imgRp)
	# 	kps_depths = find_kps_depths(combined_image)
	# 	depths = find_corner_depths(corners, kps_depths, hypothesis_no=hypotheses_no)

	# elif hypotheses_no == 3 or hypotheses_no == 4:
	# 	depths = find_corner_depths(corners, depth_map_path='./', hypothesis_no=hypotheses_no)

	# z1 = depths[0]
	# z2 = depths[1]
	# z3 = depths[2]
	# z4 = depths[3]

	# u1 = x_coord_1 - centreCol
	# u2 = x_coord_2 - centreCol
	# u3 = x_coord_3 - centreCol
	# u4 = x_coord_4 - centreCol
	# v1 = -(y_coord_1 - centreRow)
	# v2 = -(y_coord_2 - centreRow)
	# v3 = -(y_coord_3 - centreRow)
	# v4 = -(y_coord_4 - centreRow)
	# x1_left = u1 * z1 / f
	# y1_left = v1 * z1 / f
	# x2_left = u2 * z2 / f
	# y2_left = v2 * z2 / f
	# x3_left = u3 * z3 / f
	# y3_left = v3 * z3 / f
	# x4_left = u4 * z4 / f
	# y4_left = v4 * z4 / f
	# x_arr = [x1_left, x2_left, x3_left, x4_left]
	# y_arr = [y1_left, y2_left, y3_left, y4_left]
	# z_arr = [z1, z2, z3, z4]
	# final_coord = []

	ret_kps_left, ret_kps_right, ret_desc_left, ret_desc_right = [], [], [], []
	image_coord_left = []
	image_coord_right = []
	image_coord_left.append((x_coord_1_left, y_coord_1_left))
	image_coord_left.append((x_coord_2_left, y_coord_2_left))
	image_coord_left.append((x_coord_3_left, y_coord_3_left))
	image_coord_left.append((x_coord_4_left, y_coord_4_left))
 
	image_coord_right.append((x_coord_1_right, y_coord_1_right))
	image_coord_right.append((x_coord_2_right, y_coord_2_right))
	image_coord_right.append((x_coord_3_right, y_coord_3_right))
	image_coord_right.append((x_coord_4_right, y_coord_4_right))

	# if (len(kps_left) < 4 or len(kps_right) < 4):
	# 	return None

	for i in range(min(len(kps_left), len(kps_right))):
		ret_kps_left.append(kps_left[i])
		ret_kps_right.append(kps_right[i])
		ret_desc_left.append(descs_left[i])
		ret_desc_right.append(descs_right[i])
 
	# if (len(kps_left) >= 4):
	return  ret_kps_left, ret_kps_right, ret_desc_left, ret_desc_right, image_coord_left, image_coord_right

	# print("None found")
	#return None
# print("Z : " + str(z) + " X : " + str(x) + " Y :" + str(y))
# print(disparity[y_coord][x_coord])
# plt.imshow(disparity,'gray')

# plt.show()
