import numpy as np
import cv2
import os
import json
import glob
import copy
from tracker import find_all_corners_from_bboxs

def compute_coord_robot(imgLp, imgRp, j):

	# print("Beohar output: ", j)

	name = j[0]

	x1 , y1 = j[4][0], j[4][1]
	x2 , y2 = j[3][0], j[3][1]
	x3 , y3 = j[2][0], j[2][1]
	x4 , y4 = j[1][0], j[1][1]

	x1 = min(x1, 1279)
	x2 = min(x2, 1279)
	x3 = min(x3, 1279)
	x4 = min(x4, 1279)

	y1 = min(y1, 719)
	y2 = min(y2, 719)
	y3 = min(y3, 719)
	y4 = min(y4, 719)

	# print(j)
	# print("Corners: ", [x1, y1, x2, y2, x3, y3, x4, y4])


	# print(imgLp)
	# print(imgRp)

	imgL = cv2.imread('/home/mehul/Downloads/seq22/00/image_0/'+imgLp, 0)
	imgR = cv2.imread('/home/mehul/Downloads/seq22/00/image_1/'+imgRp, 0)

	stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
	disparity = stereo.compute(imgL,imgR)

	# print("Disparity: ", [disparity[round(y1), round(x1)], disparity[round(y2), round(x2)], disparity[round(y3.0), round(x3.0)], disparity[round(y4), round(x4)]])

	x_coord_1, y_coord_1 = int(x1), int(y1)
	x_coord_2, y_coord_2 = int(x2), int(y2)
	x_coord_3, y_coord_3 = int(x3), int(y3)
	x_coord_4, y_coord_4 = int(x4), int(y4)

	centreCol = 640.0
	centreRow = 360.0
	b = 0.120008
	f = 699.014
	c = 605.254
	c1 = 612.0278930664062
	c2 = 609.567138671875
	c = c1-c2

	extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

	kps_left=[]
	kps_left.append(cv2.KeyPoint(x_coord_1, y_coord_1,3))
	kps_left.append(cv2.KeyPoint(x_coord_2, y_coord_2,3))
	kps_left.append(cv2.KeyPoint(x_coord_3, y_coord_3,3))
	kps_left.append(cv2.KeyPoint(x_coord_4, y_coord_4,3))

	kps_left, descs_left = extractor.compute(imgL, kps_left)

	kps_right=[]
	kps_right.append(cv2.KeyPoint(x_coord_1+disparity[y_coord_1,x_coord_1], y_coord_1,3))
	kps_right.append(cv2.KeyPoint(x_coord_2+disparity[y_coord_2,x_coord_2], y_coord_2,3))
	kps_right.append(cv2.KeyPoint(x_coord_3+disparity[y_coord_3,x_coord_3], y_coord_3,3))
	kps_right.append(cv2.KeyPoint(x_coord_4+disparity[y_coord_4,x_coord_4], y_coord_4,3))
	kps_right, descs_right = extractor.compute(imgR, kps_right)

	#kps_l = [cv2.KeyPoint(x_coord_1,y_coord_1,32.0),cv2.KeyPoint(x_coord_2,y_coord_2,32.0), cv2.KeyPoint(x_coord_3,y_coord_3,32.0), cv2.KeyPoint(x_coord_4,y_coord_4,32.0)]

	#kps_r = [cv2.KeyPoint(x_coord_1 + disparity[y_coord_1, x_coord_1],y_coord_1,32.0),cv2.KeyPoint(x_coord_2 + disparity[y_coord_2, x_coord_2],y_coord_2,32.0), cv2.KeyPoint(x_coord_3 + disparity[y_coord_3, x_coord_3],y_coord_3,32.0), cv2.KeyPoint(x_coord_4 + disparity[y_coord_4, x_coord_4],y_coord_4,32.0)]

	z1 = (b*f)/(disparity[y_coord_1,x_coord_1] + c)
	z2 = (b*f)/(disparity[y_coord_2,x_coord_2] + c)
	z3 = (b*f)/(disparity[y_coord_3,x_coord_3] + c)
	z4 = (b*f)/(disparity[y_coord_4,x_coord_4] + c)

	u1 = x_coord_1 - centreCol
	u2 = x_coord_2 - centreCol
	u3 = x_coord_3 - centreCol
	u4 = x_coord_4 - centreCol
	v1 = -(y_coord_1 - centreRow)
	v2 = -(y_coord_2 - centreRow)
	v3 = -(y_coord_3 - centreRow)
	v4 = -(y_coord_4 - centreRow)
	x1 = u1*z1/f
	y1 = v1*z1/f
	x2 = u2*z2/f
	y2 = v2*z2/f
	x3 = u3*z3/f
	y3 = v3*z3/f
	x4 = u4*z4/f
	y4 = v4*z4/f
	x_arr = [x1,x2,x3,x4]
	y_arr = [y1,y2,y3,y4]
	z_arr = [z1,z2,z3,z4]
	final_coord = []

	ret_kps_left, ret_kps_right, ret_desc_left, ret_desc_right = [], [], [], []
	image_coord = []
	image_coord.append((x_coord_1,y_coord_1))
	image_coord.append((x_coord_2,y_coord_2))
	image_coord.append((x_coord_3,y_coord_3))
	image_coord.append((x_coord_4,y_coord_4))

	if(len(kps_left)<4 or len(kps_right)<4):
		return None
	for i in range(len(z_arr)):
		if(z_arr[i] > 0 and  z_arr[i] <= 1 ):
			final_coord.append([10*x_arr[i], 10*y_arr[i], 10*z_arr[i]])
			ret_kps_left.append(kps_left[i])
			ret_kps_right.append(kps_right[i])
			ret_desc_left.append(descs_left[i])
			ret_desc_right.append(descs_right[i])

	if(len(final_coord) >=4):
		return final_coord, ret_kps_left, ret_kps_right, ret_desc_left, ret_desc_right, image_coord

	#print("None found")
	return None
#print("Z : " + str(z) + " X : " + str(x) + " Y :" + str(y))
#print(disparity[y_coord][x_coord])
#plt.imshow(disparity,'gray')

#plt.show()
