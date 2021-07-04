import numpy as np
import sys

def corresp_corner(corner_l, corner_r):
	final_left_corner = []
	final_right_corner = []
	for obj_l_1 in corner_l:
		obj_l = obj_l_1[0]
		right_corner = None
		min_dist = sys.maxsize
		for obj_r in corner_r:
			if(obj_l[0] == obj_r[0]):
				corner_1_l, corner_1_r = np.array(obj_l[1:]).reshape(4,2), np.array(obj_r[1:]).reshape(4,2)
				dist = np.sum((corner_1_l - corner_1_r)**2)
				if(dist < min_dist):
					min_dist = dist
					right_corner = obj_r
		if(right_corner is not None):
			final_left_corner.append(obj_l_1)
			final_right_corner.append(right_corner)

	return final_left_corner, final_right_corner