import json
import numpy as np
import pandas as pd
from config import *

corner_gt_file = f"{robot_ground_truth}"
obj_det_file = "./pred.json"

corner_gt = pd.read_csv(corner_gt_file , dtype=str)
with open(obj_det_file) as f:
 	obj_det = json.load(f)

mapping = {"C1" : "bl", "C2" : "br", "C3" : "tr", "C4" : "tl"}
print(list(corner_gt['label']))
final_pred = {}
for key in obj_det.keys():
	fin_list = []
	for i in obj_det[key]:
		final_label = i[0]
		obj_dict = {}
		obj_list = []
		if(i[0] in list(corner_gt['label'])):
			for idx,label in enumerate(list(corner_gt["label"])):
				if(label == i[0]):
					print(label, i[0])
					index = idx
					final_label = label
					break
			
			bl = (i[1], i[4])
			br = (i[3], i[4])
			tr = (i[3], i[2])
			tl = (i[1], i[2])

			obj_list.append([tl, corner_gt['C1_X'][index], corner_gt['C1_Y'][index], corner_gt['C1_Z'][index]])
			obj_list.append([tr, corner_gt['C2_X'][index], corner_gt['C2_Y'][index], corner_gt['C2_Z'][index]])
			obj_list.append([bl, corner_gt['C4_X'][index], corner_gt['C4_Y'][index], corner_gt['C4_Z'][index]])
			obj_list.append([br, corner_gt['C3_X'][index], corner_gt['C3_Y'][index], corner_gt['C3_Z'][index]])
			print(final_label)
			obj_dict[final_label] = obj_list
			fin_list.append(obj_dict)
	final_pred[key] = np.array(fin_list).tolist()

f = open("./place_semantic.json", "w+")
json_data = json.dumps(final_pred)
f.write(json_data)

