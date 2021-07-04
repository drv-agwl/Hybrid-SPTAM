import os as os
from run import get_corners, euclidean
import json
import pandas as pd
from tqdm import tqdm
import math

dataset_path = './evaluate_corner_detection/test_dataset'

with open('./evaluate_corner_detection/ground_bboxes.json') as f:
    bboxes = json.load(f)

gt_corners = pd.read_csv('./evaluate_corner_detection/ground_truth_corners.csv', header=None)
gt_corners = gt_corners.iloc[:, 0:4]

# get all corners in a dict
corners_dict = {}
for i in tqdm(range(len(os.listdir(dataset_path)))):
    img_name = os.listdir(dataset_path)[i]
    img_path = os.path.join(dataset_path, img_name)
    corners = get_corners(img_path, bboxes[img_name])
    corners_dict[img_name] = corners

rmse_error = []
for i in tqdm(range(len(os.listdir(dataset_path)))):
    img_name = os.listdir(dataset_path)[i]
    all_pred_corners = corners_dict[img_name]
    all_gt_corners = gt_corners[gt_corners.iloc[:, 3] == img_name]

    rmse_obj_errors = []
    for obj, pred_obj_corners in all_pred_corners.items():
        gt_obj_corners = all_gt_corners[all_gt_corners.iloc[:, 0] == obj].iloc[:, 1:3].to_numpy()
        corner_error = 0
        for corner_idx in range(len(pred_obj_corners)):
            pred_corner = pred_obj_corners[corner_idx]
            error = min([euclidean(pred_corner, gt_obj_corner) for gt_obj_corner in gt_obj_corners])
            corner_error += math.sqrt(error)

        rmse_obj_errors.append(corner_error/4)

    rmse_error.append(sum(rmse_obj_errors)/len(rmse_obj_errors))

print(sum(rmse_error)/len(rmse_error))

