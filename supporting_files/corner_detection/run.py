from demo_1 import main
import os as os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from config import dataset_path

dataset_paths = [f'{dataset_path}\\image_0',f'{dataset_path}\\image_1']


def euclidean(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def get_corners(img_path, bboxes, pad_value=2.5):
    # plt.imshow(plt.imread(img_path))
    all_corners = []
    nlines, nscores = main(img_path)
    for i, t in enumerate([0.94, 0.95, 0.96, 0.97, 0.98, 0.99]):
        for (a, b), s in zip(nlines, nscores):
            if s < t:
                continue
            all_corners.append([a[1], a[0]])
            all_corners.append([b[1], b[0]])

    # corners lying inside all padded bbox
    filtered_corners = {}
    for bbox in bboxes:
        object_class = bbox[0]
        corners_of_an_object = []
        padded_bbox = [bbox[1] - pad_value, bbox[2] - pad_value, bbox[3] + pad_value, bbox[4] + pad_value]

        # get corners within padded bbox
        # hard-coding for 4 corners only (sorry for disappointing :( )
        padded_bbox_coords = [[padded_bbox[0], padded_bbox[1]],
                              [padded_bbox[2], padded_bbox[1]],
                              [padded_bbox[0], padded_bbox[3]],
                              [padded_bbox[2], padded_bbox[3]]]

        for i, bbox_coord in enumerate(padded_bbox_coords):
            dist = [euclidean(corner_point, bbox_coord) for corner_point in all_corners]
            min_dist_corner = all_corners[dist.index(min(dist))]
            corners_of_an_object.append(min_dist_corner)
            # plt.scatter(min_dist_corner[0], min_dist_corner[1])

        filtered_corners[object_class] = corners_of_an_object

    # plt.show()
    return filtered_corners


if __name__ == '__main__':
    with open('bboxes/final_pred.json') as f:
        bboxes = json.load(f)

    # save corners in a file
    corners_dict = {}
    for v, dataset_path in enumerate(dataset_paths):
        for i in tqdm(range(len(os.listdir(dataset_path)))):
            img_name = os.listdir(dataset_path)[i]
            img_path = os.path.join(dataset_path, img_name)
            corners = get_corners(img_path, bboxes[img_name])
            corners_dict[img_name] = corners

        if v==0:
            with open('./corners.json', 'w') as f:
                json.dump(corners_dict, f, indent=2)
        else:
            with open('./corners_right.json', 'w') as f:
                json.dump(corners_dict, f, indent=2)