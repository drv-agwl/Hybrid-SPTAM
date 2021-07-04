import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
import os as os
from config import dataset_dir, objs

bboxes_path = './SPTAM/final_corners.json'
images_path = f'{dataset_dir}/image_0'
processed_img_dir = f'{dataset_dir}/processed_images'

all_bboxes = load_json(bboxes_path)

for img_name in os.listdir(images_path):
    print('Processing ' + img_name)
    img_path = os.path.join(images_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (224, 224))
    plt.imshow(img)
    plt.show()
    bboxes = all_bboxes[img_name]

    for obj in objs:
        semantic_channel = np.zeros((img.shape[0], img.shape[1]))

        for detected_objs in bboxes:
            if detected_objs[0] != obj:
                continue

            x1 = round(detected_objs[1])
            x2 = round(detected_objs[3])
            y1 = round(detected_objs[2])
            y2 = round(detected_objs[4])

            for x in range(min(x1, 1280), min(x2 + 1, 1280)):
                for y in range(min(y1, 720), min(y2 + 1, 720)):
                    semantic_channel[y][x] = 255.

        semantic_channel = np.expand_dims(semantic_channel, axis=-1)
        plt.imshow(semantic_channel[:,:,0], cmap='gray')
        plt.show()
        semantic_channel = cv2.resize(semantic_channel, (224, 224))
        resized_img = np.concatenate([resized_img, np.expand_dims(semantic_channel, axis=-1)], axis=-1)

    resized_img = resized_img/255.
    np.save(os.path.join(processed_img_dir, img_name[:-4] + '.npy'), resized_img)
