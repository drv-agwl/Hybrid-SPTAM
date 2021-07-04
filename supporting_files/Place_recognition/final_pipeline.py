from keras import backend as K
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os as os
import json
from tqdm import tqdm
from config import dataset_dir

img_dirs = [f'{dataset_dir}/image_0', f'{dataset_dir}/image_1']
processed_img_dir = f'{dataset_dir}/processed_images'
preds = {}


for v, img_dir in enumerate(img_dirs):
    for i in tqdm(range(len(os.listdir(img_dir)))):
        img_name = os.listdir(img_dir)[i]
        input_image_path = os.path.join(img_dir, img_name)

        from object_detection.object_detector.inference import make_predictions

        object_boxes = make_predictions(input_image_path)
        K.clear_session()
        tf.reset_default_graph()

        from object_detection.alphabet_detector.inference import make_predictions_alphabet

        alphabet_boxes = make_predictions_alphabet(input_image_path)

        all_boxes = object_boxes + alphabet_boxes

        # preparing json of predictions
        preds[img_name] = all_boxes

        # preparing semantically enhanced images for place recognition
        objs = ['door', 'shelf', 'mcb', 'window', 'fire_extinguisher', 'tv']
        objs = ['label']

        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (224, 224))
        # if img_name == 'left000310.png':
        #     print("here")
        # plt.imshow(img)
        # plt.show()

        for obj in objs:
            semantic_channel = np.zeros((img.shape[0], img.shape[1]))

            for detected_objs in all_boxes:
                if detected_objs[0] != obj:
                    continue

                x1 = int(round(detected_objs[1]))
                x2 = int(round(detected_objs[3]))
                y1 = int(round(detected_objs[2]))
                y2 = int(round(detected_objs[4]))

                for x in range(min(x1, 1920), min(x2 + 1, 1920)):
                    for y in range(min(y1, 1080), min(y2 + 1, 1080)):
                        semantic_channel[y][x] = 255.

            semantic_channel = np.expand_dims(semantic_channel, axis=-1)
            # plt.imshow(semantic_channel[:, :, 0], cmap='gray')
            # plt.show()
            semantic_channel = cv2.resize(semantic_channel, (224, 224))
            resized_img = np.concatenate([resized_img, np.expand_dims(semantic_channel, axis=-1)], axis=-1)

        resized_img = resized_img / 255.
        np.save(os.path.join(processed_img_dir, img_name[:-4] + '.npy'), resized_img)

    if v == 0:
        with open('./final_preds.json', 'w') as f:
            json.dump(preds, f)
    else:
        with open('./final_preds_right.json', 'w') as f:
            json.dump(preds, f)


# # combine objects and alphabets detections
# processed_img_dir = './Complete_dataset/image_with_semantics_objs'
# obj_semantics_dir = './Complete_dataset/objs'
# alpha_semantics_dir = './Complete_dataset/alphabets'
#
# for img_name in os.listdir(obj_semantics_dir):
#     obj_arr = np.load(os.path.join(obj_semantics_dir, img_name))
#     alpha_arr = np.load(os.path.join(alpha_semantics_dir, img_name))
#     combined_arr = np.concatenate([obj_arr, np.expand_dims(alpha_arr[:, :, -1], -1)], -1)
#     np.save(os.path.join(processed_img_dir, img_name), combined_arr)

