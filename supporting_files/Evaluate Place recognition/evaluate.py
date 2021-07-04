import os as os
import json
import os as os
import numpy as np
from model import get_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from utils import convert_to_one_hot
from sklearn.model_selection import train_test_split
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


dataset_dir = '../Complete_dataset'
test_dir = os.path.join(dataset_dir, 'test')

with open('./place_pediction.json') as f:
    predictions = json.load(f)

print(predictions)

gt_test_labels = {}

test_images = {}
for place_id in os.listdir(test_dir):
    for img in os.listdir(os.path.join(test_dir, place_id)):
        gt_test_labels[img] = place_id
        test_images[img] = np.load(os.path.join(os.path.join(test_dir, place_id), img))

test_size = len(gt_test_labels)

############################################################################################
model = get_model(channels=10)

opt = Adam(0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])
model.load_weights('../Models/saved-model-99-0.94.hdf5')

inputs = []
for place_id, img in test_images.items():
    inputs.append(img)
inputs = np.array(inputs)

pred = model.predict(inputs, verbose=True)

i = 0
for place_id, img in test_images.items():
    test_images[place_id] = 'Place_'+str(np.argmax(pred[i], axis=-1))
    i += 1

print(test_images)
print(gt_test_labels)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
f1 = f1_score(list(gt_test_labels.values()), list(test_images.values()), average='micro')
prec = precision_score(list(gt_test_labels.values()), list(test_images.values()), average='micro')
rec = recall_score(list(gt_test_labels.values()), list(test_images.values()), average='micro')
acc = accuracy_score(list(gt_test_labels.values()), list(test_images.values()), average='micro')