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


train_X = []
train_y = []
val_X = []
val_y = []

train_dir = './Complete_dataset/train'
# val_dir = './Dataset/images_with_semantics/val'

for place_id in range(14):
    category = 'Place_' + str(place_id)
    for img_name in os.listdir(os.path.join(train_dir, category)):
        img = np.load(os.path.join(os.path.join(train_dir, category), img_name))
        train_X.append(img)
        train_y.append(place_id)

    # for img_name in os.listdir(os.path.join(val_dir, category)):
    #     img = np.load(os.path.join(os.path.join(val_dir, category), img_name))
    #     val_X.append(img)
    #     val_y.append(place_id)

train_X = np.array(train_X)
train_y = np.array(train_y)
# val_X = np.array(val_X)
# val_y = np.array(val_y)

train_X, train_y = shuffle(train_X, train_y)
# val_X, val_y = shuffle(val_X, val_y)

train_y = convert_to_one_hot(train_y)
# val_y = convert_to_one_hot(val_y)

train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1)

# print(train_X.shape)
# print(train_y.shape)

model = get_model(channels=10)

opt = Adam(0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

filepath = "./Models/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=False)

print(model.summary())

history = model.fit(train_X, train_y,
                    validation_data=(val_X, val_y),
                    epochs=100,
                    callbacks=[checkpoint])

print(history.history.keys())