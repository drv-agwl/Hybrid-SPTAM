from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input, concatenate
from keras import layers
from keras import models


def get_model(channels=3, num_classes=14):
    input_image = Input((224, 224, channels))

    x = Conv2D(128, (10, 10), activation='relu')(input_image)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (8, 8), activation='relu', strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(512, (5, 5), activation='relu', strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Flatten()(x)  # 512

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_image, outputs=output)

    return model


def get_model_mobilenet():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(14, activation='softmax'))
    return model
