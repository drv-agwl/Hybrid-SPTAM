from keras.preprocessing.image import ImageDataGenerator
from model import get_model, get_model_mobilenet
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
from config import train_images, val_images


batch_size = 32

train_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(train_images,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(val_images,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')


model = get_model(channels=3)


opt = Adam(0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

filepath = "./Models/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=False)

print(model.summary())

# json_model = model.to_json()
# with open("model.json", "w") as outfile:
#     outfile.write(json_model)

history = model.fit_generator(training_set,
                              steps_per_epoch=666 // batch_size,
                              epochs=100,
                              validation_data=test_set,
                              validation_steps=81 // batch_size,
                              callbacks=[checkpoint])

model.load_weights('./saved-model-11-1.00.hdf5')

# img = cv2.imread('./Dataset/images/test/left008788.png', cv2.IMREAD_UNCHANGED)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# resized_img = cv2.resize(img, (224, 224))
# test_image = np.expand_dims(resized_img, axis=0)/255.
# result = model.predict(test_image)
# print(training_set.class_indices)
# print(np.argmax(result))