from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from math import ceil
from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from keras import backend as K
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize

weights_source_path = './weights/epoch-10_loss-5.7711_val_loss-5.4053.h5'


# CONFIG
classifier_names = ['conv4_3_norm_mbox_conf',
                    'fc7_mbox_conf',
                    'conv6_2_mbox_conf',
                    'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf',
                    'conv9_2_mbox_conf']

img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
mean_color = [123, 117, 104]
subtract_mean = [123, 117, 104]  # The per-channel mean of the images in the dataset
swap_channels = [2, 1, 0]  # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
n_classes = 2  # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
# scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
          1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
           0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
normalize_coords = True

# Build the Keras model.
K.clear_session()  # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

# 2: Load some weights into the model.

# weights_path = weights_destination_path
# model.load_weights(weights_path, by_name=True)

model.load_weights(weights_source_path, by_name=True)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

# Load images
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# The directories that contain the images.

# temp = '../Image/img/'
# temp_A = '../Image/Annot/'
# file = '../Image/files.txt'

images_A = './Images/train/seq-22_images_0/'
images_B = './Images/train/seq-23_images_0/'
# images_C = '../Image/C/'

Val_A = './Images/val/seq-22_images_0/'
Val_B = './Images/val/seq-23_images_0/'
# Val_C = '../Image/C/'

# The directories that contain the annotations.
annotations_A = './Images/train/seq-22_images_0_annot/'
annotations_B = './Images/train/seq-23_images_0_annot/'
# annotations_C = '../Image/C_Annotations/'

Val_A_Annotations = './Images/val/seq-22_images_0_annot/'
Val_B_Annotations = './Images/val/seq-23_images_0_annot/'
# Val_C_Annotations = '../Image/Val_C_Annotations/'

# The paths to the image sets.
files_A = './Images/train/seq-22_images_0_files.txt'
files_B = './Images/train/seq-23_images_0_files.txt'
# files_C = '../Image/C_files.txt'

Val_files_A = './Images/val/seq-22_images_0_files.txt'
Val_files_B = './Images/val/seq-23_images_0_files.txt'
# Val_files_C = '../Image/Val_C_files.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'alphabet']

# temp_classes = ['Car', 'Truck', 'Human', 'Motorcycle', 'Animal', 'Bicycle', 'Ricksaw']

train_dataset.parse_xml(images_dirs=[images_A, images_B],
                        image_set_filenames=[files_A, files_B],
                        annotations_dirs=[annotations_A, annotations_B],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[Val_A, Val_B],
                      image_set_filenames=[Val_files_A, Val_files_B],
                      annotations_dirs=[Val_A_Annotations, Val_B_Annotations],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

# # Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# # speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# # option in the constructor, because in that cas the images are in memory already anyway. If you don't
# # want to create HDF5 datasets, comment out the subsequent two function calls.

train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_trainval.h5',
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)


# 3: Set the batch size.

batch_size = 4  # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


def lr_schedule(epoch):
    if epoch < 80:
        return 0.000001
    elif epoch < 100:
        return 0.000001
    else:
        return 0.0000001


model_checkpoint = ModelCheckpoint(filepath='./weights/epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 20
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size / batch_size),
                              initial_epoch=initial_epoch)
