from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
from Models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from imageio import imread
from keras.preprocessing import image

# Set the image size.
img_height = 300
img_width = 300

# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=11,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.3,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.
weights_path = './object_detection/object_detector/weights/epoch-84_loss-4.2308_val_loss-3.2708.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


########################################### Model-load finished ################################################################

def make_predictions(img_path='./object_detection/object_detector/Images/val/seq-22_images_0/left000475.png'):
    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.

    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    # print(y_pred_thresh[0])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background',
               'door-knob', 'tv', 'shelf', 'fire_extinguisher',
               'fire_button', 'fire_alarm', 'mcb', 'light',
               'door', 'window']

    # plt.figure(figsize=(20, 12))
    # plt.imshow(orig_images[0])

    current_axis = plt.gca()

    bboxes = []
    for box in y_pred_thresh[0]:
        if int(box[0]) in [1, 5, 6, 8]:
            continue  # neglect door-knobs, fire_button, fire_alarm, light

        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        # color = colors[int(box[0])]
        # label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        # current_axis.add_patch(
        #     plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
        bboxes.append([classes[int(box[0])], xmin, ymin, xmax, ymax])

    return bboxes


# function call to predict boxes
# make_predictions('./Images/val/seq-23_images_0/left003392.png')
# make_predictions('./Images/val/seq-23_images_0/left001852.png')
# make_predictions('./Images/val/seq-23_images_0/left003419.png')
# make_predictions('./Images/val/seq-23_images_0/left003392.png')
