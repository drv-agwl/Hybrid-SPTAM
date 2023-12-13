#STEPS to RUN

Checkout our paper at: https://ieeexplore.ieee.org/document/9837164

## Essential steps before running
1. Create a sample of images whose ground-truth poses and ground-truth corners of the semantics present in them are available
2. Place the ground-truth poses inside the SPTAM folder. You can find a sample file present in the SPTAM folder
3. Update the config file for dataset paths and characteristic images

## Training object detection model
1. Go to the object_detection folder; there, you will find two directories, object_detector (to train over permanent objects) and alphabet_detector (to train over temporarily placed alphabet placards).
2. Go to object_detector and run train.py to train the object detection model
3. Go to alphabet_detector and run train.py to train the alphabet detection model

## Training Place Recognition model
1. Sampled out images from the images with the ground-truth poses for creating characteristic images
2. Assign a sample of images from the dataset the labels from their corresponding characteristic image.
3. Inside Place_recognition, run generate_semantic_enhanced_images.py to generate semantically enhanced images.
4. Run training_with_semantics.py inside the Place_recognition to train the place recognition model

## Inference Object detection and Place recognition outputs
1. Run final_pipeline.py inside Place_recognition to generate a json file

## Inference Corner Detection output
1. Run run.py inside corner_detection to generate object corners in two json files (one each for left and right images).

## Installing and Running SPTAM
1. To install SPTAM, refer to and follow the steps from: https://github.com/uoip/stereo_ptam
2. Run place_image.py inside SPTAM 
3. Run sptam.py --path=/path/to/dataset to generate final_positions.txt (inside output_files directory) as the final results.
