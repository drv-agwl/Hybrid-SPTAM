# CONFIG

dataset_dir = "/path/to/dataset"
robot_ground_truth = "/path/to/robot_ground_truth.csv"
corner_gt = "/path/to/char_images.json"

characteristics = {"Place_0" : "000000_rgb.png",
"Place_1" : "000014_rgb.png",
"Place_2" : "000018_rgb.png",
"Place_3": "000023_rgb.png",
"Place_4" : "000028_rgb.png",
"Place_5" : "000032_rgb.png",
"Place_6" : "000051_rgb.png",
"Place_7" : "000055_rgb.png",
"Place_8" : "000056_rgb.png",
"Place_9" : "000064_rgb.png",
"Place_10" : "000035_rgb.png",
"Place_11" : "000039_rgb.png",
"Place_12" : "000043_rgb.png",
"Place_13" : "000011_rgb.png"
}

train_images = "/path/to/images/train"  # for place recognition
val_images = "/path/to/imags/val"   # for place recognition

objs = ['door', 'shelf', 'mcb', 'window', 'fire_extinguisher', 'tv']
