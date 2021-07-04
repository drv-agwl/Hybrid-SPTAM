import os as os
from shutil import move

src_dir = './Dataset/images'
target_dir = './Dataset/images_with_semantics'

# train images
# for cat in range(7):
#     category = 'Place_'+str(cat)
#     for img in os.listdir(os.path.join(os.path.join(src_dir, 'train'), category)):
#         target_file = os.path.join(target_dir, img)[:-4]+'.npy'
#         move(target_file, os.path.join(os.path.join(os.path.join(target_dir, 'train'), category), img[:-4]+'.npy'))

# val images
for cat in range(7):
    category = 'Place_'+str(cat)
    for img in os.listdir(os.path.join(os.path.join(src_dir, 'val'), category)):
        target_file = os.path.join(target_dir, img)[:-4]+'.npy'
        move(target_file, os.path.join(os.path.join(os.path.join(target_dir, 'val'), category), img[:-4]+'.npy'))