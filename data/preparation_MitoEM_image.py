# ------------------------------------------------------------ #
#
# File : data/preparation_MitoEM_image.py
# Authors : CM
#
# ------------------------------------------------------------ #

import os
import sys

import numpy as np
import skimage
from skimage import io


# Read
print("READ")

PATH_IN = "/HDD1/data/MitoEM/MitoEM-H/"
PATH_OUT = "/home/cyril/Documents/Data/MitoEM/MitoEM-H/"
'''
PATH_IN = "/HDD1/data/MitoEM/MitoEM-R/"
PATH_OUT = "/home/cyril/Documents/Data/MitoEM/MitoEM-R/"
'''

image_train = np.array(skimage.io.imread(PATH_IN + "im_train.tif")).astype(np.float16)
image_valid = np.array(skimage.io.imread(PATH_IN + "im_val.tif")).astype(np.float16)

print("PROCESSING")
image_train_min = image_train.min()
image_train_max = image_train.max()
image_train_normalized_f16 = ((image_train - image_train_min) / (image_train_max - image_train_min)).astype(np.float16)
print(image_train_normalized_f16.min(), image_train_normalized_f16.max())

if image_train_normalized_f16.min() != 0 or image_train_normalized_f16.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)

image_valid_min = image_valid.min()
image_valid_max = image_valid.max()
image_valid_normalized_f16 = ((image_valid - image_valid_min) / (image_valid_max - image_valid_min)).astype(np.float16)
print(image_valid_normalized_f16.min(), image_valid_normalized_f16.max())

if image_valid_normalized_f16.min() != 0 or image_valid_normalized_f16.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)

# Save
print("SAVE")
np.save(PATH_OUT + "TRAIN_IMAGE_NORMALIZED_F16.npy", image_train_normalized_f16)
np.save(PATH_OUT + "VALID_IMAGE_NORMALIZED_F16.npy", image_valid_normalized_f16)
