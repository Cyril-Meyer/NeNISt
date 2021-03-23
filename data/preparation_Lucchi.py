# ------------------------------------------------------------ #
#
# File : data/preparation_Lucchi.py
# Authors : CM
#
# ------------------------------------------------------------ #

import os
import sys

import numpy as np
import skimage
from skimage import io

from distance_transform import *

anisotropy=(5, 5, 5)
normalize = normalize_tanh
normalize_scale_pos = 10.0
normalize_scale_neg = 20.0


PATH_IN = "/HDD1/data/EPFL_EMD_Lucchi/"
PATH_OUT = "/home/cyril/Documents/Data/Lucchi/"

print("READ")
image_train = np.array(skimage.io.imread(PATH_IN + "training.tif")).astype(np.float16)
image_valid = np.array(skimage.io.imread(PATH_IN + "testing.tif")).astype(np.float16)
label_train = ((np.array(skimage.io.imread(PATH_IN + "training_groundtruth.tif")) > 0)*1.0).astype(np.uint8)
label_valid = ((np.array(skimage.io.imread(PATH_IN + "testing_groundtruth.tif")) > 0)*1.0).astype(np.uint8)

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

label_train_dt = label_dt_f16(label_train, anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)
label_valid_dt = label_dt_f16(label_valid, anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)

print("SAVE")
np.save(PATH_OUT + "TRAIN_IMAGE_NORMALIZED_F16.npy", image_train_normalized_f16)
np.save(PATH_OUT + "VALID_IMAGE_NORMALIZED_F16.npy", image_valid_normalized_f16)
np.save(PATH_OUT + "TRAIN_LABEL.npy", label_train)
np.save(PATH_OUT + "TRAIN_LABEL_DT.npy", label_train_dt)
np.save(PATH_OUT + "VALID_LABEL.npy", label_valid)
np.save(PATH_OUT + "VALID_LABEL_DT.npy", label_valid_dt)
skimage.io.imsave(PATH_OUT + "train_bin.tif", label_train*255)
skimage.io.imsave(PATH_OUT + "train_bin_dt.tif", ((label_train_dt+1)/2*255).astype(np.uint8))
skimage.io.imsave(PATH_OUT + "valid_bin.tif", label_valid*255)
skimage.io.imsave(PATH_OUT + "valid_bin_dt.tif", ((label_valid_dt+1)/2*255).astype(np.uint8))

if ((label_train_dt > 0) == (label_train > 0)).all():
    print("OK")
else:
    print("ERROR 01 : CHECK CODE")
