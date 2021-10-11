# ------------------------------------------------------------ #
#
# File : data/preparation_I3_79_4.py
# Authors : CM
#
# ------------------------------------------------------------ #

import os
import sys

import numpy as np
import skimage
from skimage import io
import h5py

from distance_transform import *


# Read
# Label 1 : Mitochondrion
# Label 2 : Cell membrane
# Label 3 : Endoplasmic reticulum
# Label 4 : Nucleus

image = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/I3/i3.tif"))
label_1 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/I3/Labels_i3-mitos_1-500.tif"))
label_2 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/I3/Labels_i3_MembraneCellule_1-250.tif"))
label_3 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/I3/Labels_i3_Reticulum_172-251.tif"))
label_4 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/I3/Labels_i3_Noyau_1-250.tif"))

if not (image.shape == label_1.shape == label_2.shape == label_3.shape == label_4.shape):
    print("ERROR, IMAGE SHAPE")
    sys.exit(os.EX_DATAERR)

# [172, 250] (1 based in imageJ) 79 annotated slices
selection = np.arange(79)+171
image = image[selection]

label_1 = label_1[selection]
label_2 = label_2[selection]
label_3 = label_3[selection]
label_4 = label_4[selection]

anisotropy=(20, 5, 5)
normalize = normalize_tanh
normalize_scale_neg = 20
normalize_scale_pos = 20
label_dt_1 = label_dt((label_1*1.0).astype(np.float32), anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)
label_dt_2 = label_dt((label_2*1.0).astype(np.float32), anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)
label_dt_3 = label_dt((label_3*1.0).astype(np.float32), anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)
label_dt_4 = label_dt((label_4*1.0).astype(np.float32), anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)

print(((label_dt_1 > 0) == (label_1 == 1)).all())
print(((label_dt_2 > 0) == (label_2 == 1)).all())
print(((label_dt_3 > 0) == (label_3 == 1)).all())
print(((label_dt_4 > 0) == (label_4 == 1)).all())

# Normalize
image_min = image.min()
image_max = image.max()
image_normalized_f32 = np.array((image - image_min) / (image_max - image_min)).astype(np.float32)
image_normalized_f16 = image_normalized_f32.astype(np.float16)
print(image_normalized_f32.min(), image_normalized_f32.max())

if image_normalized_f32.min() != 0 or image_normalized_f32.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)

# Split
train_selection = np.arange(50)
test_selection = np.arange(50) + 29

train_image_normalized_f32 = image_normalized_f32[train_selection]
train_image_normalized_f16 = image_normalized_f16[train_selection]
train_label_dt_1 = label_dt_1[train_selection]
train_label_dt_2 = label_dt_2[train_selection]
train_label_dt_3 = label_dt_3[train_selection]
train_label_dt_4 = label_dt_4[train_selection]

test_image_normalized_f32 = image_normalized_f32[test_selection]
test_image_normalized_f16 = image_normalized_f16[test_selection]
test_label_dt_1 = label_dt_1[test_selection]
test_label_dt_2 = label_dt_2[test_selection]
test_label_dt_3 = label_dt_3[test_selection]
test_label_dt_4 = label_dt_4[test_selection]

train_labels_one_hot = np.stack([train_label_dt_1, train_label_dt_2, train_label_dt_3, train_label_dt_4], axis=-1)

test_labels_one_hot = np.stack([test_label_dt_1, test_label_dt_2, test_label_dt_3, test_label_dt_4], axis=-1)

# Save
np.save("/home/cyril/Documents/Data/I3_79_4/TRAIN_IMAGE_NORMALIZED_F32.npy", train_image_normalized_f32)
np.save("/home/cyril/Documents/Data/I3_79_4/TRAIN_IMAGE_NORMALIZED_F16.npy", train_image_normalized_f16)
np.save("/home/cyril/Documents/Data/I3_79_4/TRAIN_LABELS_DT.npy", train_labels_one_hot)

np.save("/home/cyril/Documents/Data/I3_79_4/TEST_IMAGE_NORMALIZED_F32.npy", test_image_normalized_f32)
np.save("/home/cyril/Documents/Data/I3_79_4/TEST_IMAGE_NORMALIZED_F16.npy", test_image_normalized_f16)
np.save("/home/cyril/Documents/Data/I3_79_4/TEST_LABELS_DT.npy", test_labels_one_hot)
