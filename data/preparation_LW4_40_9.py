# ------------------------------------------------------------ #
#
# File : data/preparation_LW4_40_9.py
# Authors : CM
# Read, check, prepare and export data.
# - Only the annotated slices are kept.
# - Selected slices are checked (check if shape match and if label is not missing).
# - Images are normalized between 0 and 1.
# - Dataset is split between train, valid and test.
# - Images and annotations are saved as numpy arrays.
# - Annotations are also saved as indexes arrays.
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
# Label 4 : Endosome
# Label 5 : Nuclear membrane
# Label 6 : Nucleus
# Label 7 : Nucleus HeteroChro
# Label 8 : Nucleus Rest
# Label 9 : Golgi

lw4_image = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/LW4-600.tif"))
lw4_label_1 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_All-Step40_mito.tif"))
lw4_label_2 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_All-Step40_cell.tif"))
lw4_label_3 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_1-40_81-120_Reti.tif"))
lw4_label_4 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_All-Step40_endo.tif"))
lw4_label_5 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_1-40_Nuc_membrane.tif"))
lw4_label_6 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_1-40_Nucleus.tif"))
lw4_label_7 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_1-40_Nuc_heteroChro.tif"))
lw4_label_8 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_1-40_Nuc_rest.tif"))
lw4_label_9 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/LW4/Labels_LW4-600_1-40_Golgi.tif"))

if not (lw4_image.shape == lw4_label_1.shape == lw4_label_2.shape == lw4_label_3.shape == lw4_label_4.shape == lw4_label_5.shape == lw4_label_6.shape == lw4_label_7.shape == lw4_label_8.shape == lw4_label_9.shape):
    print("ERROR, IMAGE SHAPE")
    sys.exit(os.EX_DATAERR)

selection = np.arange(40)
lw4_image = lw4_image[selection]

lw4_label_1 = lw4_label_1[selection]
lw4_label_2 = lw4_label_2[selection]
lw4_label_3 = lw4_label_3[selection]
lw4_label_4 = lw4_label_4[selection]
lw4_label_5 = lw4_label_5[selection]
lw4_label_6 = lw4_label_6[selection]
lw4_label_7 = lw4_label_7[selection]
lw4_label_8 = lw4_label_8[selection]
lw4_label_9 = lw4_label_9[selection]

lw4_label_dt_1 = label_dt((lw4_label_1*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_2 = label_dt((lw4_label_2*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_3 = label_dt((lw4_label_3*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_4 = label_dt((lw4_label_4*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_5 = label_dt((lw4_label_5*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_6 = label_dt((lw4_label_6*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_7 = label_dt((lw4_label_7*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_8 = label_dt((lw4_label_8*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)
lw4_label_dt_9 = label_dt((lw4_label_9*1.0).astype(np.float32), normalize=normalize_tanh, normalize_scale=50.0)

print(((lw4_label_dt_1 > 0) == (lw4_label_1 == 1)).all())
print(((lw4_label_dt_2 > 0) == (lw4_label_2 == 1)).all())
print(((lw4_label_dt_3 > 0) == (lw4_label_3 == 1)).all())
print(((lw4_label_dt_4 > 0) == (lw4_label_4 == 1)).all())
print(((lw4_label_dt_5 > 0) == (lw4_label_5 == 1)).all())
print(((lw4_label_dt_6 > 0) == (lw4_label_6 == 1)).all())
print(((lw4_label_dt_7 > 0) == (lw4_label_7 == 1)).all())
print(((lw4_label_dt_8 > 0) == (lw4_label_8 == 1)).all())
print(((lw4_label_dt_9 > 0) == (lw4_label_9 == 1)).all())

# Normalize
lw4_image_min = lw4_image.min()
lw4_image_max = lw4_image.max()
lw4_image_normalized_f32 = np.array((lw4_image - lw4_image_min) / (lw4_image_max - lw4_image_min)).astype(np.float32)
lw4_image_normalized_f16 = lw4_image_normalized_f32.astype(np.float16)
print(lw4_image_normalized_f32.min(), lw4_image_normalized_f32.max())

if lw4_image_normalized_f32.min() != 0 or lw4_image_normalized_f32.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)

# Standardize
lw4_image_mean = lw4_image.mean()
lw4_image_std = lw4_image.std()
lw4_image_standardized_f32 = np.array((lw4_image - lw4_image_mean) / lw4_image_std).astype(np.float32)
lw4_image_standardized_f32 = lw4_image_standardized_f32 / max(abs(lw4_image_standardized_f32.min()), abs(lw4_image_standardized_f32.max()))
lw4_image_standardized_f16 = lw4_image_standardized_f32.astype(np.float16)
print(lw4_image_standardized_f32.min(), lw4_image_standardized_f32.max())


# Split
train_lw4_selection = np.arange(24)
test_lw4_selection = np.arange(16) + 24

train_lw4_image_normalized_f32 = lw4_image_normalized_f32[train_lw4_selection]
train_lw4_image_normalized_f16 = lw4_image_normalized_f16[train_lw4_selection]
train_lw4_image_standardized_f32 = lw4_image_standardized_f32[train_lw4_selection]
train_lw4_image_standardized_f16 = lw4_image_standardized_f16[train_lw4_selection]
train_lw4_label_dt_1 = lw4_label_dt_1[train_lw4_selection]
train_lw4_label_dt_2 = lw4_label_dt_2[train_lw4_selection]
train_lw4_label_dt_3 = lw4_label_dt_3[train_lw4_selection]
train_lw4_label_dt_4 = lw4_label_dt_4[train_lw4_selection]
train_lw4_label_dt_5 = lw4_label_dt_5[train_lw4_selection]
train_lw4_label_dt_6 = lw4_label_dt_6[train_lw4_selection]
train_lw4_label_dt_7 = lw4_label_dt_7[train_lw4_selection]
train_lw4_label_dt_8 = lw4_label_dt_8[train_lw4_selection]
train_lw4_label_dt_9 = lw4_label_dt_9[train_lw4_selection]

train_lw4_label_1_indexes = np.argwhere(train_lw4_label_dt_1 > 0)
train_lw4_label_2_indexes = np.argwhere(train_lw4_label_dt_2 > 0)
train_lw4_label_3_indexes = np.argwhere(train_lw4_label_dt_3 > 0)
train_lw4_label_4_indexes = np.argwhere(train_lw4_label_dt_4 > 0)
train_lw4_label_5_indexes = np.argwhere(train_lw4_label_dt_5 > 0)
train_lw4_label_6_indexes = np.argwhere(train_lw4_label_dt_6 > 0)
train_lw4_label_7_indexes = np.argwhere(train_lw4_label_dt_7 > 0)
train_lw4_label_8_indexes = np.argwhere(train_lw4_label_dt_8 > 0)
train_lw4_label_9_indexes = np.argwhere(train_lw4_label_dt_9 > 0)

test_lw4_image_normalized_f32 = lw4_image_normalized_f32[test_lw4_selection]
test_lw4_image_normalized_f16 = lw4_image_normalized_f16[test_lw4_selection]
test_lw4_image_standardized_f32 = lw4_image_standardized_f32[test_lw4_selection]
test_lw4_image_standardized_f16 = lw4_image_standardized_f16[test_lw4_selection]
test_lw4_label_dt_1 = lw4_label_dt_1[test_lw4_selection]
test_lw4_label_dt_2 = lw4_label_dt_2[test_lw4_selection]
test_lw4_label_dt_3 = lw4_label_dt_3[test_lw4_selection]
test_lw4_label_dt_4 = lw4_label_dt_4[test_lw4_selection]
test_lw4_label_dt_5 = lw4_label_dt_5[test_lw4_selection]
test_lw4_label_dt_6 = lw4_label_dt_6[test_lw4_selection]
test_lw4_label_dt_7 = lw4_label_dt_7[test_lw4_selection]
test_lw4_label_dt_8 = lw4_label_dt_8[test_lw4_selection]
test_lw4_label_dt_9 = lw4_label_dt_9[test_lw4_selection]

test_lw4_label_1_indexes = np.argwhere(test_lw4_label_dt_1 > 0)
test_lw4_label_2_indexes = np.argwhere(test_lw4_label_dt_2 > 0)
test_lw4_label_3_indexes = np.argwhere(test_lw4_label_dt_3 > 0)
test_lw4_label_4_indexes = np.argwhere(test_lw4_label_dt_4 > 0)
test_lw4_label_5_indexes = np.argwhere(test_lw4_label_dt_5 > 0)
test_lw4_label_6_indexes = np.argwhere(test_lw4_label_dt_6 > 0)
test_lw4_label_7_indexes = np.argwhere(test_lw4_label_dt_7 > 0)
test_lw4_label_8_indexes = np.argwhere(test_lw4_label_dt_8 > 0)
test_lw4_label_9_indexes = np.argwhere(test_lw4_label_dt_9 > 0)

train_lw4_labels_one_hot = np.stack([train_lw4_label_dt_1, train_lw4_label_dt_2, train_lw4_label_dt_3, train_lw4_label_dt_4, train_lw4_label_dt_5, train_lw4_label_dt_6, train_lw4_label_dt_7, train_lw4_label_dt_8, train_lw4_label_dt_9], axis=-1)

test_lw4_labels_one_hot = np.stack([test_lw4_label_dt_1, test_lw4_label_dt_2, test_lw4_label_dt_3, test_lw4_label_dt_4, test_lw4_label_dt_5, test_lw4_label_dt_6, test_lw4_label_dt_7, test_lw4_label_dt_8, test_lw4_label_dt_9], axis=-1)

# Save
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_IMAGE_NORMALIZED_F32.npy", train_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_IMAGE_NORMALIZED_F16.npy", train_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_IMAGE_STANDARDIZED_F32.npy", train_lw4_image_standardized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_IMAGE_STANDARDIZED_F16.npy", train_lw4_image_standardized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABELS_DT.npy", train_lw4_labels_one_hot)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_1_INDEXES.npy", train_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_2_INDEXES.npy", train_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_3_INDEXES.npy", train_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_4_INDEXES.npy", train_lw4_label_4_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_5_INDEXES.npy", train_lw4_label_5_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_6_INDEXES.npy", train_lw4_label_6_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_7_INDEXES.npy", train_lw4_label_7_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_8_INDEXES.npy", train_lw4_label_8_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_9_INDEXES.npy", train_lw4_label_9_indexes)

np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_IMAGE_NORMALIZED_F32.npy", test_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_IMAGE_NORMALIZED_F16.npy", test_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_IMAGE_STANDARDIZED_F32.npy", test_lw4_image_standardized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_IMAGE_STANDARDIZED_F16.npy", test_lw4_image_standardized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABELS_DT.npy", test_lw4_labels_one_hot)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_1_INDEXES.npy", test_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_2_INDEXES.npy", test_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_3_INDEXES.npy", test_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_4_INDEXES.npy", test_lw4_label_4_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_5_INDEXES.npy", test_lw4_label_5_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_6_INDEXES.npy", test_lw4_label_6_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_7_INDEXES.npy", test_lw4_label_7_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_8_INDEXES.npy", test_lw4_label_8_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_9_INDEXES.npy", test_lw4_label_9_indexes)

hf = h5py.File('/home/cyril/Documents/Data/LW4_40_9/LW4_40_9.h5', 'w')
hf.create_dataset('train_image_normalized', data=train_lw4_image_normalized_f32)
hf.create_dataset('test_image_normalized', data=test_lw4_image_normalized_f32)
hf.create_dataset('train_image_standardized', data=train_lw4_image_standardized_f32)
hf.create_dataset('test_image_standardized', data=test_lw4_image_standardized_f32)
hf.create_dataset('train_label_1', data=train_lw4_label_dt_1)
hf.create_dataset('train_label_2', data=train_lw4_label_dt_2)
hf.create_dataset('train_label_3', data=train_lw4_label_dt_3)
hf.create_dataset('train_label_4', data=train_lw4_label_dt_4)
hf.create_dataset('train_label_5', data=train_lw4_label_dt_5)
hf.create_dataset('train_label_6', data=train_lw4_label_dt_6)
hf.create_dataset('train_label_7', data=train_lw4_label_dt_7)
hf.create_dataset('train_label_8', data=train_lw4_label_dt_8)
hf.create_dataset('train_label_9', data=train_lw4_label_dt_9)
hf.create_dataset('test_label_1', data=test_lw4_label_dt_1)
hf.create_dataset('test_label_2', data=test_lw4_label_dt_2)
hf.create_dataset('test_label_3', data=test_lw4_label_dt_3)
hf.create_dataset('test_label_4', data=test_lw4_label_dt_4)
hf.create_dataset('test_label_5', data=test_lw4_label_dt_5)
hf.create_dataset('test_label_6', data=test_lw4_label_dt_6)
hf.create_dataset('test_label_7', data=test_lw4_label_dt_7)
hf.create_dataset('test_label_8', data=test_lw4_label_dt_8)
hf.create_dataset('test_label_9', data=test_lw4_label_dt_9)
hf.close()
