# ------------------------------------------------------------ #
#
# File : data/preparation_LW4_9.py
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


# Check
def check_label(label):
    for z in range(label.shape[0]):
        if label[z].min() == label[z].max():
            print("ERROR, EMPTY LABEL SLICE")
            sys.exit(os.EX_DATAERR)

check_label(lw4_label_1)
check_label(lw4_label_2)
check_label(lw4_label_3)
check_label(lw4_label_4)
check_label(lw4_label_5)
check_label(lw4_label_6)
check_label(lw4_label_7)
check_label(lw4_label_8)
check_label(lw4_label_9)


# Normalize
lw4_image_min = lw4_image.min()
lw4_image_max = lw4_image.max()
lw4_image_normalized_f32 = np.array((lw4_image - lw4_image_min) / (lw4_image_max - lw4_image_min)).astype(np.float32)
lw4_image_normalized_f16 = lw4_image_normalized_f32.astype(np.float16)

if lw4_image_normalized_f32.min() != 0 or lw4_image_normalized_f32.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)


# Split
train_lw4_selection = np.arange(20)
valid_lw4_selection = np.arange(10) + 20
test_lw4_selection = np.arange(10) + 30

train_lw4_image_normalized_f32 = lw4_image_normalized_f32[train_lw4_selection]
train_lw4_image_normalized_f16 = lw4_image_normalized_f16[train_lw4_selection]
train_lw4_label_1 = lw4_label_1[train_lw4_selection]
train_lw4_label_2 = lw4_label_2[train_lw4_selection]
train_lw4_label_3 = lw4_label_3[train_lw4_selection]
train_lw4_label_4 = lw4_label_4[train_lw4_selection]
train_lw4_label_5 = lw4_label_5[train_lw4_selection]
train_lw4_label_6 = lw4_label_6[train_lw4_selection]
train_lw4_label_7 = lw4_label_7[train_lw4_selection]
train_lw4_label_8 = lw4_label_8[train_lw4_selection]
train_lw4_label_9 = lw4_label_9[train_lw4_selection]

train_lw4_label_1_indexes = np.argwhere(train_lw4_label_1 == 1)
train_lw4_label_2_indexes = np.argwhere(train_lw4_label_2 == 1)
train_lw4_label_3_indexes = np.argwhere(train_lw4_label_3 == 1)
train_lw4_label_4_indexes = np.argwhere(train_lw4_label_4 == 1)
train_lw4_label_5_indexes = np.argwhere(train_lw4_label_5 == 1)
train_lw4_label_6_indexes = np.argwhere(train_lw4_label_6 == 1)
train_lw4_label_7_indexes = np.argwhere(train_lw4_label_7 == 1)
train_lw4_label_8_indexes = np.argwhere(train_lw4_label_8 == 1)
train_lw4_label_9_indexes = np.argwhere(train_lw4_label_9 == 1)

valid_lw4_image_normalized_f32 = lw4_image_normalized_f32[valid_lw4_selection]
valid_lw4_image_normalized_f16 = lw4_image_normalized_f16[valid_lw4_selection]
valid_lw4_label_1 = lw4_label_1[valid_lw4_selection]
valid_lw4_label_2 = lw4_label_2[valid_lw4_selection]
valid_lw4_label_3 = lw4_label_3[valid_lw4_selection]
valid_lw4_label_4 = lw4_label_4[valid_lw4_selection]
valid_lw4_label_5 = lw4_label_5[valid_lw4_selection]
valid_lw4_label_6 = lw4_label_6[valid_lw4_selection]
valid_lw4_label_7 = lw4_label_7[valid_lw4_selection]
valid_lw4_label_8 = lw4_label_8[valid_lw4_selection]
valid_lw4_label_9 = lw4_label_9[valid_lw4_selection]

valid_lw4_label_1_indexes = np.argwhere(valid_lw4_label_1 == 1)
valid_lw4_label_2_indexes = np.argwhere(valid_lw4_label_2 == 1)
valid_lw4_label_3_indexes = np.argwhere(valid_lw4_label_3 == 1)
valid_lw4_label_4_indexes = np.argwhere(valid_lw4_label_4 == 1)
valid_lw4_label_5_indexes = np.argwhere(valid_lw4_label_5 == 1)
valid_lw4_label_6_indexes = np.argwhere(valid_lw4_label_6 == 1)
valid_lw4_label_7_indexes = np.argwhere(valid_lw4_label_7 == 1)
valid_lw4_label_8_indexes = np.argwhere(valid_lw4_label_8 == 1)
valid_lw4_label_9_indexes = np.argwhere(valid_lw4_label_9 == 1)

test_lw4_image_normalized_f32 = lw4_image_normalized_f32[test_lw4_selection]
test_lw4_image_normalized_f16 = lw4_image_normalized_f16[test_lw4_selection]
test_lw4_label_1 = lw4_label_1[test_lw4_selection]
test_lw4_label_2 = lw4_label_2[test_lw4_selection]
test_lw4_label_3 = lw4_label_3[test_lw4_selection]
test_lw4_label_4 = lw4_label_4[test_lw4_selection]
test_lw4_label_5 = lw4_label_5[test_lw4_selection]
test_lw4_label_6 = lw4_label_6[test_lw4_selection]
test_lw4_label_7 = lw4_label_7[test_lw4_selection]
test_lw4_label_8 = lw4_label_8[test_lw4_selection]
test_lw4_label_9 = lw4_label_9[test_lw4_selection]

test_lw4_label_1_indexes = np.argwhere(test_lw4_label_1 == 1)
test_lw4_label_2_indexes = np.argwhere(test_lw4_label_2 == 1)
test_lw4_label_3_indexes = np.argwhere(test_lw4_label_3 == 1)
test_lw4_label_4_indexes = np.argwhere(test_lw4_label_4 == 1)
test_lw4_label_5_indexes = np.argwhere(test_lw4_label_5 == 1)
test_lw4_label_6_indexes = np.argwhere(test_lw4_label_6 == 1)
test_lw4_label_7_indexes = np.argwhere(test_lw4_label_7 == 1)
test_lw4_label_8_indexes = np.argwhere(test_lw4_label_8 == 1)
test_lw4_label_9_indexes = np.argwhere(test_lw4_label_9 == 1)

# Save
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_IMAGE_NORMALIZED_F32.npy", train_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_IMAGE_NORMALIZED_F16.npy", train_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_1.npy", train_lw4_label_1)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_2.npy", train_lw4_label_2)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_3.npy", train_lw4_label_3)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_4.npy", train_lw4_label_4)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_5.npy", train_lw4_label_5)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_6.npy", train_lw4_label_6)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_7.npy", train_lw4_label_7)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_8.npy", train_lw4_label_8)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_9.npy", train_lw4_label_9)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_1_INDEXES.npy", train_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_2_INDEXES.npy", train_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_3_INDEXES.npy", train_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_4_INDEXES.npy", train_lw4_label_4_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_5_INDEXES.npy", train_lw4_label_5_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_6_INDEXES.npy", train_lw4_label_6_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_7_INDEXES.npy", train_lw4_label_7_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_8_INDEXES.npy", train_lw4_label_8_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TRAIN_LW4_LABEL_9_INDEXES.npy", train_lw4_label_9_indexes)

np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_IMAGE_NORMALIZED_F32.npy", valid_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_IMAGE_NORMALIZED_F16.npy", valid_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_1.npy", valid_lw4_label_1)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_2.npy", valid_lw4_label_2)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_3.npy", valid_lw4_label_3)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_4.npy", valid_lw4_label_4)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_5.npy", valid_lw4_label_5)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_6.npy", valid_lw4_label_6)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_7.npy", valid_lw4_label_7)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_8.npy", valid_lw4_label_8)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_9.npy", valid_lw4_label_9)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_1_INDEXES.npy", valid_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_2_INDEXES.npy", valid_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_3_INDEXES.npy", valid_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_4_INDEXES.npy", valid_lw4_label_4_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_5_INDEXES.npy", valid_lw4_label_5_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_6_INDEXES.npy", valid_lw4_label_6_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_7_INDEXES.npy", valid_lw4_label_7_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_8_INDEXES.npy", valid_lw4_label_8_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/VALID_LW4_LABEL_9_INDEXES.npy", valid_lw4_label_9_indexes)

np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_IMAGE_NORMALIZED_F32.npy", test_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_IMAGE_NORMALIZED_F16.npy", test_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_1.npy", test_lw4_label_1)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_2.npy", test_lw4_label_2)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_3.npy", test_lw4_label_3)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_4.npy", test_lw4_label_4)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_5.npy", test_lw4_label_5)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_6.npy", test_lw4_label_6)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_7.npy", test_lw4_label_7)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_8.npy", test_lw4_label_8)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_9.npy", test_lw4_label_9)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_1_INDEXES.npy", test_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_2_INDEXES.npy", test_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_3_INDEXES.npy", test_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_4_INDEXES.npy", test_lw4_label_4_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_5_INDEXES.npy", test_lw4_label_5_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_6_INDEXES.npy", test_lw4_label_6_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_7_INDEXES.npy", test_lw4_label_7_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_8_INDEXES.npy", test_lw4_label_8_indexes)
np.save("/home/cyril/Documents/Data/LW4_40_9/TEST_LW4_LABEL_9_INDEXES.npy", test_lw4_label_9_indexes)
