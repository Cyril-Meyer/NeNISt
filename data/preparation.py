# ------------------------------------------------------------ #
#
# File : data/preparation.py
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
# Label 4 : Nuclear envelope (only I3)
# Label 5 : Endosome (only LW4)

i3_image = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/I3/i3.tif"))
i3_label_1 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/I3/Labels_i3-mitos_1-500.tif"))
i3_label_2 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/I3/Labels_i3_MembraneCellule_1-250.tif"))
i3_label_3 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/I3/Labels_i3_Reticulum_172-251.tif"))
i3_label_4 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/I3/Labels_i3_Noyau_1-250.tif"))

lw4_image = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/LW4/LW4-600.tif"))
lw4_label_1 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/LW4/Labels_LW4-600_All-Step40_mito.tif"))
lw4_label_2 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/LW4/Labels_LW4-600_All-Step40_cell.tif"))
lw4_label_3 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/LW4/Labels_LW4-600_1-40_81-120_Reti.tif"))
lw4_label_5 = np.array(
    skimage.io.imread("/home/cyril/Documents/Data/MEYER_ISBI_2021/LW4/Labels_LW4-600_All-Step40_endo.tif"))


# Check shape
if not (i3_image.shape == i3_label_1.shape == i3_label_2.shape == i3_label_3.shape == i3_label_4.shape):
    print("ERROR, IMAGE SHAPE")
    sys.exit(os.EX_DATAERR)


if not (lw4_image.shape == lw4_label_1.shape == lw4_label_2.shape == lw4_label_3.shape == lw4_label_5.shape):
    print("ERROR, IMAGE SHAPE")
    sys.exit(os.EX_DATAERR)


# Select
# [172, 250] (1 based in imageJ) 79 annotated slices
selection = np.arange(79)+171
i3_image = i3_image[selection]
i3_label_1 = i3_label_1[selection]
i3_label_2 = i3_label_2[selection]
i3_label_3 = i3_label_3[selection]
i3_label_4 = i3_label_4[selection]

# [1, 41] and [81, 120] (1 based in imageJ) 80 annotated slices
selection = np.concatenate((np.arange(40)+0, np.arange(40)+80))
lw4_image = lw4_image[selection]
lw4_label_1 = lw4_label_1[selection]
lw4_label_2 = lw4_label_2[selection]
lw4_label_3 = lw4_label_3[selection]
lw4_label_5 = lw4_label_5[selection]


# Check
def check_label(label):
    for z in range(label.shape[0]):
        if label[z].min() == label[z].max():
            print("ERROR, EMPTY LABEL SLICE")
            sys.exit(os.EX_DATAERR)


check_label(i3_label_1)
check_label(i3_label_2)
check_label(i3_label_3)
check_label(i3_label_4)

check_label(lw4_label_1)
check_label(lw4_label_2)
check_label(lw4_label_3)
check_label(lw4_label_5)


# Normalize
i3_image_min = i3_image.min()
i3_image_max = i3_image.max()
i3_image_normalized_f32 = np.array((i3_image - i3_image_min) / (i3_image_max - i3_image_min)).astype(np.float32)
i3_image_normalized_f16 = i3_image_normalized_f32.astype(np.float16)

if i3_image_normalized_f32.min() != 0 or i3_image_normalized_f32.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)

lw4_image_min = lw4_image.min()
lw4_image_max = lw4_image.max()
lw4_image_normalized_f32 = np.array((lw4_image - lw4_image_min) / (lw4_image_max - lw4_image_min)).astype(np.float32)
lw4_image_normalized_f16 = lw4_image_normalized_f32.astype(np.float16)

if lw4_image_normalized_f32.min() != 0 or lw4_image_normalized_f32.max() != 1:
    print("ERROR, INVALID NORMALIZATION")
    sys.exit(os.EX_SOFTWARE)


# Split
train_i3_selection = np.arange(40)
valid_i3_selection = np.arange(20) + 40
test_i3_selection = np.arange(19) + 60

train_i3_image_normalized_f32 = i3_image_normalized_f32[train_i3_selection]
train_i3_image_normalized_f16 = i3_image_normalized_f16[train_i3_selection]
train_i3_label_1 = i3_label_1[train_i3_selection]
train_i3_label_2 = i3_label_2[train_i3_selection]
train_i3_label_3 = i3_label_3[train_i3_selection]
train_i3_label_4 = i3_label_4[train_i3_selection]
train_i3_label_1_indexes = np.argwhere(train_i3_label_1 == 1)
train_i3_label_2_indexes = np.argwhere(train_i3_label_2 == 1)
train_i3_label_3_indexes = np.argwhere(train_i3_label_3 == 1)
train_i3_label_4_indexes = np.argwhere(train_i3_label_4 == 1)

valid_i3_image_normalized_f32 = i3_image_normalized_f32[valid_i3_selection]
valid_i3_image_normalized_f16 = i3_image_normalized_f16[valid_i3_selection]
valid_i3_label_1 = i3_label_1[valid_i3_selection]
valid_i3_label_2 = i3_label_2[valid_i3_selection]
valid_i3_label_3 = i3_label_3[valid_i3_selection]
valid_i3_label_4 = i3_label_4[valid_i3_selection]
valid_i3_label_1_indexes = np.argwhere(valid_i3_label_1 == 1)
valid_i3_label_2_indexes = np.argwhere(valid_i3_label_2 == 1)
valid_i3_label_3_indexes = np.argwhere(valid_i3_label_3 == 1)
valid_i3_label_4_indexes = np.argwhere(valid_i3_label_4 == 1)

test_i3_image_normalized_f32 = i3_image_normalized_f32[test_i3_selection]
test_i3_image_normalized_f16 = i3_image_normalized_f16[test_i3_selection]
test_i3_label_1 = i3_label_1[test_i3_selection]
test_i3_label_2 = i3_label_2[test_i3_selection]
test_i3_label_3 = i3_label_3[test_i3_selection]
test_i3_label_4 = i3_label_4[test_i3_selection]
test_i3_label_1_indexes = np.argwhere(test_i3_label_1 == 1)
test_i3_label_2_indexes = np.argwhere(test_i3_label_2 == 1)
test_i3_label_3_indexes = np.argwhere(test_i3_label_3 == 1)
test_i3_label_4_indexes = np.argwhere(test_i3_label_4 == 1)


train_lw4_selection = np.arange(40)
valid_lw4_selection = np.arange(20) + 40
test_lw4_selection = np.arange(20) + 60

train_lw4_image_normalized_f32 = lw4_image_normalized_f32[train_lw4_selection]
train_lw4_image_normalized_f16 = lw4_image_normalized_f16[train_lw4_selection]
train_lw4_label_1 = lw4_label_1[train_lw4_selection]
train_lw4_label_2 = lw4_label_2[train_lw4_selection]
train_lw4_label_3 = lw4_label_3[train_lw4_selection]
train_lw4_label_5 = lw4_label_5[train_lw4_selection]
train_lw4_label_1_indexes = np.argwhere(train_lw4_label_1 == 1)
train_lw4_label_2_indexes = np.argwhere(train_lw4_label_2 == 1)
train_lw4_label_3_indexes = np.argwhere(train_lw4_label_3 == 1)
train_lw4_label_5_indexes = np.argwhere(train_lw4_label_5 == 1)

valid_lw4_image_normalized_f32 = lw4_image_normalized_f32[valid_lw4_selection]
valid_lw4_image_normalized_f16 = lw4_image_normalized_f16[valid_lw4_selection]
valid_lw4_label_1 = lw4_label_1[valid_lw4_selection]
valid_lw4_label_2 = lw4_label_2[valid_lw4_selection]
valid_lw4_label_3 = lw4_label_3[valid_lw4_selection]
valid_lw4_label_5 = lw4_label_5[valid_lw4_selection]
valid_lw4_label_1_indexes = np.argwhere(valid_lw4_label_1 == 1)
valid_lw4_label_2_indexes = np.argwhere(valid_lw4_label_2 == 1)
valid_lw4_label_3_indexes = np.argwhere(valid_lw4_label_3 == 1)
valid_lw4_label_5_indexes = np.argwhere(valid_lw4_label_5 == 1)

test_lw4_image_normalized_f32 = lw4_image_normalized_f32[test_lw4_selection]
test_lw4_image_normalized_f16 = lw4_image_normalized_f16[test_lw4_selection]
test_lw4_label_1 = lw4_label_1[test_lw4_selection]
test_lw4_label_2 = lw4_label_2[test_lw4_selection]
test_lw4_label_3 = lw4_label_3[test_lw4_selection]
test_lw4_label_5 = lw4_label_5[test_lw4_selection]
test_lw4_label_1_indexes = np.argwhere(test_lw4_label_1 == 1)
test_lw4_label_2_indexes = np.argwhere(test_lw4_label_2 == 1)
test_lw4_label_3_indexes = np.argwhere(test_lw4_label_3 == 1)
test_lw4_label_5_indexes = np.argwhere(test_lw4_label_5 == 1)

# Save
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_IMAGE_NORMALIZED_F32.npy", train_i3_image_normalized_f32)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_IMAGE_NORMALIZED_F16.npy", train_i3_image_normalized_f16)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_1.npy", train_i3_label_1)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_2.npy", train_i3_label_2)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_3.npy", train_i3_label_3)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_4.npy", train_i3_label_4)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_1_INDEXES.npy", train_i3_label_1_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_2_INDEXES.npy", train_i3_label_2_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_3_INDEXES.npy", train_i3_label_3_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_I3_LABEL_4_INDEXES.npy", train_i3_label_4_indexes)

np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_IMAGE_NORMALIZED_F32.npy", valid_i3_image_normalized_f32)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_IMAGE_NORMALIZED_F16.npy", valid_i3_image_normalized_f16)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_1.npy", valid_i3_label_1)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_2.npy", valid_i3_label_2)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_3.npy", valid_i3_label_3)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_4.npy", valid_i3_label_4)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_1_INDEXES.npy", valid_i3_label_1_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_2_INDEXES.npy", valid_i3_label_2_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_3_INDEXES.npy", valid_i3_label_3_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_I3_LABEL_4_INDEXES.npy", valid_i3_label_4_indexes)

np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_IMAGE_NORMALIZED_F32.npy", test_i3_image_normalized_f32)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_IMAGE_NORMALIZED_F16.npy", test_i3_image_normalized_f16)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_1.npy", test_i3_label_1)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_2.npy", test_i3_label_2)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_3.npy", test_i3_label_3)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_4.npy", test_i3_label_4)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_1_INDEXES.npy", test_i3_label_1_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_2_INDEXES.npy", test_i3_label_2_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_3_INDEXES.npy", test_i3_label_3_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_I3_LABEL_4_INDEXES.npy", test_i3_label_4_indexes)


np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_IMAGE_NORMALIZED_F32.npy", train_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_IMAGE_NORMALIZED_F16.npy", train_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_1.npy", train_lw4_label_1)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_2.npy", train_lw4_label_2)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_3.npy", train_lw4_label_3)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_5.npy", train_lw4_label_5)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_1_INDEXES.npy", train_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_2_INDEXES.npy", train_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_3_INDEXES.npy", train_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TRAIN_LW4_LABEL_5_INDEXES.npy", train_lw4_label_5_indexes)

np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_IMAGE_NORMALIZED_F32.npy", valid_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_IMAGE_NORMALIZED_F16.npy", valid_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_1.npy", valid_lw4_label_1)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_2.npy", valid_lw4_label_2)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_3.npy", valid_lw4_label_3)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_5.npy", valid_lw4_label_5)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_1_INDEXES.npy", valid_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_2_INDEXES.npy", valid_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_3_INDEXES.npy", valid_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/VALID_LW4_LABEL_5_INDEXES.npy", valid_lw4_label_5_indexes)

np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_IMAGE_NORMALIZED_F32.npy", test_lw4_image_normalized_f32)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_IMAGE_NORMALIZED_F16.npy", test_lw4_image_normalized_f16)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_1.npy", test_lw4_label_1)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_2.npy", test_lw4_label_2)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_3.npy", test_lw4_label_3)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_5.npy", test_lw4_label_5)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_1_INDEXES.npy", test_lw4_label_1_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_2_INDEXES.npy", test_lw4_label_2_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_3_INDEXES.npy", test_lw4_label_3_indexes)
np.save("/home/cyril/Documents/Data/MEYER_ISBI_2021/TEST_LW4_LABEL_5_INDEXES.npy", test_lw4_label_5_indexes)
