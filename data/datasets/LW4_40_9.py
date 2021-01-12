# ------------------------------------------------------------ #
#
# File : data/datasets/LW4_9.py
# Authors : CM
# Easy LW4 40 slice 9 class dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np

if(os.uname()[1] == 'lythandas'):
    DATA_FOLDER = "/home/cyril/Documents/Data/LW4_40_9/"
else:
    DATA_FOLDER = "/b/home/miv/cmeyer/DATA/LW4_40_9/"

train_image_normalized_f32 = np.load(DATA_FOLDER + "TRAIN_LW4_IMAGE_NORMALIZED_F32.npy")
train_image_normalized_f16 = np.load(DATA_FOLDER + "TRAIN_LW4_IMAGE_NORMALIZED_F16.npy")
train_label_1 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_1.npy")
train_label_2 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_2.npy")
train_label_3 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_3.npy")
train_label_4 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_4.npy")
train_label_5 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_5.npy")
train_label_6 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_6.npy")
train_label_7 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_7.npy")
train_label_8 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_8.npy")
train_label_9 = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_9.npy")
train_label_1_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_1_INDEXES.npy")
train_label_2_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_2_INDEXES.npy")
train_label_3_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_3_INDEXES.npy")
train_label_4_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_4_INDEXES.npy")
train_label_5_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_5_INDEXES.npy")
train_label_6_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_6_INDEXES.npy")
train_label_7_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_7_INDEXES.npy")
train_label_8_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_8_INDEXES.npy")
train_label_9_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_9_INDEXES.npy")

valid_image_normalized_f32 = np.load(DATA_FOLDER + "VALID_LW4_IMAGE_NORMALIZED_F32.npy")
valid_image_normalized_f16 = np.load(DATA_FOLDER + "VALID_LW4_IMAGE_NORMALIZED_F16.npy")
valid_label_1 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_1.npy")
valid_label_2 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_2.npy")
valid_label_3 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_3.npy")
valid_label_4 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_4.npy")
valid_label_5 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_5.npy")
valid_label_6 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_6.npy")
valid_label_7 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_7.npy")
valid_label_8 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_8.npy")
valid_label_9 = np.load(DATA_FOLDER + "VALID_LW4_LABEL_9.npy")
valid_label_1_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_1_INDEXES.npy")
valid_label_2_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_2_INDEXES.npy")
valid_label_3_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_3_INDEXES.npy")
valid_label_4_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_4_INDEXES.npy")
valid_label_5_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_5_INDEXES.npy")
valid_label_6_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_6_INDEXES.npy")
valid_label_7_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_7_INDEXES.npy")
valid_label_8_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_8_INDEXES.npy")
valid_label_9_indexes = np.load(DATA_FOLDER + "VALID_LW4_LABEL_9_INDEXES.npy")

test_image_normalized_f32 = np.load(DATA_FOLDER + "TEST_LW4_IMAGE_NORMALIZED_F32.npy")
test_image_normalized_f16 = np.load(DATA_FOLDER + "TEST_LW4_IMAGE_NORMALIZED_F16.npy")
test_label_1 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_1.npy")
test_label_2 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_2.npy")
test_label_3 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_3.npy")
test_label_4 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_4.npy")
test_label_5 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_5.npy")
test_label_6 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_6.npy")
test_label_7 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_7.npy")
test_label_8 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_8.npy")
test_label_9 = np.load(DATA_FOLDER + "TEST_LW4_LABEL_9.npy")
test_label_1_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_1_INDEXES.npy")
test_label_2_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_2_INDEXES.npy")
test_label_3_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_3_INDEXES.npy")
test_label_4_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_4_INDEXES.npy")
test_label_5_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_5_INDEXES.npy")
test_label_6_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_6_INDEXES.npy")
test_label_7_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_7_INDEXES.npy")
test_label_8_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_8_INDEXES.npy")
test_label_9_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_9_INDEXES.npy")
