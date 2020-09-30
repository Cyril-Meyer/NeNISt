# ------------------------------------------------------------ #
#
# File : data/datasets/I3.py
# Authors : CM
# Easy I3 dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np

if(os.uname()[1] == 'lythandas'):
    DATA_FOLDER = "/home/cyril/Documents/Data/MEYER_ISBI_2021/"
else:
    DATA_FOLDER = "/b/home/miv/cmeyer/I_2021/MEYER_ISBI_2021/"

train_i3_image_normalized_f32 = np.load(DATA_FOLDER + "TRAIN_I3_IMAGE_NORMALIZED_F32.npy")
train_i3_image_normalized_f16 = np.load(DATA_FOLDER + "TRAIN_I3_IMAGE_NORMALIZED_F16.npy")
train_i3_label_1 = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_1.npy")
train_i3_label_2 = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_2.npy")
train_i3_label_3 = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_3.npy")
train_i3_label_4 = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_4.npy")
train_i3_label_1_indexes = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_1_INDEXES.npy")
train_i3_label_2_indexes = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_2_INDEXES.npy")
train_i3_label_3_indexes = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_3_INDEXES.npy")
train_i3_label_4_indexes = np.load(DATA_FOLDER + "TRAIN_I3_LABEL_4_INDEXES.npy")

valid_i3_image_normalized_f32 = np.load(DATA_FOLDER + "VALID_I3_IMAGE_NORMALIZED_F32.npy")
valid_i3_image_normalized_f16 = np.load(DATA_FOLDER + "VALID_I3_IMAGE_NORMALIZED_F16.npy")
valid_i3_label_1 = np.load(DATA_FOLDER + "VALID_I3_LABEL_1.npy")
valid_i3_label_2 = np.load(DATA_FOLDER + "VALID_I3_LABEL_2.npy")
valid_i3_label_3 = np.load(DATA_FOLDER + "VALID_I3_LABEL_3.npy")
valid_i3_label_4 = np.load(DATA_FOLDER + "VALID_I3_LABEL_4.npy")
valid_i3_label_1_indexes = np.load(DATA_FOLDER + "VALID_I3_LABEL_1_INDEXES.npy")
valid_i3_label_2_indexes = np.load(DATA_FOLDER + "VALID_I3_LABEL_2_INDEXES.npy")
valid_i3_label_3_indexes = np.load(DATA_FOLDER + "VALID_I3_LABEL_3_INDEXES.npy")
valid_i3_label_4_indexes = np.load(DATA_FOLDER + "VALID_I3_LABEL_4_INDEXES.npy")

test_i3_image_normalized_f32 = np.load(DATA_FOLDER + "TEST_I3_IMAGE_NORMALIZED_F32.npy")
test_i3_image_normalized_f16 = np.load(DATA_FOLDER + "TEST_I3_IMAGE_NORMALIZED_F16.npy")
test_i3_label_1 = np.load(DATA_FOLDER + "TEST_I3_LABEL_1.npy")
test_i3_label_2 = np.load(DATA_FOLDER + "TEST_I3_LABEL_2.npy")
test_i3_label_3 = np.load(DATA_FOLDER + "TEST_I3_LABEL_3.npy")
test_i3_label_4 = np.load(DATA_FOLDER + "TEST_I3_LABEL_4.npy")
test_i3_label_1_indexes = np.load(DATA_FOLDER + "TEST_I3_LABEL_1_INDEXES.npy")
test_i3_label_2_indexes = np.load(DATA_FOLDER + "TEST_I3_LABEL_2_INDEXES.npy")
test_i3_label_3_indexes = np.load(DATA_FOLDER + "TEST_I3_LABEL_3_INDEXES.npy")
test_i3_label_4_indexes = np.load(DATA_FOLDER + "TEST_I3_LABEL_4_INDEXES.npy")
