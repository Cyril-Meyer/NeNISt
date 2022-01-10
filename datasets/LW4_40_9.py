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
train_image_standardized_f32 = np.load(DATA_FOLDER + "TRAIN_LW4_IMAGE_STANDARDIZED_F32.npy")
train_image_standardized_f16 = np.load(DATA_FOLDER + "TRAIN_LW4_IMAGE_STANDARDIZED_F16.npy")
train_labels_dt = np.load(DATA_FOLDER + "TRAIN_LW4_LABELS_DT.npy")
train_label_1_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_1_INDEXES.npy")
train_label_2_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_2_INDEXES.npy")
train_label_3_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_3_INDEXES.npy")
train_label_4_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_4_INDEXES.npy")
train_label_5_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_5_INDEXES.npy")
train_label_6_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_6_INDEXES.npy")
train_label_7_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_7_INDEXES.npy")
train_label_8_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_8_INDEXES.npy")
train_label_9_indexes = np.load(DATA_FOLDER + "TRAIN_LW4_LABEL_9_INDEXES.npy")


test_image_normalized_f32 = np.load(DATA_FOLDER + "TEST_LW4_IMAGE_NORMALIZED_F32.npy")
test_image_normalized_f16 = np.load(DATA_FOLDER + "TEST_LW4_IMAGE_NORMALIZED_F16.npy")
test_image_standardized_f32 = np.load(DATA_FOLDER + "TEST_LW4_IMAGE_STANDARDIZED_F32.npy")
test_image_standardized_f16 = np.load(DATA_FOLDER + "TEST_LW4_IMAGE_STANDARDIZED_F16.npy")
test_labels_dt = np.load(DATA_FOLDER + "TEST_LW4_LABELS_DT.npy")
test_label_1_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_1_INDEXES.npy")
test_label_2_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_2_INDEXES.npy")
test_label_3_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_3_INDEXES.npy")
test_label_4_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_4_INDEXES.npy")
test_label_5_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_5_INDEXES.npy")
test_label_6_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_6_INDEXES.npy")
test_label_7_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_7_INDEXES.npy")
test_label_8_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_8_INDEXES.npy")
test_label_9_indexes = np.load(DATA_FOLDER + "TEST_LW4_LABEL_9_INDEXES.npy")
