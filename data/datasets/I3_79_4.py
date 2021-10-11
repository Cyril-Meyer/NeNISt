# ------------------------------------------------------------ #
#
# File : data/datasets/I3_79_4.py
# Authors : CM
# Easy LW4 40 slice 9 class dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np

if(os.uname()[1] == 'lythandas'):
    DATA_FOLDER = "/home/cyril/Documents/Data/I3_79_4/"
else:
    DATA_FOLDER = "/b/home/miv/cmeyer/Data/I3_79_4/"

train_image_normalized_f32 = np.load(DATA_FOLDER + "TRAIN_IMAGE_NORMALIZED_F32.npy")
train_image_normalized_f16 = np.load(DATA_FOLDER + "TRAIN_IMAGE_NORMALIZED_F16.npy")
train_labels_dt = np.load(DATA_FOLDER + "TRAIN_LABELS_DT.npy")


test_image_normalized_f32 = np.load(DATA_FOLDER + "TEST_IMAGE_NORMALIZED_F32.npy")
test_image_normalized_f16 = np.load(DATA_FOLDER + "TEST_IMAGE_NORMALIZED_F16.npy")
test_labels_dt = np.load(DATA_FOLDER + "TEST_LABELS_DT.npy")
