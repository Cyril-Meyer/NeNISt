# ------------------------------------------------------------ #
#
# File : data/datasets/MitoEMR.py
# Authors : CM
# Easy MitoEM dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np

if(os.uname()[1] == 'lythandas'):
    DATA_FOLDER = "/home/cyril/Documents/Data/MitoEM/MitoEM-R/"
else:
    DATA_FOLDER = "/b/home/miv/cmeyer/Data/MitoEM/MitoEM-R/"

train_image_normalized_f16 = np.load(DATA_FOLDER + "TRAIN_IMAGE_NORMALIZED_F16.npy")
train_label = np.load(DATA_FOLDER + "TRAIN_LABEL.npy")
train_label_dt = np.load(DATA_FOLDER + "TRAIN_LABEL_DT.npy")

valid_image_normalized_f16 = np.load(DATA_FOLDER + "VALID_IMAGE_NORMALIZED_F16.npy")
valid_label = np.load(DATA_FOLDER + "VALID_LABEL.npy")
valid_label_dt = np.load(DATA_FOLDER + "VALID_LABEL_DT.npy")
