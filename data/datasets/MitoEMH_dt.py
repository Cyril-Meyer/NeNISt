# ------------------------------------------------------------ #
#
# File : data/datasets/MitoEMH_dt.py
# Authors : CM
# Easy MitoEM dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np

if(os.uname()[1] == 'lythandas'):
    DATA_FOLDER = "/home/cyril/Documents/Data/MitoEM/MitoEM-H/"
else:
    DATA_FOLDER = "/b/home/miv/cmeyer/Data/MitoEM/MitoEM-H/"

'''
train_label = np.load(DATA_FOLDER + "TRAIN_LABEL_DT.npy")
valid_label = np.load(DATA_FOLDER + "VALID_LABEL_DT.npy")
'''
train_image_normalized_f16 = np.load(DATA_FOLDER + "TRAIN_IMAGE_NORMALIZED_F16.npy")
train_label = None
valid_image_normalized_f16 = np.load(DATA_FOLDER + "VALID_IMAGE_NORMALIZED_F16.npy")
valid_label = None

def load_dt(dt_neg, dt_pos):
    global train_label
    global valid_label
    train_label = np.load(DATA_FOLDER + "TRAIN_LABEL_DT_" + str(int(dt_neg)) + "_" + str(int(dt_pos)) + ".npy")
    valid_label = np.load(DATA_FOLDER + "VALID_LABEL_DT_" + str(int(dt_neg)) + "_" + str(int(dt_pos)) + ".npy")
