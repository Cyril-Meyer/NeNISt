# ------------------------------------------------------------ #
#
# File : data/datasets/IGBMC_small_multi.py
# Authors : CM
# I3 + LW4, 80 slices each
#
# ------------------------------------------------------------ #

import os
import numpy as np

if(os.uname()[1] == 'lythandas'):
    DATA_FOLDER = "/HDD1/data/IGBMC_Small/"
else:
    DATA_FOLDER = "/b/home/miv/cmeyer/Data/IGBMC_Small/"

i3_image = np.load(DATA_FOLDER + "I3_MULTI_IMAGE.npy")
i3_label_1 = np.load(DATA_FOLDER + "I3_MULTI_LABEL_MITO.npy")
i3_label_2 = np.load(DATA_FOLDER + "I3_MULTI_LABEL_RETI.npy")
i3_label_1_dt = np.load(DATA_FOLDER + "I3_MULTI_LABEL_MITO_DT.npy")
i3_label_2_dt = np.load(DATA_FOLDER + "I3_MULTI_LABEL_RETI_DT.npy")

i3_labels = np.stack([i3_label_1, i3_label_2], axis=-1)
i3_labels_dt = np.stack([i3_label_1_dt, i3_label_2_dt], axis=-1)


lw4_image = np.load(DATA_FOLDER + "LW4_MULTI_IMAGE.npy")
lw4_label_1 = np.load(DATA_FOLDER + "LW4_MULTI_LABEL_MITO.npy")
lw4_label_2 = np.load(DATA_FOLDER + "LW4_MULTI_LABEL_RETI.npy")
lw4_label_1_dt = np.load(DATA_FOLDER + "LW4_MULTI_LABEL_MITO_DT.npy")
lw4_label_2_dt = np.load(DATA_FOLDER + "LW4_MULTI_LABEL_RETI_DT.npy")

lw4_labels = np.stack([lw4_label_1, lw4_label_2], axis=-1)
lw4_labels_dt = np.stack([lw4_label_1_dt, lw4_label_2_dt], axis=-1)

train_image = [i3_image[0:40], lw4_image[0:40]]
train_labels = [i3_labels[0:40], lw4_labels[0:40]]
train_labels_dt = [i3_labels_dt[0:40], lw4_labels_dt[0:40]]

valid_image = [i3_image[40:60], lw4_image[40:60]]
valid_labels = [i3_labels[40:60], lw4_labels[40:60]]
valid_labels_dt = [i3_labels_dt[40:60], lw4_labels_dt[40:60]]

test_image = [i3_image[60:80], lw4_image[60:80]]
test_labels = [i3_labels[60:80], lw4_labels[60:80]]
test_labels_dt = [i3_labels_dt[60:80], lw4_labels_dt[60:80]]

