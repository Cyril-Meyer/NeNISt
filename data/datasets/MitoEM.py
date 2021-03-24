# ------------------------------------------------------------ #
#
# File : data/datasets/MitoEM.py
# Authors : CM
# Easy MitoEM dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np
import data.datasets.MitoEMH as D1
import data.datasets.MitoEMR as D2

train_image_normalized_f16 = np.stack([D1.train_image_normalized_f16, D2.train_image_normalized_f16])
train_label = np.stack([D1.train_label, D2.train_label])
# train_label_dt = np.stack([D1.train_label_dt, D2.train_label_dt])
valid_image_normalized_f16 = np.stack([D1.valid_image_normalized_f16, D2.valid_image_normalized_f16])
valid_label = np.stack([D1.valid_label, D2.valid_label])
# valid_label_dt = np.stack([D1.valid_label_dt, D2.valid_label_dt])
