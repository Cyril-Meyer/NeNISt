# ------------------------------------------------------------ #
#
# File : data/datasets/MitoEM_dt.py
# Authors : CM
# Easy MitoEM dataset access
#
# ------------------------------------------------------------ #

import os
import numpy as np
import data.datasets.MitoEMH_dt as D1
import data.datasets.MitoEMR_dt as D2

train_image_normalized_f16 = np.stack([D1.train_image_normalized_f16, D2.train_image_normalized_f16])
train_label_dt = np.stack([D1.train_label_dt, D2.train_label_dt])
valid_image_normalized_f16 = np.stack([D1.valid_image_normalized_f16, D2.valid_image_normalized_f16])
valid_label_dt = np.stack([D1.valid_label_dt, D2.valid_label_dt])
