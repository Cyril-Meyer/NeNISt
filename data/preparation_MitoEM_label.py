# ------------------------------------------------------------ #
#
# File : data/preparation_MitoEM_label.py
# Authors : CM
#
# ------------------------------------------------------------ #

import os
import sys

import numpy as np
import skimage
from skimage import io

from distance_transform import *

anisotropy=(30, 8, 8)
normalize = normalize_tanh
normalize_scale_neg = int(sys.argv[1])
normalize_scale_pos = int(sys.argv[2])
print(normalize_scale_neg, normalize_scale_pos)

PATH_IN = "/HDD1/data/MitoEM/MitoEM-H/"
PATH_OUT = "/HDD1/data/MitoEM/MitoEM-H/"
'''
PATH_IN = "/HDD1/data/MitoEM/MitoEM-R/"
PATH_OUT = "/HDD1/data/MitoEM/MitoEM-R/"
'''

print("READ TRAIN")
label_train = ((np.array(skimage.io.imread(PATH_IN + "mito_train.tif")) > 0)*1.0).astype(np.uint8)

print("PROCESSING TRAIN")
label_train_dt = label_dt_f16(label_train, anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)

print("SAVE TRAIN")
# only needed once
#np.save(PATH_OUT + "TRAIN_LABEL.npy", label_train)
np.save(PATH_OUT + "TRAIN_LABEL_DT_" + str(int(normalize_scale_neg)) + "_" + str(int(normalize_scale_pos)) + ".npy", label_train_dt)
skimage.io.imsave(PATH_OUT + "mito_train_bin.tif", label_train*255)
skimage.io.imsave(PATH_OUT + "mito_train_bin_dt.tif", ((label_train_dt+1)/2*255).astype(np.uint8))

if ((label_train_dt > 0) == (label_train > 0)).all():
    print("OK")
else:
    print("ERROR 01 : CHECK CODE")

# free memory required.
del label_train, label_train_dt


print("READ VALID")
label_valid = ((np.array(skimage.io.imread(PATH_IN + "mito_val.tif")) > 0)*1.0).astype(np.uint8)

print("PROCESSING VALID")
label_valid_dt = label_dt_f16(label_valid, anisotropy, normalize, normalize_scale_pos, normalize_scale_neg)

print("SAVE VALID")
# only needed once
#np.save(PATH_OUT + "VALID_LABEL.npy", label_valid)
np.save(PATH_OUT + "VALID_LABEL_DT_" + str(int(normalize_scale_neg)) + "_" + str(int(normalize_scale_pos)) + ".npy", label_valid_dt)
skimage.io.imsave(PATH_OUT + "mito_valid_bin.tif", label_valid*255)
skimage.io.imsave(PATH_OUT + "mito_valid_bin_dt.tif", ((label_valid_dt+1)/2*255).astype(np.uint8))

if ((label_valid_dt > 0) == (label_valid > 0)).all():
    print("OK")
else:
    print("ERROR 01 : CHECK CODE")
