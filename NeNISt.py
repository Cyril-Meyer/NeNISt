# ------------------------------------------------------------ #
#
# File : NeNISt.py
# Authors : CM
# Interface module for training, evaluation and prediction
# 
# ------------------------------------------------------------ #
import sys
import datetime
import time
import configparser

# Check if an argument is given, user is trusted to give
# a valid configuration filename.
if not (__name__ == "__main__" and len(sys.argv) == 2):
    print("usage : python NeNISt.py config.cfg")
    exit(0)


dt = datetime.datetime.today().strftime("%H%M%f")

config = configparser.ConfigParser()
config.read(sys.argv[1])


cfg_common = dict(config['NeNISt'])

name = cfg_common['name'] + "_" + dt
print(name)

PATCH_SIZE_X = cfg_common['patch_size_x']
PATCH_SIZE_Y = cfg_common['patch_size_y']
PATCH_SIZE_Z = cfg_common['patch_size_z']



'''
DATASET = LW4_40_9

MODEL = 2DUNET
MODEL_OUTPUT_CLASSES = 10
MODEL_OUTPUT_ACTIVATION = sigmoid
MODEL_FILTERS = 64
MODEL_DEPTH = 4
MODEL_CONV_PER_BLOCK = 2
MODEL_DROPOUTS = 0.50
MODEL_BATCH_NORM = True
GROUPS = 1
'''

if "train" in cfg_common['do'].lower():
    
    print(cfg_common['do'])
