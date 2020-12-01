# ------------------------------------------------------------ #
#
# File : NeNISt.py
# Authors : CM
# Interface module for training, evaluation and prediction
# 
# ------------------------------------------------------------ #
import sys
import configparser

# Check if an argument is given, user is trusted to give
# a valid configuration filename.
if not (__name__ == "__main__" and len(sys.argv) == 2):
    print("usage : python NeNISt.py config.cfg")
    exit(0)

config = configparser.ConfigParser()
config.read(sys.argv[1])

print(config['NeNISt'])
