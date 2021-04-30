import sys
import os

import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *


def get_filename_extension(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower()


def normalize(image, dtype=np.float32):
    image_min = image.min()
    image_max = image.max()
    return np.array((image - image_min) / (image_max - image_min)).astype(dtype)


def ok_msg(msg):
    err = QtWidgets.QMessageBox(QMessageBox.Information, "", msg)
    return err.exec_()


def err_msg(msg):
    err = QtWidgets.QMessageBox(QMessageBox.Critical, "Erreur", msg)
    return err.exec_()
