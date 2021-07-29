import sys
import os

import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

import edt


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


def scribble_points_to_distance_map(map_shape, scribble_points, clip_dist=20):
    distance_map = np.zeros(map_shape)
    for a in scribble_points:
        distance_map[a] = 1
    distance_map = -edt.edt(1-distance_map)
    distance_map = (np.clip(distance_map, -clip_dist, clip_dist)/clip_dist)+1
    return distance_map
