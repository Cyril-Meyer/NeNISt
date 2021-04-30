import sys
import os
import time

import numpy as np
import tensorflow as tf
from skimage import io
import qimage2ndarray

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from cutterui import Ui_MainWindow
from utils import *


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.MIN_VIEW_SIZE = 8

        self.current_slice = 0
        self.current_zoom = 50
        self.current_pos_x = 0
        self.current_pos_y = 0

        self.current_selection_min_x = 0
        self.current_selection_min_y = 0
        self.current_selection_min_z = 0
        self.current_selection_size_x = 32
        self.current_selection_size_y = 32
        self.current_selection_size_z = 1

        self.selected_image_row = 0

        self.selected_image = None
        self.selected_model = None
        self.selected_label = None
        self.images = []
        self.models = []
        self.labels = []

    def resizeEvent(self, event):
        self.xsize = event.size().width()
        self.ysize = event.size().height()
        QMainWindow.resizeEvent(self, event)
        self.update_view()

    # ---------------------------------------------------------------------------------------------------------------- #
    # inputs
    # ---------------------------------------------------------------------------------------------------------------- #
    def update_lists(self):
        self.listWidget_images.clear()
        self.listWidget_models.clear()
        self.listWidget_labels.clear()

        for i in self.images:
            self.listWidget_images.addItem(str(i['shape']) + " " + i['name'])

        self.listWidget_images.setCurrentRow(self.selected_image_row)
        self.change_selected_image()

    def dialog_add_image(self):
        images_filenames = QFileDialog.getOpenFileNames(self, "Sélectionner des images", "/home/cyril/Documents/Data/")
        for filename in images_filenames[0]:
            print(filename)

            ext = get_filename_extension(filename)
            name = os.path.basename(filename)

            if ext == ".npy":
                data = np.load(filename)
            elif ext in [".tiff", ".tif"]:
                # tiff are normalized by default
                data = normalize(np.array(io.imread(filename)))
            else:
                err_msg(filename + " extension de fichier invalide")
                continue

            image = {'name':name,
                     'shape':data.shape,
                     'data':data}
            self.images.append(image)

        self.update_lists()

    def dialog_add_model(self):
        raise NotImplementedError

    # ---------------------------------------------------------------------------------------------------------------- #
    # change in selection or view
    # ---------------------------------------------------------------------------------------------------------------- #
    def change_selected_image(self):
        self.selected_image_row = self.listWidget_images.currentRow()
        self.selected_image = self.images[self.listWidget_images.currentRow()]

        self.horizontalSlider_slice.setRange(0, self.selected_image['shape'][0]-1)
        self.horizontalSlider_pos_x.setRange(0, self.selected_image['shape'][2]-self.MIN_VIEW_SIZE)
        self.verticalSlider_pos_y.setRange(0, self.selected_image['shape'][1]-self.MIN_VIEW_SIZE)
        self.spinBox_selection_min_x.setRange(0, self.selected_image['shape'][2])
        self.spinBox_selection_min_y.setRange(0, self.selected_image['shape'][1])
        self.spinBox_selection_min_z.setRange(0, self.selected_image['shape'][0])
        self.update_view()

    def change_slice(self):
        if self.selected_image is not None:
            self.current_slice = self.horizontalSlider_slice.value()
            self.label_slice.setText(str(self.current_slice+1) + "/" + str(self.selected_image['shape'][0]))
        else:
            self.label_slice.setText("0/0")
        self.update_view()

    def change_zoom(self):
        # value between 0 and 100
        self.current_zoom = self.verticalSlider_zoom.value()
        # todo fill with space current_zoom string
        self.label_zoom.setText(str(self.current_zoom)+"%")
        self.update_view()

    def change_pos(self):
        self.current_pos_x = self.horizontalSlider_pos_x.value()
        self.current_pos_y = self.verticalSlider_pos_y.value()
        self.update_view()

    def center_selection(self):
        self.spinBox_selection_min_x.setValue(self.current_pos_x)
        self.spinBox_selection_min_y.setValue(self.current_pos_y)
        self.spinBox_selection_min_z.setValue(self.current_slice)
        return

    def update_selection(self):
        self.current_selection_min_x = self.spinBox_selection_min_x.value()
        self.current_selection_min_y = self.spinBox_selection_min_y.value()
        self.current_selection_min_z = self.spinBox_selection_min_z.value()
        self.current_selection_size_x = self.spinBox_selection_size_x.value()
        self.current_selection_size_y = self.spinBox_selection_size_y.value()
        self.current_selection_size_z = self.spinBox_selection_size_z.value()
        self.update_view()

    # ---------------------------------------------------------------------------------------------------------------- #
    # update view
    # ---------------------------------------------------------------------------------------------------------------- #
    def update_view(self):
        if self.selected_image is not None:
            data = self.selected_image['data']

            view = data[self.current_slice]*255

            # todo : remove the and True with an RGB mode selection
            if len(view.shape) == 2 and True:
                view = np.stack((view,) * 3, axis=-1)

            min_x = self.current_selection_min_x
            min_y = self.current_selection_min_y
            min_z = self.current_selection_min_z
            size_x = self.current_selection_size_x
            size_y = self.current_selection_size_y
            size_z = self.current_selection_size_z

            w = 4
            if self.current_slice in range(min_z, min_z+size_z):
                view[min_y:min_y+w, min_x:min_x+size_x] = (255, 0, 0)
                view[min_y-w+size_y:min_y+size_y, min_x:min_x+size_x] = (255, 0, 0)
                view[min_y:min_y+size_y, min_x:min_x+w] = (255, 0, 0)
                view[min_y:min_y+size_y, min_x-w+size_x:min_x+size_x] = (255, 0, 0)


            z, y, x = data.shape
            zoom = 100-self.current_zoom
            zoom_x = int(max(self.MIN_VIEW_SIZE, int((x/100)*zoom)))
            zoom_y = int(max(self.MIN_VIEW_SIZE, int((y/100)*zoom)))
            view = view[self.current_pos_y:zoom_y+self.current_pos_y, self.current_pos_x:self.current_pos_x+zoom_x]

            if len(view.shape) in [2, 3]:
                pixmap = QPixmap(qimage2ndarray.array2qimage(view))
                pixmap = pixmap.scaled(self.label_main_view.geometry().width()-10-int(self.xsize*0.05),
                                       self.label_main_view.geometry().height()-10-int(self.ysize*0.05),
                                       QtCore.Qt.KeepAspectRatio)
                self.label_main_view.setPixmap(pixmap)
            else:
                raise NotImplementedError


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
