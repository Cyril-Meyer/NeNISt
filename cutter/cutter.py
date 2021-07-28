import sys
import os
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np
import tensorflow as tf
from skimage import io
import cv2
import h5py
from tqdm import tqdm
from lii import LargeImageInference as lii

from cutterui import Ui_MainWindow
from utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
VERBOSE = 1


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.MIN_VIEW_SIZE = 8

        self.current_slice = 0
        self.current_zoom = 0
        self.current_pos_x = 0
        self.current_pos_y = 0
        self.current_class = 0
        self.label_opacity = 200

        self.current_selection_min_x = 0
        self.current_selection_min_y = 0
        self.current_selection_min_z = 0
        self.current_selection_size_x = 256
        self.current_selection_size_y = 256
        self.current_selection_size_z = 16

        self.selected_image_row = 0
        self.selected_model_row = 0
        self.selected_label_row = 0

        self.selected_image = None
        self.selected_model = None
        self.selected_label = None
        self.images = []
        self.models = []
        self.labels = []

        self.mouse_left_pressed = False
        self.mouse_right_pressed = False

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

        for m in self.models:
            self.listWidget_models.addItem(str(m['dim']) + "D " + m['name'])

        for l in self.labels:
            self.listWidget_labels.addItem(str(l['offset']) + " " + str(l['shape']) + " " + str(l['name']))

        self.listWidget_images.setCurrentRow(self.selected_image_row)
        self.listWidget_models.setCurrentRow(0)
        self.listWidget_labels.setCurrentRow(0)

        self.change_selected_image()

    def dialog_add_image(self):
        images_filenames = QFileDialog.getOpenFileNames(self, "Sélectionner des images",
                                                        "/home/cyril/Development/NeNISt/cutter_example/")
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

            image = {'name': name,
                     'shape': data.shape,
                     'data': data}
            self.images.append(image)

        self.update_lists()

    def dialog_add_model(self):
        models_filenames = QFileDialog.getOpenFileNames(self, "Sélectionner des modèles",
                                                        "/home/cyril/Development/NeNISt/cutter_example/")
        for filename in models_filenames[0]:
            print(filename)

            ext = get_filename_extension(filename)
            name = os.path.basename(filename)

            if ext == ".h5":
                model = tf.keras.models.load_model(filename)

            else:
                err_msg(filename + " extension de fichier invalide")
                continue

            # print(model.count_params())

            model = {'name': name,
                     'dim': len(model.input_shape)-2,
                     'model': model}
            self.models.append(model)

        self.update_lists()

    def dialog_add_label(self):
        labels_filenames = QFileDialog.getOpenFileNames(self, "Sélectionner des étiquettes",
                                                        "/home/cyril/Development/NeNISt/cutter_example/")
        for filename in labels_filenames[0]:
            print(filename)

            ext = get_filename_extension(filename)
            name = os.path.basename(filename)

            if ext == ".h5":
                h5f = h5py.File(filename, 'r')
            else:
                err_msg(filename + " extension de fichier invalide")
                continue
            shape = tuple(h5f['shape'])

            data = np.zeros(shape)
            for c in range(shape[-1]):
                data[:, :, :, c] = np.array(h5f[f"data_{c}"])

            label = {'name': str(np.array(h5f['name'])),
                     'shape': shape,
                     'offset': tuple(h5f['offset']),
                     'data': data}

            self.labels.append(label)

        self.update_lists()

    # ---------------------------------------------------------------------------------------------------------------- #
    # outputs
    # ---------------------------------------------------------------------------------------------------------------- #
    def export_selected_label(self):
        if self.selected_label is not None:
            label_filename = QFileDialog.getSaveFileName(self, "Sélectionner le fichier",
                                                         "/home/cyril/Development/NeNISt/cutter_example/",
                                                         "HDF5 files (*.h5)")[0]
            if label_filename == '':
                ok_msg("Aucune fichier sélectionnée")
                return

            if not get_filename_extension(label_filename) == ".h5":
                label_filename = label_filename + ".h5"

            h5f = h5py.File(label_filename, "w")
            h5f.create_dataset("name", data=self.selected_label['name'])
            for c in range(self.selected_label['data'].shape[-1]):
                h5f.create_dataset(f"data_{c}", data=self.selected_label['data'][:, :, :, c])
                # h5f.create_dataset(f"data_{c}", data=self.selected_label['data'][::2, ::2, ::2, c])
            h5f.create_dataset("shape", data=self.selected_label['shape'])
            h5f.create_dataset("offset", data=self.selected_label['offset'])
        else:
            err_msg("Aucune étiquette sélectionnée")

    # ---------------------------------------------------------------------------------------------------------------- #
    # predict
    # ---------------------------------------------------------------------------------------------------------------- #
    def predict_selection(self):
        tf.keras.backend.clear_session()
        if self.selected_image is not None and self.selected_model is not None:
            selection = self.selected_image['data']

            min_x = self.current_selection_min_x
            min_y = self.current_selection_min_y
            min_z = self.current_selection_min_z
            size_x = self.current_selection_size_x
            size_y = self.current_selection_size_y
            size_z = self.current_selection_size_z

            selection = selection[min_z:min_z+size_z, min_y:min_y+size_y, min_x:min_x+size_x]
            if not selection.shape == (size_z, size_y, size_x):
                err_msg(f"La selection déborde : {selection.shape} est différent de {(size_z, size_y, size_x)}")
                return

            selection = np.expand_dims(np.expand_dims(selection, -1), 0)

            if self.selected_model['dim'] == 3:
                # prediction = self.selected_model['model'](selection, training=False)[0]
                prediction = self.selected_model['model'].predict(selection)[0]
            elif self.selected_model['dim'] == 2:
                prediction = []
                for z in tqdm(range(size_z), disable=(not VERBOSE >= 2)):
                    # pred = self.selected_model['model'](selection[:, z, :, :, :], training=False)[0]
                    pred = self.selected_model['model'].predict(selection[:, z, :, :, :])[0]
                    prediction.append(pred)
                prediction = np.array(prediction)
            else:
                raise NotImplementedError

            label = {'name':self.selected_image['name'] + self.selected_model['name'],
                     'shape':prediction.shape,
                     'offset':(min_z, min_y, min_x),
                     'data':prediction}
            self.labels.append(label)
            self.update_lists()
        else:
            err_msg("Pas d'image ou pas de modèle selectionné")

    def predict_full_image(self):
        # raise NotImplementedError
        if self.selected_image is not None and self.selected_model is not None:
            # image = np.expand_dims(np.expand_dims(self.selected_image['data'], -1), 0)
            image = np.expand_dims(self.selected_image['data'], -1)
            size_x = self.current_selection_size_x
            size_y = self.current_selection_size_y
            size_z = self.current_selection_size_z

            overlap = 1
            if self.checkBox_overlap.isChecked():
                overlap = 2

            if self.selected_model['dim'] == 3:
                def predict(x):
                    tf.keras.backend.clear_session()
                    return self.selected_model['model'].predict(x)

            elif self.selected_model['dim'] == 2:
                def predict(x):
                    tf.keras.backend.clear_session()
                    x = x[0]
                    return np.expand_dims(self.selected_model['model'].predict(x), 0)

                if overlap == 2:
                    overlap = (1, 2, 2)
                size_z = 1
            else:
                raise NotImplementedError

            prediction = lii.infer(image,
                                   (size_z,
                                    size_y,
                                    size_x),
                                   predict,
                                   overlap,
                                   verbose=VERBOSE)

            label = {'name': self.selected_image['name'] + self.selected_model['name'],
                     'shape': prediction.shape,
                     'offset': (0, 0, 0),
                     'data': prediction}
            self.labels.append(label)
            self.update_lists()
        else:
            err_msg("Pas d'image ou pas de modèle selectionné")

    # ---------------------------------------------------------------------------------------------------------------- #
    # change in selection or view
    # ---------------------------------------------------------------------------------------------------------------- #
    def change_selected_image(self):
        self.selected_image_row = self.listWidget_images.currentRow()
        if self.selected_image_row >= 0:
            self.selected_image = self.images[self.selected_image_row]
            self.horizontalSlider_slice.setRange(0, self.selected_image['shape'][0]-1)
            self.horizontalSlider_pos_x.setRange(0, self.selected_image['shape'][2]-self.MIN_VIEW_SIZE)
            self.verticalSlider_pos_y.setRange(0, self.selected_image['shape'][1]-self.MIN_VIEW_SIZE)
            self.spinBox_selection_min_x.setRange(0, self.selected_image['shape'][2])
            self.spinBox_selection_min_y.setRange(0, self.selected_image['shape'][1])
            self.spinBox_selection_min_z.setRange(0, self.selected_image['shape'][0])
            self.update_view()

    def change_selected_model(self):
        self.selected_model_row = self.listWidget_models.currentRow()
        if self.selected_model_row >= 0:
            self.selected_model = self.models[self.selected_model_row]

    def change_selected_label(self):
        self.selected_label_row = self.listWidget_labels.currentRow()
        if self.selected_label_row >= 0:
            self.selected_label = self.labels[self.selected_label_row]
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

    def change_label_opacity(self):
        self.label_opacity = self.spinBox_label_opacity.value()
        self.update_view()

    def change_class(self):
        self.current_class = self.spinBox_class.value()
        self.update_view()

    # ---------------------------------------------------------------------------------------------------------------- #
    # update view
    # ---------------------------------------------------------------------------------------------------------------- #
    def update_view(self):
        color_selection = (0, 204, 0)
        color_label = (255, 0, 0)
        color_label_alpha = self.label_opacity / 255.0
        if self.selected_image is not None:
            data = self.selected_image['data']
            # select slice
            view = data[self.current_slice]

            # Grayscale to RGB
            # todo : remove the and True with an RGB mode selection
            if len(view.shape) == 2 and True:
                view = np.stack((view,) * 3, axis=-1)

            # label superposition
            if self.selected_label is not None:
                min_z, min_y, min_x = self.selected_label['offset']
                size_z, size_y, size_x, n_class = self.selected_label['shape']
                label = self.selected_label['data']

                if self.current_slice in range(min_z, min_z+size_z) and self.current_class < n_class:
                    label = label[self.current_slice-min_z]
                    view[min_y:min_y + size_y, min_x:min_x + size_x, 0] = \
                        view[min_y:min_y + size_y, min_x:min_x + size_x, 0] + \
                        label[:, :, self.current_class]*1.0 * color_label_alpha

            # [0,1] -> [0,255]
            view = (np.clip(view, 0, 1)*255).astype(np.uint8)

            # full slice to GUI selected area
            min_x = self.current_selection_min_x
            min_y = self.current_selection_min_y
            min_z = self.current_selection_min_z
            size_x = self.current_selection_size_x
            size_y = self.current_selection_size_y
            size_z = self.current_selection_size_z

            # select area
            w = 4
            if self.current_slice in range(min_z, min_z+size_z):
                view[min_y:min_y+w, min_x:min_x+size_x] = color_selection
                view[min_y-w+size_y:min_y+size_y, min_x:min_x+size_x] = color_selection
                view[min_y:min_y+size_y, min_x:min_x+w] = color_selection
                view[min_y:min_y+size_y, min_x-w+size_x:min_x+size_x] = color_selection

            z, y, x = data.shape
            zoom = 100-self.current_zoom
            zoom_x = int(max(self.MIN_VIEW_SIZE, int((x/100)*zoom)))
            zoom_y = int(max(self.MIN_VIEW_SIZE, int((y/100)*zoom)))
            view = view[self.current_pos_y:zoom_y+self.current_pos_y, self.current_pos_x:self.current_pos_x+zoom_x]

            # show
            if len(view.shape) == 3:
                height, width, channel = view.shape
                bytesPerLine = 3 * width
                qImg = QImage(view.tobytes(), width, height, bytesPerLine, QImage.Format_RGB888)# .rgbSwapped()
                pixmap = QPixmap.fromImage(qImg)
                pixmap = pixmap.scaled(self.label_main_view.geometry().width()-25,
                                       self.label_main_view.geometry().height()-25,
                                       QtCore.Qt.KeepAspectRatio)
                self.label_main_view.setPixmap(pixmap)
                # self.label_main_view.setFixedSize(pixmap.size())

                self.current_view_shape = view.shape
                self.main_view_pixmap_size = pixmap.size()

            else:
                raise NotImplementedError

    def main_view_mouse_event(self, event):
        pos = event.localPos()
        # local pos to view position
        if pos.x() > 0 and pos.y() > 0:
            xr = pos.x() / self.main_view_pixmap_size.width()
            yr = pos.y() / self.main_view_pixmap_size.height()
            x = int(xr * self.current_view_shape[1]) + self.current_pos_x
            y = int(yr * self.current_view_shape[0]) + self.current_pos_y
        else:
            x = 0
            y = 0

        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.mouse_left_pressed = True
            elif event.button() == Qt.RightButton:
                self.mouse_right_pressed = True
                self.spinBox_selection_min_x.setValue(x)
                self.spinBox_selection_min_y.setValue(y)
                self.spinBox_selection_min_z.setValue(self.current_slice)
                self.update_selection()
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self.mouse_left_pressed = False
            elif event.button() == Qt.RightButton:
                self.mouse_right_pressed = False
                if x > self.current_selection_min_x and y > self.current_selection_min_y:
                    self.spinBox_selection_size_x.setValue(((x - self.current_selection_min_x) // 32)*32)
                    self.spinBox_selection_size_y.setValue(((y - self.current_selection_min_y) // 32)*32)
                    self.update_selection()
        elif event.type() == QEvent.MouseMove:
            if self.mouse_right_pressed == True:
                if x > self.current_selection_min_x and y > self.current_selection_min_y:
                    self.spinBox_selection_size_x.setValue(((x - self.current_selection_min_x) // 32)*32)
                    self.spinBox_selection_size_y.setValue(((y - self.current_selection_min_y) // 32)*32)
                    self.update_selection()


app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
