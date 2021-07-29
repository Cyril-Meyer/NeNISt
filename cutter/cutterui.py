# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cutter.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_centralwidget = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_centralwidget.setObjectName("gridLayout_centralwidget")
        self.horizontalSlider_pos_x = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_pos_x.setMaximum(1)
        self.horizontalSlider_pos_x.setProperty("value", 0)
        self.horizontalSlider_pos_x.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_pos_x.setObjectName("horizontalSlider_pos_x")
        self.gridLayout_centralwidget.addWidget(self.horizontalSlider_pos_x, 0, 2, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_slice = QtWidgets.QLabel(self.centralwidget)
        self.label_slice.setObjectName("label_slice")
        self.horizontalLayout.addWidget(self.label_slice)
        self.horizontalSlider_slice = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_slice.setMaximum(1)
        self.horizontalSlider_slice.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_slice.setObjectName("horizontalSlider_slice")
        self.horizontalLayout.addWidget(self.horizontalSlider_slice)
        self.label_zoom = QtWidgets.QLabel(self.centralwidget)
        self.label_zoom.setObjectName("label_zoom")
        self.horizontalLayout.addWidget(self.label_zoom)
        self.gridLayout_centralwidget.addLayout(self.horizontalLayout, 3, 2, 1, 3)
        self.verticalSlider_zoom = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_zoom.setMinimum(0)
        self.verticalSlider_zoom.setMaximum(100)
        self.verticalSlider_zoom.setPageStep(10)
        self.verticalSlider_zoom.setProperty("value", 0)
        self.verticalSlider_zoom.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_zoom.setObjectName("verticalSlider_zoom")
        self.gridLayout_centralwidget.addWidget(self.verticalSlider_zoom, 1, 4, 2, 1)
        self.verticalSlider_pos_y = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_pos_y.setMaximum(1)
        self.verticalSlider_pos_y.setProperty("value", 0)
        self.verticalSlider_pos_y.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_pos_y.setInvertedAppearance(True)
        self.verticalSlider_pos_y.setObjectName("verticalSlider_pos_y")
        self.gridLayout_centralwidget.addWidget(self.verticalSlider_pos_y, 1, 1, 1, 1)
        self.label_main_view = InteractiveQLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_main_view.sizePolicy().hasHeightForWidth())
        self.label_main_view.setSizePolicy(sizePolicy)
        self.label_main_view.setText("")
        self.label_main_view.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_main_view.setObjectName("label_main_view")
        self.gridLayout_centralwidget.addWidget(self.label_main_view, 1, 2, 2, 2)
        self.gridLayout_main_left = QtWidgets.QGridLayout()
        self.gridLayout_main_left.setObjectName("gridLayout_main_left")
        self.pushButton_add_image = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_image.setMaximumSize(QtCore.QSize(25, 16777215))
        self.pushButton_add_image.setObjectName("pushButton_add_image")
        self.gridLayout_main_left.addWidget(self.pushButton_add_image, 1, 1, 1, 1)
        self.listWidget_models = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_models.setObjectName("listWidget_models")
        self.gridLayout_main_left.addWidget(self.listWidget_models, 4, 0, 1, 3)
        self.gridLayout_selection = QtWidgets.QGridLayout()
        self.gridLayout_selection.setObjectName("gridLayout_selection")
        self.label_class = QtWidgets.QLabel(self.centralwidget)
        self.label_class.setObjectName("label_class")
        self.gridLayout_selection.addWidget(self.label_class, 0, 0, 1, 2)
        self.label_selection_size = QtWidgets.QLabel(self.centralwidget)
        self.label_selection_size.setAlignment(QtCore.Qt.AlignCenter)
        self.label_selection_size.setObjectName("label_selection_size")
        self.gridLayout_selection.addWidget(self.label_selection_size, 3, 2, 1, 1)
        self.spinBox_selection_size_y = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_selection_size_y.sizePolicy().hasHeightForWidth())
        self.spinBox_selection_size_y.setSizePolicy(sizePolicy)
        self.spinBox_selection_size_y.setMinimum(32)
        self.spinBox_selection_size_y.setMaximum(4096)
        self.spinBox_selection_size_y.setSingleStep(32)
        self.spinBox_selection_size_y.setProperty("value", 256)
        self.spinBox_selection_size_y.setObjectName("spinBox_selection_size_y")
        self.gridLayout_selection.addWidget(self.spinBox_selection_size_y, 5, 2, 1, 1)
        self.spinBox_class = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_class.setMaximum(4096)
        self.spinBox_class.setObjectName("spinBox_class")
        self.gridLayout_selection.addWidget(self.spinBox_class, 0, 2, 1, 1)
        self.spinBox_selection_size_z = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_selection_size_z.sizePolicy().hasHeightForWidth())
        self.spinBox_selection_size_z.setSizePolicy(sizePolicy)
        self.spinBox_selection_size_z.setMinimum(1)
        self.spinBox_selection_size_z.setMaximum(4096)
        self.spinBox_selection_size_z.setSingleStep(1)
        self.spinBox_selection_size_z.setProperty("value", 16)
        self.spinBox_selection_size_z.setObjectName("spinBox_selection_size_z")
        self.gridLayout_selection.addWidget(self.spinBox_selection_size_z, 6, 2, 1, 1)
        self.spinBox_label_opacity = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_label_opacity.setMaximum(255)
        self.spinBox_label_opacity.setProperty("value", 200)
        self.spinBox_label_opacity.setObjectName("spinBox_label_opacity")
        self.gridLayout_selection.addWidget(self.spinBox_label_opacity, 1, 2, 1, 1)
        self.label_selection = QtWidgets.QLabel(self.centralwidget)
        self.label_selection.setObjectName("label_selection")
        self.gridLayout_selection.addWidget(self.label_selection, 2, 0, 1, 2)
        self.spinBox_selection_min_y = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_selection_min_y.sizePolicy().hasHeightForWidth())
        self.spinBox_selection_min_y.setSizePolicy(sizePolicy)
        self.spinBox_selection_min_y.setMaximum(0)
        self.spinBox_selection_min_y.setSingleStep(1)
        self.spinBox_selection_min_y.setObjectName("spinBox_selection_min_y")
        self.gridLayout_selection.addWidget(self.spinBox_selection_min_y, 5, 1, 1, 1)
        self.label_selection_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_selection_pos.setAlignment(QtCore.Qt.AlignCenter)
        self.label_selection_pos.setObjectName("label_selection_pos")
        self.gridLayout_selection.addWidget(self.label_selection_pos, 3, 1, 1, 1)
        self.label_selection_x = QtWidgets.QLabel(self.centralwidget)
        self.label_selection_x.setObjectName("label_selection_x")
        self.gridLayout_selection.addWidget(self.label_selection_x, 4, 0, 1, 1)
        self.spinBox_selection_size_x = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_selection_size_x.sizePolicy().hasHeightForWidth())
        self.spinBox_selection_size_x.setSizePolicy(sizePolicy)
        self.spinBox_selection_size_x.setMinimum(32)
        self.spinBox_selection_size_x.setMaximum(4096)
        self.spinBox_selection_size_x.setSingleStep(32)
        self.spinBox_selection_size_x.setProperty("value", 256)
        self.spinBox_selection_size_x.setObjectName("spinBox_selection_size_x")
        self.gridLayout_selection.addWidget(self.spinBox_selection_size_x, 4, 2, 1, 1)
        self.label_selection_z = QtWidgets.QLabel(self.centralwidget)
        self.label_selection_z.setObjectName("label_selection_z")
        self.gridLayout_selection.addWidget(self.label_selection_z, 6, 0, 1, 1)
        self.label_label_opacity = QtWidgets.QLabel(self.centralwidget)
        self.label_label_opacity.setObjectName("label_label_opacity")
        self.gridLayout_selection.addWidget(self.label_label_opacity, 1, 0, 1, 2)
        self.label_selection_y = QtWidgets.QLabel(self.centralwidget)
        self.label_selection_y.setObjectName("label_selection_y")
        self.gridLayout_selection.addWidget(self.label_selection_y, 5, 0, 1, 1)
        self.spinBox_selection_min_x = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_selection_min_x.sizePolicy().hasHeightForWidth())
        self.spinBox_selection_min_x.setSizePolicy(sizePolicy)
        self.spinBox_selection_min_x.setMaximum(0)
        self.spinBox_selection_min_x.setSingleStep(1)
        self.spinBox_selection_min_x.setObjectName("spinBox_selection_min_x")
        self.gridLayout_selection.addWidget(self.spinBox_selection_min_x, 4, 1, 1, 1)
        self.spinBox_selection_min_z = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_selection_min_z.sizePolicy().hasHeightForWidth())
        self.spinBox_selection_min_z.setSizePolicy(sizePolicy)
        self.spinBox_selection_min_z.setMaximum(0)
        self.spinBox_selection_min_z.setSingleStep(1)
        self.spinBox_selection_min_z.setObjectName("spinBox_selection_min_z")
        self.gridLayout_selection.addWidget(self.spinBox_selection_min_z, 6, 1, 1, 1)
        self.pushButton_crop_selection = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_crop_selection.setObjectName("pushButton_crop_selection")
        self.gridLayout_selection.addWidget(self.pushButton_crop_selection, 2, 2, 1, 1)
        self.checkBox_overlap = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_overlap.setChecked(True)
        self.checkBox_overlap.setObjectName("checkBox_overlap")
        self.gridLayout_selection.addWidget(self.checkBox_overlap, 8, 0, 1, 3)
        self.label_automatic_segmentation = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_automatic_segmentation.setFont(font)
        self.label_automatic_segmentation.setAlignment(QtCore.Qt.AlignCenter)
        self.label_automatic_segmentation.setObjectName("label_automatic_segmentation")
        self.gridLayout_selection.addWidget(self.label_automatic_segmentation, 7, 0, 1, 3)
        self.gridLayout_main_left.addLayout(self.gridLayout_selection, 8, 0, 1, 3)
        self.label_models = QtWidgets.QLabel(self.centralwidget)
        self.label_models.setObjectName("label_models")
        self.gridLayout_main_left.addWidget(self.label_models, 3, 0, 1, 1)
        self.pushButton_remove_image = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_remove_image.setEnabled(False)
        self.pushButton_remove_image.setMaximumSize(QtCore.QSize(25, 16777215))
        self.pushButton_remove_image.setObjectName("pushButton_remove_image")
        self.gridLayout_main_left.addWidget(self.pushButton_remove_image, 1, 2, 1, 1)
        self.label_images = QtWidgets.QLabel(self.centralwidget)
        self.label_images.setObjectName("label_images")
        self.gridLayout_main_left.addWidget(self.label_images, 1, 0, 1, 1)
        self.pushButton_add_model = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_model.setMaximumSize(QtCore.QSize(25, 16777215))
        self.pushButton_add_model.setObjectName("pushButton_add_model")
        self.gridLayout_main_left.addWidget(self.pushButton_add_model, 3, 1, 1, 1)
        self.listWidget_images = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_images.setObjectName("listWidget_images")
        self.gridLayout_main_left.addWidget(self.listWidget_images, 2, 0, 1, 3)
        self.pushButton_add_label = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_add_label.setEnabled(True)
        self.pushButton_add_label.setMaximumSize(QtCore.QSize(25, 16777215))
        self.pushButton_add_label.setObjectName("pushButton_add_label")
        self.gridLayout_main_left.addWidget(self.pushButton_add_label, 5, 1, 1, 1)
        self.listWidget_labels = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget_labels.setObjectName("listWidget_labels")
        self.gridLayout_main_left.addWidget(self.listWidget_labels, 6, 0, 1, 3)
        self.label_labels = QtWidgets.QLabel(self.centralwidget)
        self.label_labels.setObjectName("label_labels")
        self.gridLayout_main_left.addWidget(self.label_labels, 5, 0, 1, 1)
        self.pushButton_export_label = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_export_label.setEnabled(True)
        self.pushButton_export_label.setMaximumSize(QtCore.QSize(25, 16777215))
        self.pushButton_export_label.setObjectName("pushButton_export_label")
        self.gridLayout_main_left.addWidget(self.pushButton_export_label, 5, 2, 1, 1)
        self.pushButton_remove_model = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_remove_model.setEnabled(False)
        self.pushButton_remove_model.setMaximumSize(QtCore.QSize(25, 16777215))
        self.pushButton_remove_model.setObjectName("pushButton_remove_model")
        self.gridLayout_main_left.addWidget(self.pushButton_remove_model, 3, 2, 1, 1)
        self.horizontalLayout_predict = QtWidgets.QHBoxLayout()
        self.horizontalLayout_predict.setObjectName("horizontalLayout_predict")
        self.pushButton_predict_selection = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_predict_selection.sizePolicy().hasHeightForWidth())
        self.pushButton_predict_selection.setSizePolicy(sizePolicy)
        self.pushButton_predict_selection.setObjectName("pushButton_predict_selection")
        self.horizontalLayout_predict.addWidget(self.pushButton_predict_selection)
        self.pushButton_predict_image = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_predict_image.sizePolicy().hasHeightForWidth())
        self.pushButton_predict_image.setSizePolicy(sizePolicy)
        self.pushButton_predict_image.setObjectName("pushButton_predict_image")
        self.horizontalLayout_predict.addWidget(self.pushButton_predict_image)
        self.gridLayout_main_left.addLayout(self.horizontalLayout_predict, 9, 0, 1, 3)
        self.checkBox_show_label = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_show_label.setChecked(True)
        self.checkBox_show_label.setObjectName("checkBox_show_label")
        self.gridLayout_main_left.addWidget(self.checkBox_show_label, 7, 0, 1, 3)
        self.gridLayout_centralwidget.addLayout(self.gridLayout_main_left, 0, 0, 4, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_interactive_segmentation = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_interactive_segmentation.setFont(font)
        self.label_interactive_segmentation.setAlignment(QtCore.Qt.AlignCenter)
        self.label_interactive_segmentation.setObjectName("label_interactive_segmentation")
        self.gridLayout.addWidget(self.label_interactive_segmentation, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.pushButton_interactive_segmentation = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_interactive_segmentation.setObjectName("pushButton_interactive_segmentation")
        self.gridLayout.addWidget(self.pushButton_interactive_segmentation, 1, 0, 1, 1)
        self.label_interaction_map_view_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label_interaction_map_view_2.sizePolicy().hasHeightForWidth())
        self.label_interaction_map_view_2.setSizePolicy(sizePolicy)
        self.label_interaction_map_view_2.setText("")
        self.label_interaction_map_view_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_interaction_map_view_2.setObjectName("label_interaction_map_view_2")
        self.gridLayout.addWidget(self.label_interaction_map_view_2, 5, 0, 1, 1)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout.addWidget(self.listWidget, 3, 0, 1, 1)
        self.label_interaction_map_view_1 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label_interaction_map_view_1.sizePolicy().hasHeightForWidth())
        self.label_interaction_map_view_1.setSizePolicy(sizePolicy)
        self.label_interaction_map_view_1.setText("")
        self.label_interaction_map_view_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_interaction_map_view_1.setObjectName("label_interaction_map_view_1")
        self.gridLayout.addWidget(self.label_interaction_map_view_1, 4, 0, 1, 1)
        self.gridLayout_centralwidget.addLayout(self.gridLayout, 0, 5, 4, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.pushButton_add_image.clicked.connect(MainWindow.dialog_add_image)
        self.pushButton_add_model.clicked.connect(MainWindow.dialog_add_model)
        self.pushButton_predict_selection.clicked.connect(MainWindow.predict_selection)
        self.verticalSlider_pos_y.valueChanged['int'].connect(MainWindow.change_pos)
        self.verticalSlider_zoom.valueChanged['int'].connect(MainWindow.change_zoom)
        self.horizontalSlider_pos_x.valueChanged['int'].connect(MainWindow.change_pos)
        self.horizontalSlider_slice.valueChanged['int'].connect(MainWindow.change_slice)
        self.listWidget_images.itemSelectionChanged.connect(MainWindow.change_selected_image)
        self.listWidget_labels.itemSelectionChanged.connect(MainWindow.change_selected_label)
        self.listWidget_models.itemSelectionChanged.connect(MainWindow.change_selected_model)
        self.pushButton_crop_selection.clicked.connect(MainWindow.crop_selection)
        self.spinBox_selection_size_x.valueChanged['int'].connect(MainWindow.update_selection)
        self.spinBox_selection_size_y.valueChanged['int'].connect(MainWindow.update_selection)
        self.spinBox_selection_min_x.valueChanged['int'].connect(MainWindow.update_selection)
        self.spinBox_selection_min_y.valueChanged['int'].connect(MainWindow.update_selection)
        self.spinBox_selection_min_z.valueChanged['int'].connect(MainWindow.update_selection)
        self.spinBox_selection_size_z.valueChanged['int'].connect(MainWindow.update_selection)
        self.spinBox_class.valueChanged['int'].connect(MainWindow.change_class)
        self.spinBox_label_opacity.valueChanged['int'].connect(MainWindow.change_label_opacity)
        self.pushButton_predict_image.clicked.connect(MainWindow.predict_full_image)
        self.pushButton_add_label.clicked.connect(MainWindow.dialog_add_label)
        self.pushButton_export_label.clicked.connect(MainWindow.export_selected_label)
        self.label_main_view.mousePress['QMouseEvent'].connect(MainWindow.main_view_mouse_event)
        self.label_main_view.mouseRelease['QMouseEvent'].connect(MainWindow.main_view_mouse_event)
        self.label_main_view.mouseMove['QMouseEvent'].connect(MainWindow.main_view_mouse_event)
        self.label_main_view.wheel['QWheelEvent'].connect(MainWindow.main_view_wheel_event)
        self.pushButton_interactive_segmentation.clicked.connect(MainWindow.interactive_segmentation_image)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.pushButton_add_image, self.pushButton_remove_image)
        MainWindow.setTabOrder(self.pushButton_remove_image, self.listWidget_images)
        MainWindow.setTabOrder(self.listWidget_images, self.pushButton_add_model)
        MainWindow.setTabOrder(self.pushButton_add_model, self.pushButton_remove_model)
        MainWindow.setTabOrder(self.pushButton_remove_model, self.listWidget_models)
        MainWindow.setTabOrder(self.listWidget_models, self.pushButton_add_label)
        MainWindow.setTabOrder(self.pushButton_add_label, self.pushButton_export_label)
        MainWindow.setTabOrder(self.pushButton_export_label, self.listWidget_labels)
        MainWindow.setTabOrder(self.listWidget_labels, self.spinBox_selection_min_x)
        MainWindow.setTabOrder(self.spinBox_selection_min_x, self.spinBox_selection_size_x)
        MainWindow.setTabOrder(self.spinBox_selection_size_x, self.spinBox_selection_min_y)
        MainWindow.setTabOrder(self.spinBox_selection_min_y, self.spinBox_selection_size_y)
        MainWindow.setTabOrder(self.spinBox_selection_size_y, self.spinBox_selection_min_z)
        MainWindow.setTabOrder(self.spinBox_selection_min_z, self.spinBox_selection_size_z)
        MainWindow.setTabOrder(self.spinBox_selection_size_z, self.pushButton_predict_selection)
        MainWindow.setTabOrder(self.pushButton_predict_selection, self.horizontalSlider_pos_x)
        MainWindow.setTabOrder(self.horizontalSlider_pos_x, self.verticalSlider_pos_y)
        MainWindow.setTabOrder(self.verticalSlider_pos_y, self.horizontalSlider_slice)
        MainWindow.setTabOrder(self.horizontalSlider_slice, self.verticalSlider_zoom)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cutter"))
        self.label_slice.setText(_translate("MainWindow", "0/0"))
        self.label_zoom.setText(_translate("MainWindow", "0%"))
        self.pushButton_add_image.setText(_translate("MainWindow", "+"))
        self.label_class.setText(_translate("MainWindow", "Classe affiché"))
        self.label_selection_size.setText(_translate("MainWindow", "taille"))
        self.label_selection.setText(_translate("MainWindow", "Sélection"))
        self.label_selection_pos.setText(_translate("MainWindow", "position"))
        self.label_selection_x.setText(_translate("MainWindow", "X"))
        self.label_selection_z.setText(_translate("MainWindow", "Z"))
        self.label_label_opacity.setText(_translate("MainWindow", "Opacité"))
        self.label_selection_y.setText(_translate("MainWindow", "Y"))
        self.pushButton_crop_selection.setText(_translate("MainWindow", "recadrer la sélection"))
        self.checkBox_overlap.setText(_translate("MainWindow", "Prédiction avec chevauchement"))
        self.label_automatic_segmentation.setText(_translate("MainWindow", "Segmentation Automatique"))
        self.label_models.setText(_translate("MainWindow", "Modèles"))
        self.pushButton_remove_image.setText(_translate("MainWindow", "-"))
        self.label_images.setText(_translate("MainWindow", "Images"))
        self.pushButton_add_model.setText(_translate("MainWindow", "+"))
        self.pushButton_add_label.setText(_translate("MainWindow", "+"))
        self.label_labels.setText(_translate("MainWindow", "Etiquettes"))
        self.pushButton_export_label.setText(_translate("MainWindow", "↗"))
        self.pushButton_remove_model.setText(_translate("MainWindow", "-"))
        self.pushButton_predict_selection.setText(_translate("MainWindow", "Prédire sélection"))
        self.pushButton_predict_image.setText(_translate("MainWindow", "Prédire image"))
        self.checkBox_show_label.setText(_translate("MainWindow", "Afficher etiquette"))
        self.label_interactive_segmentation.setText(_translate("MainWindow", "Segmentation Interactive"))
        self.label.setText(_translate("MainWindow", "Masques"))
        self.pushButton_interactive_segmentation.setText(_translate("MainWindow", "Segmenter l\'image"))
from InteractiveQLabel import InteractiveQLabel
