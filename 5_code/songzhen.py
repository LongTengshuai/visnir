# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'songzhen.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1236, 1072)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_13.setGeometry(QtCore.QRect(0, 0, 81, 21))
        self.pushButton_13.setObjectName("pushButton_13")
        self.horizontalWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalWidget.setGeometry(QtCore.QRect(0, 20, 491, 491))
        self.horizontalWidget.setStyleSheet("border: 1px solid rgb(41, 57, 85)")
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setContentsMargins(1, 1, 0, 1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalWidget_2.setGeometry(QtCore.QRect(0, 530, 491, 491))
        self.horizontalWidget_2.setStyleSheet("border: 1px solid rgb(41, 57, 85)")
        self.horizontalWidget_2.setObjectName("horizontalWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalWidget_2)
        self.horizontalLayout_2.setContentsMargins(1, 1, 0, 1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalWidget_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.horizontalWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalWidget_3.setGeometry(QtCore.QRect(510, 530, 721, 491))
        self.horizontalWidget_3.setStyleSheet("border: 1px solid rgb(41, 57, 85)")
        self.horizontalWidget_3.setObjectName("horizontalWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalWidget_3)
        self.horizontalLayout_3.setContentsMargins(1, 1, 0, 1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_17 = QtWidgets.QLabel(self.horizontalWidget_3)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_3.addWidget(self.label_17)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(510, 20, 391, 491))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.pushButton_8 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout_8.addWidget(self.pushButton_8, 4, 1, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.pushButton_4 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_6.addWidget(self.pushButton_4, 4, 1, 1, 1)
        self.horizontalSlider_4 = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.horizontalSlider_4.setMinimum(1)
        self.horizontalSlider_4.setMaximum(10)
        self.horizontalSlider_4.setPageStep(1)
        self.horizontalSlider_4.setProperty("value", 2)
        self.horizontalSlider_4.setSliderPosition(2)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.gridLayout_6.addWidget(self.horizontalSlider_4, 3, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_6.addWidget(self.label_8, 2, 3, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.horizontalSlider.setMinimum(1)
        self.horizontalSlider.setMaximum(176)
        self.horizontalSlider.setPageStep(1)
        self.horizontalSlider.setProperty("value", 103)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout_6.addWidget(self.horizontalSlider, 0, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout_6.addWidget(self.label_4, 0, 3, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 5, 3, 1, 1)
        self.horizontalSlider_6 = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.horizontalSlider_6.setMinimum(1)
        self.horizontalSlider_6.setMaximum(10)
        self.horizontalSlider_6.setPageStep(1)
        self.horizontalSlider_6.setProperty("value", 1)
        self.horizontalSlider_6.setSliderPosition(1)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.gridLayout_6.addWidget(self.horizontalSlider_6, 6, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_6.addWidget(self.label_6, 1, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 3, 3, 1, 1)
        self.horizontalSlider_5 = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.horizontalSlider_5.setMinimum(1)
        self.horizontalSlider_5.setMaximum(10)
        self.horizontalSlider_5.setPageStep(1)
        self.horizontalSlider_5.setProperty("value", 2)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.gridLayout_6.addWidget(self.horizontalSlider_5, 5, 2, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout_6.addWidget(self.pushButton_5, 5, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_6.addWidget(self.pushButton_3, 2, 1, 1, 1)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.horizontalSlider_2.setMinimum(1)
        self.horizontalSlider_2.setMaximum(255)
        self.horizontalSlider_2.setPageStep(1)
        self.horizontalSlider_2.setProperty("value", 107)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.gridLayout_6.addWidget(self.horizontalSlider_2, 1, 2, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_6.addWidget(self.pushButton_2, 1, 1, 1, 1)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.gridLayoutWidget_2)
        self.horizontalSlider_3.setMinimum(1)
        self.horizontalSlider_3.setMaximum(10)
        self.horizontalSlider_3.setPageStep(1)
        self.horizontalSlider_3.setProperty("value", 2)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.gridLayout_6.addWidget(self.horizontalSlider_3, 2, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_6.addWidget(self.label_14, 6, 3, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_6.addWidget(self.pushButton, 0, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_6, 0, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_3, 2, 1, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_8.addWidget(self.pushButton_7, 3, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1236, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_13.setText(_translate("MainWindow", "Data"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_17.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_8.setText(_translate("MainWindow", "高光谱数据预览"))
        self.pushButton_4.setText(_translate("MainWindow", "手动去除区域"))
        self.label_8.setText(_translate("MainWindow", "腐蚀：2     "))
        self.label_4.setText(_translate("MainWindow", "波段103     "))
        self.label_12.setText(_translate("MainWindow", "腐蚀：2     "))
        self.label_6.setText(_translate("MainWindow", "阈值：107   "))
        self.label_10.setText(_translate("MainWindow", "膨胀：2     "))
        self.pushButton_5.setText(_translate("MainWindow", "截断连通域  "))
        self.pushButton_3.setText(_translate("MainWindow", "  去除散点  "))
        self.pushButton_2.setText(_translate("MainWindow", "  阈值选取  "))
        self.label_14.setText(_translate("MainWindow", "膨胀：1     "))
        self.pushButton.setText(_translate("MainWindow", "  去除背景  "))
        self.pushButton_7.setText(_translate("MainWindow", "获取高光谱数据"))