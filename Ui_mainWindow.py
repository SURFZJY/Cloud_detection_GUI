# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\liu\eric6_workspace\cloud_detection\mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.setWindowModality(QtCore.Qt.NonModal)
        mainWindow.resize(1029, 914)
        self.centralWidget = QtWidgets.QWidget(mainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.label_4 = QtWidgets.QLabel(self.centralWidget)
        self.label_4.setGeometry(QtCore.QRect(560, 180, 101, 51))
        self.label_4.setStyleSheet("font: 75 36pt \"Agency FB\";")
        self.label_4.setObjectName("label_4")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(510, 430, 491, 451))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_4 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_4.setStyleSheet("font: 75 20pt \"仿宋\";")
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 1, 0, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_5.setStyleSheet("font: 75 20pt \"仿宋\";")
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout.addWidget(self.pushButton_5, 2, 0, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.gridLayoutWidget)
        self.spinBox.setStyleSheet("font: 20pt \"Agency FB\";")
        self.spinBox.setProperty("value", 30)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 2, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_3.setStyleSheet("font: 75 20pt \"仿宋\";")
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 0, 0, 1, 1)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox.setMouseTracking(False)
        self.doubleSpinBox.setStyleSheet("font: 20pt \"Agency FB\";")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMaximum(2.0)
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox.setProperty("value", 0.6)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout.addWidget(self.doubleSpinBox, 0, 1, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_6.setStyleSheet("font: 75 20pt \"仿宋\";")
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 3, 0, 1, 2)
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton.setStyleSheet("font: 75 20pt \"华文细黑\";")
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 8, 1, 1, 1)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.doubleSpinBox_2.setStyleSheet("font: 20pt \"Agency FB\";")
        self.doubleSpinBox_2.setDecimals(1)
        self.doubleSpinBox_2.setMaximum(2.0)
        self.doubleSpinBox_2.setSingleStep(0.1)
        self.doubleSpinBox_2.setProperty("value", 1.3)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.gridLayout.addWidget(self.doubleSpinBox_2, 1, 1, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_7.setStyleSheet("font: 75 20pt \"仿宋\";")
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout.addWidget(self.pushButton_7, 6, 0, 1, 2)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_2.setStyleSheet("font: 75 20pt \"华文细黑\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 8, 0, 1, 1)
        self.pushButton_8 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_8.setStyleSheet("font: 75 20pt \"仿宋\";")
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout.addWidget(self.pushButton_8, 4, 0, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.centralWidget)
        self.label_6.setGeometry(QtCore.QRect(710, 180, 251, 61))
        self.label_6.setStyleSheet("border-width: 2px;\n"
"border-style: solid;\n"
"border-color: rdb(0,0,0);\n"
"font: 36pt \"仿宋\";")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 481, 431))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("border-width: 1px;\n"
"border-style: solid;\n"
"border-color: rdb(0,0,0);\n"
"")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(QtCore.QRect(10, 440, 481, 441))
        self.label_2.setStyleSheet("border-width: 1px;\n"
"border-style: solid;\n"
"border-color: rdb(0,0,0);\n"
"")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        mainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(mainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1029, 23))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menuBar)
        self.menu_2.setObjectName("menu_2")
        mainWindow.setMenuBar(self.menuBar)
        self.actionOpen = QtWidgets.QAction(mainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/normal/icons/打开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(mainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/normal/icons/保存.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon1)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(mainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/normal/icons/退出.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionExit.setIcon(icon2)
        self.actionExit.setObjectName("actionExit")
        self.actionAbout = QtWidgets.QAction(mainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/normal/icons/关于.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbout.setIcon(icon3)
        self.actionAbout.setObjectName("actionAbout")
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionExit)
        self.menu_2.addAction(self.actionAbout)
        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menu_2.menuAction())

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "云检测算法展示平台"))
        mainWindow.setStatusTip(_translate("mainWindow", "选择图片加载到显示区域"))
        self.label_4.setText(_translate("mainWindow", "云量"))
        self.pushButton_4.setToolTip(_translate("mainWindow", "基于B/R的值来区分：B/R<Threshold为云，否则为天空"))
        self.pushButton_4.setStatusTip(_translate("mainWindow", "基于B/R的值来区分：B/R<Threshold为云，否则为天空"))
        self.pushButton_4.setText(_translate("mainWindow", "阈值法2 B/R"))
        self.pushButton_5.setToolTip(_translate("mainWindow", "基于B-R的值来区分：B-R>Threshold为云，否则为天空"))
        self.pushButton_5.setStatusTip(_translate("mainWindow", "基于B-R的值来区分：B-R>Threshold为云，否则为天空"))
        self.pushButton_5.setText(_translate("mainWindow", "阈值法3 B-R"))
        self.pushButton_3.setToolTip(_translate("mainWindow", "基于R/B的值来区分：R/B>Threshold为云，否则为天空"))
        self.pushButton_3.setStatusTip(_translate("mainWindow", "基于R/B的值来区分：R/B>Threshold为云，否则为天空"))
        self.pushButton_3.setText(_translate("mainWindow", "阈值法1 R/B"))
        self.pushButton_6.setToolTip(_translate("mainWindow", "基于R-B的值来区分：用Otsu法计算出合理阈值来区分云和天空"))
        self.pushButton_6.setStatusTip(_translate("mainWindow", "基于R-B的值来区分：用Otsu法计算出合理阈值来区分云和天空"))
        self.pushButton_6.setText(_translate("mainWindow", "基于R-B的自适应阈值法"))
        self.pushButton.setToolTip(_translate("mainWindow", "保存分割后的图片"))
        self.pushButton.setStatusTip(_translate("mainWindow", "保存分割后的图片"))
        self.pushButton.setText(_translate("mainWindow", "保存"))
        self.pushButton_7.setToolTip(_translate("mainWindow", "基于图割理论区分云和天空"))
        self.pushButton_7.setStatusTip(_translate("mainWindow", "基于图割理论区分云和天空"))
        self.pushButton_7.setText(_translate("mainWindow", "图割法"))
        self.pushButton_2.setText(_translate("mainWindow", "打开"))
        self.pushButton_8.setText(_translate("mainWindow", "基于R/B的自适应阈值法"))
        self.menu.setTitle(_translate("mainWindow", "文件"))
        self.menu_2.setTitle(_translate("mainWindow", "关于"))
        self.actionOpen.setText(_translate("mainWindow", "打开"))
        self.actionSave.setText(_translate("mainWindow", "保存"))
        self.actionExit.setText(_translate("mainWindow", "退出"))
        self.actionAbout.setText(_translate("mainWindow", "About"))

import icon_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
