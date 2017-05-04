# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog
from PyQt5 import QtCore, QtWidgets
from Ui_mainWindow import Ui_mainWindow
from PyQt5.QtGui import QImage, QPixmap

import os
import cv2
import numpy as np 

# from keras.preprocessing.image import load_img , img_to_array
# from keras.models import load_model

def cvimg2pixmap(cv2img):
    img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    q_img = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    q_pixmap = QPixmap.fromImage(q_img)
    return q_pixmap

def RB_method_1(input_img, thresh=0.6):
    img = input_img
    b, g, r = cv2.split(img) 
    B = np.array(b)
    R = np.array(r) 
    tmp = R / (B + 0.0001)
    cot = 0 
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if tmp[i][j] > thresh:
                tmp[i][j] = 255
                cot += 1
            else: 
                tmp[i][j] = 0
    cover = cot / (tmp.shape[0] * tmp.shape[1])                
    return tmp, cover

def RB_method_2(input_img, thresh=1.3):
    img = input_img
    b, g, r = cv2.split(img) 
    B = np.array(b)
    R = np.array(r) 
    tmp = B / (R + 0.001) 
    cot = 0
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if tmp[i][j] < thresh:
                tmp[i][j] = 255
                cot += 1
            else: 
                tmp[i][j] = 0
    cover = cot / (tmp.shape[0] * tmp.shape[1]) 
    return tmp, cover

def RB_method_3(input_img, thresh=30):
    img = input_img
    b, g, r = cv2.split(img) 
    B = np.array(b)
    R = np.array(r) 
    tmp = B - R
    tmpx = tmp.astype(np.int16)   
    cot = 0 
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if tmpx[i][j] > thresh:
                tmp[i][j] = 0
            else: 
                tmp[i][j] = 255   
                cot += 1
    cover = cot / (tmp.shape[0] * tmp.shape[1])   
    return tmp, cover

def RB_method_4(input_img):
    img = input_img
    b, g, r = cv2.split(img) 
    B = np.array(b)
    R = np.array(r) 
    tmp = R - B
    tmp.dtype=np.int8
    tmp = tmp.astype(np.int16)
    tmp = tmp-tmp.min()
    tmp = tmp.astype(np.uint8)
    thres, res = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cot = 0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i][j] == 255:
                cot += 1
    cover = cot / (res.shape[0]*res.shape[1])
    return res, cover

def RB_method_5(input_img):
    img = input_img
    b, g, r = cv2.split(img) 
    B = np.array(b)
    R = np.array(r) 
    tmp = R / (B + 0.0001)
    x = (tmp-tmp.min())/(tmp.max()-tmp.min())*255
    x = np.round(x)
    x = x.astype(np.uint8)
    T, res = cv2.threshold(x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    theta = 1.3
    hand_constraint = np.zeros((x.shape[0],x.shape[1]), np.uint8)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] > theta * T:
                hand_constraint[i][j] = 1
            elif x[i][j] < 1 / theta * T:
                hand_constraint[i][j] = 0
            else:
                hand_constraint[i][j] = 2
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(img, hand_constraint, None, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK )
    mask = np.where((mask==1),1,0).astype('uint8')
    mask = mask * 255
    
    cot = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 255:
                cot += 1
    cover = cot / (mask.shape[0] * mask.shape[1])  
    return mask, cover

def RB_method_6(input_img):
    img = input_img
    b, g, r = cv2.split(img) 
    B = np.array(b)
    R = np.array(r) 
    tmp = R / (B + 0.0001)
    tmax = tmp.max()
    tmin = tmp.min()
    x = (tmp-tmin)/(tmax-tmin)*255
    x = np.round(x)
    x = x.astype(np.uint8)
    T, res = cv2.threshold(x,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cot = 0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i][j] == 255:
                cot += 1
    cover = cot / (res.shape[0]*res.shape[1])
    return res, cover
    

class MainWindow(QMainWindow, Ui_mainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.cur_pic = None
        self.tmp_pic = None
        self.method_1 = None
        self.method_2 = None
        self.method_3 = None
        self.method_4 = None
        self.method_5 = None
        self.method_6 = None
        self.thresh_1 = 0.6 
        self.thresh_2 = 1.3
        self.thresh_3 = 30
        self.coverdict = {}
#         cur_dir = os.getcwd()
#         model_path = cur_dir + '\\' + 'kiel_tiny_new0419.h5'
#         self.classify_model = load_model(model_path)
    
    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Save the processed image 
        """
        imgName,imgType = QFileDialog.getSaveFileName(self, "保存", "", "*.jpg;;*.png;;*.jpeg;;*.bmp")
        if imgType:
            print(imgName)
            print(type(imgName))
            print(imgType)
            img_suffix = imgType.split('.')[-1]
            cv2.imwrite(imgName + '.' + img_suffix, self.tmp_pic)
            
#         print('success')
        
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Load image and display it on the label widget.
        """
        imgName,imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*)")

        if imgType:
            
            img = cv2.imread(imgName)
            
            self.img_path = imgName
            self.cur_pic = img
            self.tmp_pic = None
            self.method_1 = None
            self.method_2 = None
            self.method_3 = None
            self.method_4 = None
            self.method_5 = None
            self.method_6 = None
            
            q_pixmap = cvimg2pixmap(img)
            q_pixmap_resized = q_pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
    
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setPixmap(q_pixmap_resized)
            
#             if self.classify_model:
#                 model = self.classify_model
#                 resized_img = cv2.resize(self.cur_pic, (200, 200), interpolation=cv2.INTER_AREA)
#                 cv2.imwrite('tmp.jpg', resized_img)
#                 img = load_img('tmp.jpg')
#                 img = img_to_array(img)
#                 img /= 255
#                 img = img[np.newaxis, ...]
#                 res = model.predict(img)
#                 pred_label = np.argmax(res)
#                 predict_name_list = ['Cummulus 积云', 'Cirrus 卷云', 'Altocumulus 高积云', 'Clearsky 晴空', 'Stratocumulus 层积云', 'Stratus 层云', 'Cumulonimbus 积雨云']
#                 name = predict_name_list[pred_label]
#                 self.label_7.setText(name)
            
#         print('Image loaded !')
            
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        """
        RB_Method_1
        """
        if self.cur_pic is not None:
            if self.method_1 is None:
                self.tmp_pic = RB_method_1(self.cur_pic, self.thresh_1)[0]
                self.coverdict['m1'] = RB_method_1(self.cur_pic, self.thresh_1)[1]
                path = self.img_path.split('.')[0] + '_Ratio_0.6.jpg'
                cv2.imwrite(path, self.tmp_pic)
                cut_img = cv2.imread(path)
                self.method_1 = cut_img
                os.remove(path)
            else:
                cut_img = self.method_1
            
            self.label_6.setNum(self.coverdict['m1'])
            
            q_pixmap = cvimg2pixmap(cut_img)
            q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setPixmap(q_pixmap_resized)
        
#         print('RB_Method_1 succeed')

    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        """
        RB_Method_2
        """
        if self.cur_pic is not None:
            if self.method_2 is None:
                self.tmp_pic = RB_method_2(self.cur_pic, self.thresh_2)[0] 
                self.coverdict['m2'] = RB_method_2(self.cur_pic, self.thresh_2)[1]             
                path = self.img_path.split('.')[0] + '_Ratio_1.3.jpg'
                cv2.imwrite(path, self.tmp_pic)
                cut_img = cv2.imread(path)
                self.method_2 = cut_img
                os.remove(path)
            else:
                cut_img = self.method_2
                
            self.label_6.setNum(self.coverdict['m2'])
                
            q_pixmap = cvimg2pixmap(cut_img)
            q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setPixmap(q_pixmap_resized)
            
#         print('RB_Method_2 succeed')

    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        """
        RB_Method_3
        """
        if self.cur_pic is not None:
            if self.method_3 is None:
                self.tmp_pic = RB_method_3(self.cur_pic, self.thresh_3)[0]
                self.coverdict['m3'] = RB_method_3(self.cur_pic, self.thresh_3)[1]
                path = self.img_path.split('.')[0] + '_Diff.jpg'
                cv2.imwrite(path, self.tmp_pic)    
                cut_img = cv2.imread(path)
                self.method_3 = cut_img
                os.remove(path)
            else:
                cut_img = self.method_3
                
            self.label_6.setNum(self.coverdict['m3'])
                
            q_pixmap = cvimg2pixmap(cut_img)
            q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)

            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setPixmap(q_pixmap_resized)
        
#         print('RB_Method_3 succeed')
    
    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        """
        RB_Method_4 : R-B Adaptive Threshold Method
        """
        if self.cur_pic is not None:
            if self.method_4 is None:
                self.tmp_pic = RB_method_4(self.cur_pic)[0]
                self.coverdict['m4'] = RB_method_4(self.cur_pic)[1]
                path = self.img_path.split('.')[0] + '_AdapDiff.jpg'
                cv2.imwrite(path, self.tmp_pic)
                cut_img = cv2.imread(path)
                self.method_4 = cut_img
                os.remove(path)
            else: 
                cut_img = self.method_4
            
            self.label_6.setNum(self.coverdict['m4'])
            
            q_pixmap = cvimg2pixmap(cut_img)        
            q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setPixmap(q_pixmap_resized)
            
#         print('RB_Method_4 Adaptive Threshold Method succeed')

    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        """
        RB_Method_5 : AGC Method
        """
        if self.cur_pic is not None:
            if self.method_5 is None:
                self.tmp_pic = RB_method_5(self.cur_pic)[0]
                self.coverdict['m5'] = RB_method_5(self.cur_pic)[1]
                path = self.img_path.split('.')[0] + '_AGC.jpg'
                cv2.imwrite(path, self.tmp_pic)
                cut_img = cv2.imread(path)
                self.method_5 = cut_img
                os.remove(path)
            else:
                cut_img = self.method_5
            
            self.label_6.setNum(self.coverdict['m5'])
    
            q_pixmap = cvimg2pixmap(cut_img)
            q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setPixmap(q_pixmap_resized)
  
#         print('RB_Method_5 AGC Method succeed')
        
    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        """
        RB_Method_6 : R/B Adaptive 
        """
        if self.cur_pic is not None:
            if self.method_6 is None:
                self.tmp_pic = RB_method_6(self.cur_pic)[0]
                self.coverdict['m6'] = RB_method_6(self.cur_pic)[1]
                path = self.img_path.split('.')[0] + '_AdapRatio.jpg'
                cv2.imwrite(path, self.tmp_pic)
                cut_img = cv2.imread(path)
                self.method_6 = cut_img
                os.remove(path)
            else:
                cut_img = self.method_6
            
            self.label_6.setNum(self.coverdict['m6'])
    
            q_pixmap = cvimg2pixmap(cut_img)
            q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.label_2.setPixmap(q_pixmap_resized)
        
    @pyqtSlot()
    def on_actionOpen_triggered(self):
        """
        Open file
        """
        imgName,imgType = QFileDialog.getOpenFileName(self, "Load Image", "", "All Files(*)")
        
        if imgType:
            
            img = cv2.imread(imgName)
            
            self.img_path = imgName
            self.cur_pic = img
            self.tmp_pic = None
            self.method_1 = None
            self.method_2 = None
            self.method_3 = None
            self.method_4 = None
            self.method_5 = None
            self.method_6 = None

            q_pixmap = cvimg2pixmap(img)
            q_pixmap_resized = q_pixmap.scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio)
    
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setPixmap(q_pixmap_resized)

#             if self.classify_model:
#                 model = self.classify_model
#                 resized_img = cv2.resize(self.cur_pic, (200, 200), interpolation=cv2.INTER_AREA)
#                 cv2.imwrite('tmp.jpg', resized_img)
#                 img = load_img('tmp.jpg')
#                 img = img_to_array(img)
#                 img /= 255
#                 img = img[np.newaxis, ...]
#                 res = model.predict(img)
#                 pred_label = np.argmax(res)
#                 predict_name_list = ['Cummulus 积云', 'Cirrus 卷云', 'Altocumulus 高积云', 'Clearsky 晴空', 'Stratocumulus 层积云', 'Stratus 层云', 'Cumulonimbus 积雨云']
#                 name = predict_name_list[pred_label]
#                 self.label_7.setText(name)

#         print('success!')
    
    @pyqtSlot()
    def on_actionSave_triggered(self):
        """
        Save file
        """
        imgName,imgType = QFileDialog.getSaveFileName(self, "Save Image", "", "*.jpg;;*.png;;*.jpeg;;*.bmp")
        if imgType:
            print(imgName)
            print(type(imgName))
            print(imgType)
            img_suffix = imgType.split('.')[-1]
            cv2.imwrite(imgName + '.' + img_suffix, self.tmp_pic)
            
#         print('success')

    
    @pyqtSlot()
    def on_actionExit_triggered(self):
        """
        Exit
        """
        sys.exit(0)
    
    @pyqtSlot()
    def on_actionAbout_triggered(self):
        """
        About Messages
        """
        QMessageBox.about(self,  'About',  'Cloud Detection Algorithm Display')
    
    @pyqtSlot(float)
    def on_doubleSpinBox_valueChanged(self, p0):
        """
        Control RB_Method_1 threshold
        
        @param p0 DESCRIPTION
        @type float
        """
        self.thresh_1 = p0
        self.tmp_pic = RB_method_1(self.cur_pic, self.thresh_1)[0]
        self.coverdict['m1'] = RB_method_1(self.cur_pic, self.thresh_1)[1]
        path = self.img_path.split('.')[0] + '_RB1_tmp.jpg'
        cv2.imwrite(path, self.tmp_pic)
        cut_img = cv2.imread(path)
        self.method_1 = cut_img
        os.remove(path)
        
        q_pixmap = cvimg2pixmap(cut_img)
        q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setPixmap(q_pixmap_resized)
        self.label_6.setNum(self.coverdict['m1'])

    @pyqtSlot(float)
    def on_doubleSpinBox_2_valueChanged(self, p0):
        """
        Control RB_Method_2 threshold
        
        @param p0 DESCRIPTION
        @type float
        """
        self.thresh_2 = p0
        self.tmp_pic = RB_method_2(self.cur_pic, self.thresh_2)[0]
        self.coverdict['m2'] = RB_method_2(self.cur_pic, self.thresh_2)[1]
        path = self.img_path.split('.')[0] + '_RB2_tmp.jpg'
        cv2.imwrite(path, self.tmp_pic)
        cut_img = cv2.imread(path)
        self.method_2 = cut_img
        os.remove(path)
        
        q_pixmap = cvimg2pixmap(cut_img)
        q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setPixmap(q_pixmap_resized)
        self.label_6.setNum(self.coverdict['m2'])
            
    @pyqtSlot(int)
    def on_spinBox_valueChanged(self, p0):
        """
        Control RB_Method_3 threshold
        
        @param p0 DESCRIPTION
        @type int
        """
        self.thresh_3 = p0
        self.tmp_pic = RB_method_3(self.cur_pic, self.thresh_3)[0]
        self.coverdict['m3'] = RB_method_3(self.cur_pic, self.thresh_3)[1]
        path = self.img_path.split('.')[0] + '_RB3_tmp.jpg'
        cv2.imwrite(path, self.tmp_pic)
        cut_img = cv2.imread(path)
        self.method_3 = cut_img
        os.remove(path)
        
        q_pixmap = cvimg2pixmap(cut_img)
        q_pixmap_resized = q_pixmap.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.KeepAspectRatio)
     
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setPixmap(q_pixmap_resized)
        self.label_6.setNum(self.coverdict['m3'])

#     @pyqtSlot()
#     def on_pushButton_9_clicked(self):
#         """
#         classify the cloud  "云类" use kiel 7 classes
#         """
#         if self.classify_model:
#             model = self.classify_model
#             resized_img = cv2.resize(self.cur_pic, (200, 200), interpolation=cv2.INTER_AREA)
#             cv2.imwrite('tmp.jpg', resized_img)
#             img = load_img('tmp.jpg')
#             img = img_to_array(img)
#             img /= 255
#             img = img[np.newaxis, ...]
#             res = model.predict(img)
#             pred_label = np.argmax(res)
#             predict_name_list = ['Cummulus 积云', 'Cirrus 卷云', 'Altocumulus 高积云', 'Clearsky 晴空', 'Stratocumulus 层积云', 'Stratus 层云', 'Cumulonimbus 积雨云']
#             name = predict_name_list[pred_label]
#             self.label_5.setText(name)
        
#     @pyqtSlot()
#     def on_pushButton_10_clicked(self):
#         """
#         choose the model to classify "模型路径"
#         """
#         modelName,modelType = QFileDialog.getOpenFileName(self, "选择分类模型", "", "*.h5")
#         
#         suffix = modelName.split('.')[-1]
#         print(modelName,modelType)
#         print(suffix)
#         if modelType and suffix=='h5':
#             
#             self.classify_model = load_model(modelName)
#             self.label_3.setText(modelName.split('/')[-1])
#             
#         print('success!')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
    

