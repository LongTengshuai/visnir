#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: UTF-8 -*-
from PIL import Image
from skimage import data, exposure, img_as_float
import cv2
import os
import shutil
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import imutils
import apprcc
import pywt
import math
import random

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog
from Leaf import Ui_MainWindow

from PyQt5.Qt import *
from functools import partial
import sys,os

import xlsxwriter  
import xlrd  

import pandas as pd
from pandas import Series,DataFrame
from spectral import *
import spectral.io.envi as envi


# In[2]:


class ROI(QMainWindow,Ui_MainWindow):
    def __init__(self):       
        super(ROI,self).__init__()
        self.setupUi(self)      
        self.cwd=os.getcwd()

        #掩膜1（确定大致区域）\n",
        self.horizontalSlider.valueChanged.connect(self.valuechange)
        self.horizontalSlider_2.valueChanged.connect(self.valuechange_2)
        self.horizontalSlider_3.valueChanged.connect(self.valuechange_3)
        self.horizontalSlider_4.valueChanged.connect(self.valuechange_4)
        self.horizontalSlider_5.valueChanged.connect(self.valuechange_5)
        self.horizontalSlider_6.valueChanged.connect(self.valuechange_6)
        self.horizontalSlider_19.valueChanged.connect(self.valuechange_19)

        #掩膜2（去除反射率过大的像素）\n",
        self.horizontalSlider_11.valueChanged.connect(self.valuechange_11)
        self.horizontalSlider_12.valueChanged.connect(self.valuechange_12)
        self.horizontalSlider_7.valueChanged.connect(self.valuechange_7)
        self.horizontalSlider_10.valueChanged.connect(self.valuechange_10)
        self.horizontalSlider_9.valueChanged.connect(self.valuechange_9)

        #掩膜3（去除叶脉）\n",
        self.horizontalSlider_8.valueChanged.connect(self.valuechange_8)
        self.horizontalSlider_13.valueChanged.connect(self.valuechange_13)
        self.horizontalSlider_14.valueChanged.connect(self.valuechange_14)
        self.horizontalSlider_15.valueChanged.connect(self.valuechange_15)
        self.horizontalSlider_16.valueChanged.connect(self.valuechange_16)
        self.horizontalSlider_17.valueChanged.connect(self.valuechange_17)
        self.horizontalSlider_18.valueChanged.connect(self.valuechange_18)
        self.horizontalSlider_21.valueChanged.connect(self.valuechange_21)
        
        #掩膜1\n",
        self.pushButton.clicked.connect(self.push_1)
        self.pushButton_2.clicked.connect(self.push_2)
        self.pushButton_3.clicked.connect(self.push_3)
        self.pushButton_4.clicked.connect(self.push_4)
        self.pushButton_5.clicked.connect(self.push_5)
        self.pushButton_20.clicked.connect(self.push_20)
        
        #掩膜2\n",
        self.pushButton_11.clicked.connect(self.push_11)
        self.pushButton_10.clicked.connect(self.push_10)
        self.pushButton_6.clicked.connect(self.push_6)
        self.pushButton_9.clicked.connect(self.push_9)

        #掩膜3\n",
        self.pushButton_12.clicked.connect(self.push_12)
        self.pushButton_14.clicked.connect(self.push_14)
        self.pushButton_15.clicked.connect(self.push_15) 
        self.pushButton_16.clicked.connect(self.push_16)
        self.pushButton_21.clicked.connect(self.push_21)
        self.pushButton_17.clicked.connect(self.push_17)
        
        #ROI区域\n",
        self.pushButton_19.clicked.connect(self.push_19)
        self.pushButton_8.clicked.connect(self.push_8)
        self.pushButton_7.clicked.connect(self.push_7)
        
        self.pushButton_13.clicked.connect(self.ChoiceImage) 

        self.initUI()

    def initUI(self): #定义初始化界面的方法\n",
        self.setWindowTitle('Leaf')
        self.setWindowIcon(QIcon(':/pic/q.jpg'))
        self.show()     
 
    #滑轴值显示（掩膜1）\n",
    def valuechange(self):
        input1 = self.horizontalSlider.value()
        self.label_4.setText("波段："+str(input1))

    def valuechange_2(self):
        input2 = self.horizontalSlider_2.value()
        self.label_6.setText("阈值："+str(input2))

    def valuechange_3(self):    
        input3 = self.horizontalSlider_3.value()
        self.label_8.setText("腐蚀："+str(input3))
    
    def valuechange_4(self):  
        input4 = self.horizontalSlider_4.value()
        self.label_10.setText("膨胀："+str(input4))
 
    def valuechange_5(self):  
        input5 = self.horizontalSlider_5.value()
        self.label_12.setText("腐蚀："+str(input5))

    def valuechange_6(self):    
        input6 = self.horizontalSlider_6.value()
        self.label_14.setText("膨胀："+str(input6))

    def valuechange_19(self):    
        input19 = self.horizontalSlider_19.value()
        self.label_3.setText("像素："+str(input19))       

    #掩膜2      \n",
    def valuechange_11(self):
        input11 = self.horizontalSlider_11.value()
        self.label_7.setText("阈值："+str(input11))

    def valuechange_12(self):   
        input12 = self.horizontalSlider_12.value()
        self.label_9.setText("腐蚀："+str(input12))
      
    def valuechange_7(self):   
        input7 = self.horizontalSlider_7.value()
        self.label_11.setText("膨胀："+str(input7))

    def valuechange_10(self):   
        input10 = self.horizontalSlider_10.value()
        self.label_13.setText("腐蚀："+str(input10))
  
    def valuechange_9(self):    
        input9 = self.horizontalSlider_9.value()
        self.label_15.setText("膨胀："+str(input9))

                              
    #掩膜3\n",         
    def valuechange_8(self):    
        input8 = self.horizontalSlider_8.value()
        self.label_5.setText("分类："+str(input8))

    def valuechange_13(self):
        input13 = self.horizontalSlider_13.value()
        self.label_16.setText("最大迭代次数："+str(input13))

    def valuechange_14(self):
        input14 = self.horizontalSlider_14.value()
        self.label_18.setText("所需精度："+str(input14))
        
    def valuechange_15(self):   
        input15 = self.horizontalSlider_15.value()
        self.label_19.setText("重复试验次数："+str(input15))

    def valuechange_16(self):   
        input16 = self.horizontalSlider_16.value()
        self.label_20.setText("腐初始中心的选择："+str(input16))
  
    def valuechange_17(self):   
        input17 = self.horizontalSlider_17.value()
        self.label_21.setText("腐蚀："+str(input17))
  
    def valuechange_18(self):   
        input18 = self.horizontalSlider_18.value()
        self.label_22.setText("膨胀："+str(input18))
        
    def valuechange_21(self):   
        input21 = self.horizontalSlider_21.value()
        if input21 ==1:
            self.label_24.setText("去除叶脉：是")
        if input21 ==0:
            self.label_24.setText("去除叶脉：否")   
            
            
    #选择路径\n",
    def ChoiceImage(self):
        global path
        path = QFileDialog.getOpenFileName(self,"选取文件", self.cwd, "Image files (*.jpg *.raw *.png)")
        print(path)   
        
    def push_1(self):
        #第一步：去除背景（选择波段:103）
        global fname
        global fname2
        global img
        input1 = self.horizontalSlider.value()
        path1 = path
        path2 = path1[0]
        fname,fename = os.path.split(path2)
        fname2,fename2 = os.path.splitext(path2)
        img = envi.open('I:/N/header.hdr',fname2+'.raw')
   
        if os.path.isdir(fname+'/Small'):  #判断文件夹dir是否存在\n",
            shutil.rmtree(fname+'/Small', True)  #删除文件夹dir\n",
        os.mkdir(fname+'/Small')   #建立文件夹dir\n",
        
        if os.path.isdir(fname+'/Big'):  
            shutil.rmtree(fname+'/Big', True)  
        os.mkdir(fname+'/Big')  
  
        if os.path.isdir(fname2):  
            shutil.rmtree(fname2, True)  
        os.mkdir(fname2) 

        src = img[:,:,input1] 
        src2 = img[:,:,103] 

        cv2.imwrite(fname+'/Big/1_tezheng.jpg', src*255)
        size = (491, 481) 
        qwe2 = cv2.resize(src2, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/1_tezheng.jpg', qwe2*255) 

        self.label.setPixmap(QPixmap(fname+'/Small/1_tezheng.jpg'))
        self.label_2.setPixmap(QPixmap(fname+'/Small/1_tezheng.jpg'))
        print(fname)
        print(fename)
        print(fname2)
        print(fename2)        
       
    def push_2(self):
        # 第二步：阈值选取（10）
        input1 = self.horizontalSlider.value()
        src = img[:,:,input1]
        
        # 转换为浮点数进行计算
        fsrc = np.array(src, dtype=np.float32) /255.0
        gray = fsrc
 
        # 求取最大值和最小值\n",
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # 得到掩膜\n",
        input2 = self.horizontalSlider_2.value()
        gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
        (thresh, bin_img) = cv2.threshold(gray_u8, input2, 255, cv2.THRESH_BINARY)

        cv2.imwrite(fname+'/Big/2_yuzhi.jpg', bin_img)
        size = (491, 481)  
        bin_img1 = cv2.resize(bin_img, size, interpolation=cv2.INTER_AREA) 
        cv2.imwrite(fname+'/Small/2_yuzhi.jpg', bin_img1)      
        self.label.setPixmap(QPixmap(fname+'/Small/2_yuzhi.jpg'))

    def push_3(self):
        # 第三步：开闭运算去散点\n",
        bin_img1=cv2.imread(fname+'/Big/2_yuzhi.jpg')
        bin_img, g, r = cv2.split(bin_img1)
    
        #设置卷积核\n",
        input1 = self.horizontalSlider_3.value()
        input2 = self.horizontalSlider_4.value()
        input_1 = int(input1)
        input_2 = int(input2)
        n1 = input_1  #5\n",
        n2 = input_2  #10\n",
        kernel1 = np.ones((n1,n1), np.uint8)
        kernel2 = np.ones((n2,n2), np.uint8)
   
        #图像腐蚀，膨胀运算\n",
        erosion = cv2.erode(bin_img, kernel1)
        result = cv2.dilate(erosion, kernel2)
  
        # 得到去除背景后彩色的图像\n",
        cv2.imwrite(fname+'/Big/3_kaibi.jpg',result)
        size = (491, 481)  
        img1 = cv2.resize(result, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/3_kaibi.jpg', img1)
        self.label.setPixmap(QPixmap(fname+'/Small/3_kaibi.jpg'))
    
    def push_4(self):
        # 第四步：手动去除其他茎\n",
        def nothing(x):
            pass    
        def on_mouse(event, x, y, flags, param): # 鼠标左键按下\n",
            img1=cv2.imread(fname+'/Big/3_kaibi.jpg')
            global point1, point2
            img2 = img1.copy()
            if event == cv2.EVENT_LBUTTONDOWN:
                point1 = (x,y)
                cv2.circle(img2, point1, 10, (0,0,0), -1)
                cv2.imshow('image', img2)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 左键保持按下，且进行拖动 \n",
                cv2.rectangle(img2, point1, (x,y), (0,0,0), -1)
                cv2.imshow('image', img2)
            elif event == cv2.EVENT_LBUTTONUP: # 鼠标左键松开
                point2 = (x,y)
                cv2.rectangle(img2, point1, point2, (0,0,0), -1)
                cv2.imshow('image', img2)
                img4=cv2.imwrite(fname+'/Big/3_kaibi.jpg',img2)
        cv2.namedWindow('d')
        switch1='0:OFF\\n1:ON'      
        cv2.createTrackbar(switch1,'d',0,1,nothing)
   
        while(1):
            k=cv2.waitKey(1)&0xFF
            if k==27:
                break
            f=cv2.getTrackbarPos(switch1,'d') 
            # 当t=1时，即已经去除掉了一些点，输出当时的图片\n",
            cv2.setMouseCallback('image', on_mouse)
            img5=cv2.imread(fname+'/Big/3_kaibi.jpg')
            img3 = img5.copy()
            cv2.imshow('image', img3)
            if f==1:
                break
        cv2.destroyAllWindows()
  
        img5=cv2.imread(fname+'/Big/3_kaibi.jpg')
        cv2.imwrite(fname+'/Big/4_qujing.jpg', img5)
        
        #显示图像\n",
        size = (491, 481)  
        binary_1 = cv2.resize(img5, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/4_qujing.jpg', binary_1)        
        self.label.setPixmap(QPixmap(fname+'/Small/4_qujing.jpg'))
    
    def push_5(self):
        # 第五步：开闭运算，去除茎周围的点\n",
        input5 = self.horizontalSlider_5.value()
        input6 = self.horizontalSlider_6.value()
        input_5 = int(input5)
        input_6 = int(input6)      
        n5 = input_5  #5\n",
        n6 = input_6  #5\n",
        kernel5 = np.ones((n5,n5), np.uint8)
        kernel6 = np.ones((n6,n6), np.uint8)
    
        #图像腐蚀，膨胀运算\n",
        binary=cv2.imread(fname+'/Big/4_qujing.jpg')
        erosion2 = cv2.erode(binary, kernel5)
        result2 = cv2.dilate(erosion2, kernel6)
 
        #显示图像\n",
        cv2.imwrite(fname+'/Big/5_jieduan.jpg', result2)
        size = (491, 481)  
        result2_1 = cv2.resize(result2, size, interpolation=cv2.INTER_AREA) 
        cv2.imwrite(fname+'/Small/5_jieduan.jpg', result2_1) 
        self.label.setPixmap(QPixmap(fname+'/Small/5_jieduan.jpg'))
    
    def push_20(self):  
        # 第六步：获取轮廓，我们的目的是分块，因此只使用外层轮廓，使用点序列的形式\n",
        global xmin, ymin, xmax, ymax 
        input19 = self.horizontalSlider_19.value()
        input_19 = int(input19)
        result = cv2.imread(fname+'/Big/5_jieduan.jpg',0)
        yo = cv2.imread(fname+'/Big/1_tezheng.jpg')
        r,g,b=cv2.split(yo)
        color_img = cv2.merge([b & result, g & result , r & result])
        result2,g8,b8=cv2.split(color_img)
        binary_save = np.copy(result2)
        contours, hierarchy = cv2.findContours(binary_save, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #第七步：按面积排序\n",
        areas = np.zeros(len(contours))
        idx = 0
        for cont in contours : 
            areas[idx] = cv2.contourArea(cont)
            idx = idx + 1
        areas_s = cv2.sortIdx(areas, cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
  
        # 第八步：对每个区域进行处理，并从大到小进行排序,保存\n",
   
        for idx in areas_s :
            if areas[idx] < input_19 :
                break
            # 绘制区域图像，通过将ckness设置为-1可以填充整个区域，否则只绘制边缘\n",
            poly_img = np.zeros(binary_save.shape, dtype = np.uint8 )
            cv2.drawContours(poly_img, contours, idx, [255,255,255], -1)
            cv2.drawContours(poly_img, contours, idx, [0,0,0], 2) 
            
            (cnts, _) = cv2.findContours(poly_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            max = np.max(box,axis=0)
            min = np.min(box,axis=0)
            xmin = min[0]
            ymin = min[1]
            xmax = max[0]
            ymax = max[1] 
            box1 = [[xmin,ymax],[xmin,ymin],[xmax,ymin],[xmax,ymax]]
            box2 = np.int0(box1)
 
            color_img2 = cv2.merge([poly_img, poly_img, poly_img])
            cv2.drawContours(color_img2, [box2], 0, (0, 0, 0), 3) 
   
        cv2.imwrite(fname+'/Big/6_lunkuo.jpg', color_img2)
        size = (491, 481)
        poly_img2 = cv2.resize(color_img2, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/6_lunkuo.jpg', poly_img2)         
        self.label.setPixmap(QPixmap(fname+'/Small/6_lunkuo.jpg'))

                                           
    #掩膜2    \n",
    def push_11(self):
        # 第二步：阈值选取（20）\n",
        input_1 = self.horizontalSlider.value()
        src = img[:,:,input_1]
        
        # 转换为浮点数进行计算\n",
        fsrc = np.array(src, dtype=np.float32) /255.0
        gray = fsrc

        # 求取最大值和最小值\n",
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # 得到掩膜\n",
        input11 = self.horizontalSlider_11.value()
        gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
        (thresh, bin_img) = cv2.threshold(gray_u8, input11, 255, cv2.THRESH_BINARY)

        cv2.imwrite(fname+'/Big/2_1_yuzhi.jpg', bin_img)
        size = (491, 481) 
        bin_img1 = cv2.resize(bin_img, size, interpolation=cv2.INTER_AREA) 
        cv2.imwrite(fname+'/Small/2_1_yuzhi.jpg', bin_img1)       
        self.label.setPixmap(QPixmap(fname+'/Small/2_1_yuzhi.jpg'))

    def push_10(self):
        # 第三步：开闭运算去散点\n",
        bin_img1=cv2.imread(fname+'/Big/2_1_yuzhi.jpg')
        bin_img, g, r = cv2.split(bin_img1)

        #设置卷积核\n",
        input12 = self.horizontalSlider_12.value()
        input7 = self.horizontalSlider_7.value()
        input_12 = int(input12)
        input_7 = int(input7)
        n12 = input_12  #5\n",
        n7 = input_7  #10\n",
        kernel12 = np.ones((n12,n12), np.uint8)
        kernel7 = np.ones((n7,n7), np.uint8)
  
        #图像腐蚀，膨胀运算\n",
        erosion = cv2.erode(bin_img, kernel12)
        result = cv2.dilate(erosion, kernel7)
 
        # 得到去除背景后彩色的图像\n",
        cv2.imwrite(fname+'/Big/3_1_kaibi.jpg',result)
        size = (491, 481)  
        img1 = cv2.resize(result, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/3_1_kaibi.jpg', img1)
        self.label.setPixmap(QPixmap(fname+'/Small/3_1_kaibi.jpg'))
 
    def push_6(self):
        # 第四步：手动去除其他茎\n",
        def nothing(x):
            pass        
        def on_mouse(event, x, y, flags, param): # 鼠标左键按下\n",
            global point1, point2
            img1=cv2.imread(fname+'/Big/3_1_kaibi.jpg')
            img2 = img1.copy()
            if event == cv2.EVENT_LBUTTONDOWN: 
                point1 = (x,y)
                cv2.circle(img2, point1, 10, (0,0,0), -1)
                cv2.imshow('image', img2)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 左键保持按下，且进行拖动 \n",
                cv2.rectangle(img2, point1, (x,y), (0,0,0), -1)
                cv2.imshow('image', img2)
            elif event == cv2.EVENT_LBUTTONUP: # 鼠标左键松开\n",
                point2 = (x,y)
                cv2.rectangle(img2, point1, point2, (0,0,0), -1) 
                cv2.imshow('image', img2)
                img4=cv2.imwrite(fname+'/Big/3_1_kaibi.jpg',img2)
        cv2.namedWindow('d')
        switch1='0:OFF\\n1:ON'    
        cv2.createTrackbar(switch1,'d',0,1,nothing)
 
        while(1):
            k=cv2.waitKey(1)&0xFF
            if k==27:
                break
            f=cv2.getTrackbarPos(switch1,'d')
            # 当t=1时，即已经去除掉了一些点，输出当时的图片\n",
            cv2.setMouseCallback('image', on_mouse)
            img5=cv2.imread(fname+'/Big/3_1_kaibi.jpg')
            img3 = img5.copy()
            cv2.imshow('image', img3)
            if f==1:
                break
        cv2.destroyAllWindows()
        img5=cv2.imread(fname+'/Big/3_1_kaibi.jpg')
        cv2.imwrite(fname+'/Big/4_1_qujing.jpg', img5)

        #显示图像\n",
        size = (491, 481) 
        binary_1 = cv2.resize(img5, size, interpolation=cv2.INTER_AREA) 
        cv2.imwrite(fname+'/Small/4_1_qujing.jpg', binary_1)        
        self.label.setPixmap(QPixmap(fname+'/Small/4_1_qujing.jpg'))
        
        print(point1)
        print(point2)
        
    def push_9(self):
        # 第五步：开闭运算，去除茎周围的点\n",
        input10 = self.horizontalSlider_10.value()
        input9 = self.horizontalSlider_9.value()
        input_10 = int(input10)
        input_9 = int(input9)       
        n10 = input_10  #5\n",
        n9 = input_9  #5\n", 
        kernel10 = np.ones((n10,n10), np.uint8)
        kernel9 = np.ones((n9,n9), np.uint8)
  
        #图像腐蚀，膨胀运算\n",
        binary=cv2.imread(fname+'/Big/4_1_qujing.jpg')
        erosion2 = cv2.erode(binary, kernel10)
        result2 = cv2.dilate(erosion2, kernel9)

        #显示图像\n",
        cv2.imwrite(fname+'/Big/5_1_jieduan.jpg', result2)
 
        size = (491, 481) 
        result2_1 = cv2.resize(result2, size, interpolation=cv2.INTER_AREA) 
        cv2.imwrite(fname+'/Small/5_1_jieduan.jpg', result2_1)  
        self.label.setPixmap(QPixmap(fname+'/Small/5_1_jieduan.jpg')) 
        
   
    def push_12(self):
        global xmin1, ymin1, xmax1, ymax1
        result1 = cv2.imread(fname+'/Big/6_lunkuo.jpg',0)
        result2 = cv2.imread(fname+'/Big/5_1_jieduan.jpg',0)
        result = cv2.bitwise_xor(result1, result2, dst=None, mask=None)
        img = envi.open('I:/N/header.hdr',fname2+'.raw')
        b=img[:,:,103]
        g=img[:,:,103]
        r=img[:,:,103]
        color_img = cv2.merge([b, g, r])
        cv2.imwrite(fname+'/Big/1_te1.jpg', color_img*255)
        original_2 = cv2.imread(fname+'/Big/1_te1.jpg')
        r1,g1,b1 = cv2.split(original_2)
        color_img2 = cv2.merge([b1 & result, g1 & result , r1 & result])
        
        ymin1=900
        ymax1=0
        xmin1=900
        xmax1=0
        for i in range(1,result.shape[0]):
            for j in range(1,result.shape[1]):
                if (result[i,j] > 128).all():
                    if i < ymin1:
                        ymin1 = i          
                    if i > ymax1:
                        ymax1 = i
                    if j < xmin1:
                        xmin1 = j                                           
                    if j > xmax1:
                        xmax1 = j            
        box1 = [[xmin1,ymax1],[xmin1,ymin1],[xmax1,ymin1],[xmax1,ymax1]]
        box2 = np.int0(box1)    
        cut = color_img2 [ymin1:ymax1, xmin1:xmax1]  
        
        cv2.imwrite(fname+'/Big/7_1_jiequ.jpg', cut)     
        cv2.imwrite(fname+'/Result/7_1_jiequ.jpg', cut)   
        self.label.setPixmap(QPixmap(fname+'/Big/7_1_jiequ.jpg')) 
    
    def push_14(self):
        # 局部直方图均衡化
        src = cv2.imread(fname+'/Big/7_1_jiequ.jpg')
        r,g,b = cv2.split(src)
        img = r
        # 灰度图像矩阵的高、宽
        h, w = img.shape
        # 第一步：计算灰度直方图
        grayHist = cv2.calcHist([img],[0],None,[256],[0,256])
        # 第二步：计算累加灰度直方图
        zeroCumuMoment = np.zeros([256], np.uint32)
        for p in range(256):
            if p == 0:
                zeroCumuMoment[p] = grayHist[0]
            else:
                zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
        # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
        outPut_q = np.zeros([256], np.uint8)
        cofficient = 256.0 / (h * w)
        for p in range(256):
            q = cofficient * float(zeroCumuMoment[p]) - 1
            if q >= 0:
                outPut_q[p] = math.floor(q)
            else:
                outPut_q[p] = 0
        # 第四步：得到直方图均衡化后的图像
        equalHistImage = np.zeros(img.shape, np.uint8)
        for i in range(h):
            for j in range(w):
                equalHistImage[i][j] = outPut_q[img[i][j]]

        cv2.imwrite(fname+'/Big/8_equalizeHist.jpg', equalHistImage)      
        self.label.setPixmap(QPixmap(fname+'/Big/8_equalizeHist.jpg'))  

    def push_15(self):
        global labels
        input8 = self.horizontalSlider_8.value()
        input_8 = int(input8)
        input13 = self.horizontalSlider_13.value()
        input_13 = int(input13)
        input14 = self.horizontalSlider_14.value()
        input_14 = int(input14-0.5)
        input15 = self.horizontalSlider_15.value()
        input_15 = int(input15)
        input16 = self.horizontalSlider_16.value()
        input_16 = int(input16)
        if input_16 == 0:
            flags = cv2.KMEANS_RANDOM_CENTERS
        if input_16 == 1:
            flags = cv2.KMEANS_PP_CENTERS            
        
        #读取图片
        img = cv2.imread(fname+'/Big/8_equalizeHist.jpg', cv2.IMREAD_GRAYSCALE)
        # 展平
        img_flat = img.reshape((img.shape[0] * img.shape[1], 1))
        img_flat = np.float32(img_flat)
 
        # 迭代参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, input_13, input_14)
 
        # 进行聚类
        compactness, labels, centers = cv2.kmeans(img_flat, input_8, None, criteria, input_15, flags)
        image = ((labels - np.min(labels)) / (np.max(labels) - np.min(labels)))*255.0
        image = image.astype(np.uint8)
        img_output = image.reshape((img.shape[0], img.shape[1]))
        
        cv2.imwrite(fname+'/Big/9_clustering.jpg', img_output) 
        cv2.imwrite(fname2+'/9_clustering.jpg', img_output) 
        self.label.setPixmap(QPixmap(fname+'/Big/9_clustering.jpg')) 
        
    def push_16(self):
        src = cv2.imread(fname+'/Big/9_clustering.jpg')
        # 转换为浮点数进行计算
        fsrc = np.array(src, dtype=np.float32)
        gray = cv2.cvtColor(fsrc, cv2.COLOR_BGR2GRAY)
 
        # 求取最大值和最小值\n",
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # 得到掩膜\n",
        gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
        (thresh, bin_img) = cv2.threshold(gray_u8, 240, 255, cv2.THRESH_BINARY)

        cv2.imwrite(fname+'/Big/2_2_yuzhi.jpg', bin_img)  
        self.label.setPixmap(QPixmap(fname+'/Big/2_2_yuzhi.jpg'))

    def push_21(self): 
        input21 = self.horizontalSlider_21.value()
        if input21 ==1:
            yanmo1 = cv2.imread(fname+'/Big/6_lunkuo.jpg')
            yanmo2 = cv2.imread(fname+'/Big/5_1_jieduan.jpg')
            yanmo3 = cv2.imread(fname+'/Big/2_2_yuzhi.jpg')
            #将掩膜1与掩膜2与运算

            mask = np.zeros((yanmo1.shape[0], yanmo1.shape[1]))
            for i in range(0,yanmo3.shape[0]):
                for j in range(0,yanmo3.shape[1]):
                    if (yanmo3[i,j] >= 240).all():
                        mask[i+ymin1,j+xmin1+1] = 255
                    if (yanmo3[i,j] < 240).all():
                        mask[i+ymin1,j+xmin1+1] = 0

            cv2.imwrite(fname+'/Big/mask.jpg',mask)
            mask_1 = cv2.imread(fname+'/Big/mask.jpg')

            yanmo2_3 = cv2.bitwise_or(yanmo2, mask_1, dst=None, mask=None)

            mask2 = cv2.bitwise_xor(yanmo1, yanmo2_3, dst=None, mask=None)
            ROI = cv2.bitwise_and(yanmo1, mask2, dst=None, mask=None)

        if input21 ==0:
            result1 = cv2.imread(fname+'/Big/6_lunkuo.jpg',0)
            result2 = cv2.imread(fname+'/Big/5_1_jieduan.jpg',0)
            ROI = cv2.bitwise_xor(result1, result2, dst=None, mask=None)
            
        #保存图片
        cv2.imwrite(fname+'/Big/10_ROI.jpg', ROI)
        size = (491, 481)
        de_2 = cv2.resize(ROI, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/10_ROI.jpg', de_2)  
        self.label.setPixmap(QPixmap(fname+'/Small/10_ROI.jpg')) 

    def push_17(self):
        # 开运算，去散点
        input17 = self.horizontalSlider_17.value()
        input_17 = int(input17)
        input18 = self.horizontalSlider_18.value()
        input_18 = int(input18)
        kernel17 = np.ones((input_17,input_17), np.uint8)
        kernel18 = np.ones((input_18,input_18), np.uint8)        
        
        src = cv2.imread(fname+'/Big/10_ROI.jpg')
        #图像腐蚀，膨胀运算
        erosion2 = cv2.erode(src, kernel17)
        result2 = cv2.dilate(erosion2, kernel18)
        
        #显示图像
        cv2.imwrite(fname+'/Big/3_2_kaibi.jpg',result2)
        size = (491, 481)
        de_2 = cv2.resize(result2, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/3_2_kaibi.jpg', de_2)  
        self.label.setPixmap(QPixmap(fname+'/Small/3_2_kaibi.jpg')) 
   
    def push_19(self): 
        # 第六步：获取轮廓，我们的目的是分块，因此只使用外层轮廓，使用点序列的形式\n",
        global xmin2, ymin2, xmax2, ymax2
        input21 = self.horizontalSlider_21.value()
        if input21 ==1:  
            src = cv2.imread(fname+'/Big/3_2_kaibi.jpg')
            ret,thresh1 = cv2.threshold(src,10,255,cv2.THRESH_BINARY)
            r,g,b = cv2.split(thresh1)
            result = r
            for i in range(1,(thresh1.shape[0]-1),1):
                for j in range(1,(thresh1.shape[1]-1),1):
                    p = 0
                    if (thresh1[i-1,j] == 255).all():
                        p = p + 1
                    if (thresh1[i+1,j] == 255).all():
                        p = p + 1
                    if (thresh1[i,j-1] == 255).all():
                        p = p + 1
                    if (thresh1[i,j+1] == 255).all():
                        p = p + 1
                    if (thresh1[i-1,j-1] == 255).all():
                        p = p + 1
                    if (thresh1[i+1,j+1] == 255).all():
                        p = p + 1
                    if (thresh1[i-1,j+1] == 255).all():
                        p = p + 1
                    if (thresh1[i+1,j-1] == 255).all():
                        p = p + 1          
                    if p >= 5:
                        result[i,j] = 255
                    if p < 5:
                        result[i,j] = 0 
            color_img2 = result
                
        if input21 ==0:
            color_img2 = cv2.imread(fname+'/Big/3_2_kaibi.jpg')
        
        cv2.imwrite(fname+'/Big/11_ROI.jpg', color_img2)
        cv2.imwrite(fname2+'/11_ROI.jpg', color_img2)
        size = (491, 481)
        de_2 = cv2.resize(color_img2, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname+'/Small/11_ROI.jpg', de_2)  
        self.label.setPixmap(QPixmap(fname+'/Small/11_ROI.jpg')) 
        
    def push_8(self):
        result = cv2.imread(fname2+'/11_ROI.jpg', cv2.IMREAD_GRAYSCALE)
        result1 = np.array(result)   
 
        img = envi.open('I:/N/header.hdr',fname2+'.raw')
 
        workbook_2 = xlsxwriter.Workbook(fname2+'/Hyperspectral_data.xlsx') #创建excel簿\n",
        worksheet_2 = workbook_2.add_worksheet('sheet2')

        xload = xmax - xmin
        yload = ymax - ymin
        
        N1 = int(1./3*xload)
        a_set1 = set()
        n1 = 0
        while n1 < N1:
            i1 = random.randint(1, xload)
            a_set1.add(i1)
            n1 += 1
        a_list1 = list(a_set1)
        a_list1.sort()
        
        
        N2 = int(1./3*yload)
        a_set2 = set()
        n2 = 0
        while n2 < N2:
            i2 = random.randint(1, yload)
            a_set2.add(i2)
            n2 += 1
        a_list2 = list(a_set2)
        a_list2.sort()
        
        N1_1 = len(a_list1)
        N2_1 = len(a_list2)
        
        sum1 = 0
        
        for i in range (1, N1_1, 1):
            for j in range (1, N2_1, 1):
                m = xmin + a_list1[i]
                n = ymin + a_list2[j]
                if (result1[n, m] > 128).all():
                    sum1 = sum1 + 1
                    result2 = img[n,m,:]
                    result2 = np.squeeze(result2)
                    for k in range (176):
                        result3 = np.atleast_1d(result2[k]) 
                        worksheet_2.write_column(sum1, k, result3)                
        
        workbook_2.close()  #将excel文件保存关闭，如果没有这一行运行代码会报错 
    
    def push_7(self):    
        readbook = xlrd.open_workbook((fname2+'/Hyperspectral_data.xlsx'))
        sheet = readbook.sheet_by_index(0)

        readbook4 = xlrd.open_workbook((fname+'/x.xlsx'))
        sheet4 = readbook4.sheet_by_index(0)
 
        x = [sheet4.cell_value(i, 0)  for i in range(0, 176)]
        x1 = np.array(x)

        b = [sheet.cell_value(i, j) for i in range(1,sheet.nrows) for j in range(0, 176)]

        b1 = np.array(b)

        b2 = b1.reshape(sheet.nrows-1,176)

        b3 = np.mean(b2, axis=0) # axis=0，计算每一列的均值

        workbook_5 = xlsxwriter.Workbook(fname2+'/Hyperspectral_data1.xlsx') #创建excel簿
        worksheet_5 = workbook_5.add_worksheet('sheet5')
   
        for k in range (176):
            result3 = np.atleast_1d(b3[k])
            
            worksheet_5.write_column(1,k,result3)  
        workbook_5.close()
   
        plt.figure(figsize=(10,6))
        plt.grid()
        plt.plot(x1,b3)
        plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签 
        plt.rcParams['axes.unicode_minus']=False
        font2 = {'size': 15,}
  
        plt.xlabel("波段/nm",font2)
        plt.ylabel("反射率",font2)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(fname2+'/Reflectance.jpg')
        plt.show()
        self.label_17.setPixmap(QPixmap(fname2+'/Reflectance.jpg'))


# In[3]:


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    #实例化主窗口 
    main = ROI()

    #显示
    main.show()
    sys.exit(app.exec_())
    
    #fname,fename 
    #fname2,fename2 = 


# In[ ]:




