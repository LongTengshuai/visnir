#!/usr/bin/env python
# coding: utf-8

# In[6]:


#-*- coding: utf-8 -*-
import ast
import pandas as pd
import xlsxwriter
import xlrd
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[7]:


class preprocess():
    
    def Input_path(self, x_path, classification_path, samples_num, spectrum_num):
        # 1、导入excel文件
        # 输入：
        # x_path 和 classification_path 分别是1-176个波段的值和样本数据，最后一列是类别
        # samples_num 和 spectrum_num 分别是样本数和光谱数
        # 输出：
        # xx 用来画画的
        # Y2 光谱矩阵
    
        readbook_1 = xlrd.open_workbook(x_path)
        readbook_2 = xlrd.open_workbook(classification_path)

        sheet_1 = readbook_1.sheet_by_index(0)
        sheet_2 = readbook_2.sheet_by_index(0)

        x = [sheet_1.cell_value(i, j) for i in range(0, sheet_1.nrows) for j in range(0, spectrum_num-1)]
        x1 = np.array(x)  
        x2 = x1.reshape(sheet_1.nrows,spectrum_num - 1)
        xx = x2[2, 1:spectrum_num]

        Y = [sheet_2.cell_value(i, j)   for i in range(0,samples_num) for j in range(0, spectrum_num)]
        Y1 = np.array(Y) 
        Y2 = Y1.reshape(samples_num, spectrum_num)

        return xx,Y2
    
    # 1、MSC（多元散射矫正处理）
    def msc(self,sdata):
    
        n = sdata.shape[0]  # 样本数量
    
        k = np.zeros(sdata.shape[0])
        b = np.zeros(sdata.shape[0])

        M = np.mean(sdata, axis=0)
 
        for i in range(n):
            y = sdata[i, :]
            y = y.reshape(-1, 1)
            M = M.reshape(-1, 1)
            model = LinearRegression()
            model.fit(M, y)
            k[i] = model.coef_
            b[i] = model.intercept_
 
        spec_msc = np.zeros_like(sdata)
    
        for i in range(n):
            bb = np.repeat(b[i], sdata.shape[1])
            kk = np.repeat(k[i], sdata.shape[1])
            temp = (sdata[i, :] - bb)/kk
            spec_msc[i, :] = temp

        return spec_msc
    
    
    # 2、SNV（标准正态变量变换）
    def snv(self,sdata):
        """
        标准正态变量变换
        """
        temp1 = np.mean(sdata, axis=1)
        temp2 = np.tile(temp1, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))
        temp3 = np.std(sdata, axis=1)
        temp4 = np.tile(temp3, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))
        return (sdata - temp2)/temp4
    
    
    # 3、D1（一阶差分）
    def D1(self,sdata):
        """
            一阶差分
        """
        temp1 = pd.DataFrame(sdata)
        temp2 = temp1.diff(axis=1)
        temp3 = temp2.values
        return np.delete(temp3, 0, axis=1) 
    
    # 4、D2（二阶差分）
    def D2(self,sdata):
        """
        二阶差分
        """
        temp2 = (pd.DataFrame(sdata)).diff(axis=1)  
        temp3 = np.delete(temp2.values, 0, axis=1)  
        temp4 = (pd.DataFrame(temp3)).diff(axis=1)  
        spec_D2 = np.delete(temp4.values, 0, axis=1)
        return spec_D2
     
    def Noise_band_removal(self,related_coefficient_path, spectrum_num):
        # 1.1、确定想要的敏感波段，去除前后的噪声
        # 用的是相关系数（这部分用的matlab做的，将raw文件转化成一张张图片，然后计算每张图片的相关系数）
        # 输入：
        # related_coefficient_path 是相关系数矩阵的存储路径
        # spectrum_num 是光谱数
        # 输出：
        # related_coefficient_result 敏感波段矩阵

        # my_font = font_manager.FontProperties(fname=r"C:/Windows/Fonts/simsun.ttc", size=12)

        readbook_3 = xlrd.open_workbook(related_coefficient_path)
        sheet_3 = readbook_3.sheet_by_index(0)
        c = [sheet_3.cell_value(i, j) for i in range(1, sheet_3.nrows) for j in range(0, spectrum_num-2)]
        c1 = np.array(c) 
        c2 = c1.reshape(sheet_3.nrows-1,spectrum_num-2)
        c3 = c2[:,0:spectrum_num-2]
        related_coefficient_result = np.transpose(c3)
    
        return related_coefficient_result    
    
    def PlotSpectrum(self, spec, n_features):
        """
        :param spec: shape (n_samples, n_features)
        :return: plt
        """

        plt.figure(figsize=(5.2, 3.1), dpi=300)
        x = np.arange(0, n_features)
        for i in range(spec.shape[0]):
            plt.plot(x, spec[i, :], linewidth=0.6)

        fonts = 8
        #plt.xlim(350, 2550)
        # plt.ylim(0, 1)
        plt.xlabel('Wavelength (nm)', fontsize=fonts)
        plt.ylabel('absorbance (AU)', fontsize=fonts)
        plt.yticks(fontsize=fonts)
        plt.xticks(fontsize=fonts)
        plt.tight_layout(pad=0.3)
        plt.grid(True)
        return plt

