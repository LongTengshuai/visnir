#!/usr/bin/env python
# coding: utf-8

# In[1]:


#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.linalg import qr, inv, pinv
import scipy.stats
import scipy.io as scio
from progress.bar import Bar

from sklearn.neural_network import MLPRegressor
from genetic_selection import GeneticSelectionCV

import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import copy
import warnings
warnings.filterwarnings('ignore')


# In[1]:


# 1、SPA(连续投影法)

class SPA:

    def _projections_qr(self, X, k, M):
        '''
        原版连续投影算法使用MATLAB内置的QR函数
        该版本改用scipy.linalg.qr函数
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.qr.html
        X : 预测变量矩阵
        K ：投影操作的初始列的索引
        M : 结果包含的变量个数

        return ：由投影操作生成的变量集的索引
        '''

        X_projected = X.copy()

        # 计算列向量的平方和
        norms = np.sum((X ** 2), axis=0)
        # 找到norms中数值最大列的平方和
        norm_max = np.amax(norms)

        # 缩放第K列 使其成为“最大的”列
        X_projected[:, k] = X_projected[:, k] * 2 * norm_max / norms[k]

        # 矩阵分割 ，order 为列交换索引
        _, __, order = qr(X_projected, 0, pivoting=True)

        return order[:M].T

    def _validation(self, Xcal, ycal, var_sel, Xval=None, yval=None):
        '''
        [yhat,e] = validation(Xcal,var_sel,ycal,Xval,yval) -->  使用单独的验证集进行验证
        [yhat,e] = validation(Xcal,ycalvar_sel) --> 交叉验证
        '''
        N = Xcal.shape[0]  # N 测试集的个数
        if Xval is None:  # 判断是否使用验证集
            NV = 0
        else:
            NV = Xval.shape[0]  # NV 验证集的个数

        yhat = e = None

        # 使用单独的验证集进行验证
        if NV > 0:
            Xcal_ones = np.hstack(
                [np.ones((N, 1)), Xcal[:, var_sel].reshape(N, -1)])

            # 对偏移量进行多元线性回归
            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            # 对验证集进行预测
            np_ones = np.ones((NV, 1))
            Xval_ = Xval[:, var_sel]
            X = np.hstack([np.ones((NV, 1)), Xval[:, var_sel]])
            yhat = X.dot(b)
            # 计算误差
            e = yval - yhat
        else:
            # 为yhat 设置适当大小
            yhat = np.zeros((N, 1))
            for i in range(N):
                # 从测试集中 去除掉第 i 项
                cal = np.hstack([np.arange(i), np.arange(i + 1, N)])
                X = Xcal[cal, var_sel.astype(np.int)]
                y = ycal[cal]
                xtest = Xcal[i, var_sel]
                # ytest = ycal[i]
                X_ones = np.hstack([np.ones((N - 1, 1)), X.reshape(N - 1, -1)])
                # 对偏移量进行多元线性回归
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                # 对验证集进行预测
                yhat[i] = np.hstack([np.ones(1), xtest]).dot(b)
            # 计算误差
            e = ycal - yhat

        return yhat, e

    def spa(self, Xcal, ycal, m_min=1, m_max=None, Xval=None, yval=None, autoscaling=1):
        '''
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,Xval,yval,autoscaling) --> 使用单独的验证集进行验证
        [var_sel,var_sel_phase2] = spa(Xcal,ycal,m_min,m_max,autoscaling) --> 交叉验证

        如果 m_min 为空时， 默认 m_min = 1
        如果 m_max 为空时：
            1. 当使用单独的验证集进行验证时， m_max = min(N-1, K)
            2. 当使用交叉验证时，m_max = min(N-2, K)

        autoscaling : 是否使用自动刻度 yes = 1，no = 0, 默认为 1

        '''

        assert (autoscaling == 0 or autoscaling == 1), "请选择是否使用自动计算"

        N, K = Xcal.shape

        if m_max is None:
            if Xval is None:
                m_max = min(N - 1, K)
            else:
                m_max = min(N - 2, K)

        assert (m_max < min(N - 1, K)), "m_max 参数异常"

        # 第一步： 对测试集进行投影操作

        # 

        normalization_factor = None
        if autoscaling == 1:
            normalization_factor = np.std(
                Xcal, ddof=1, axis=0).reshape(1, -1)[0]
        else:
            normalization_factor = np.ones((1, K))[0]

        Xcaln = np.empty((N, K))
        for k in range(K):
            x = Xcal[:, k]
            Xcaln[:, k] = (x - np.mean(x)) / normalization_factor[k]

        SEL = np.zeros((m_max, K))

        # 进度条
        #with Bar('Projections :', max=K) as bar:
        for k in range(K):
            SEL[:, k] = self._projections_qr(Xcaln, k, m_max)
        #        bar.next()

        # 第二步： 进行评估

        PRESS = float('inf') * np.ones((m_max + 1, K))

        #with Bar('Evaluation of variable subsets :', max=(K) * (m_max - m_min + 1)) as bar:
        for k in range(K):
            for m in range(m_min, m_max + 1):
                var_sel = SEL[:m, k].astype(np.int)
                _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)
                PRESS[m, k] = np.conj(e).T.dot(e)
        
        #            bar.next()

        PRESSmin = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESSmin)

        # 第 k_sel波段为初始波段时最佳，波段数目为 m_sel（k_sel）
        var_sel_phase2 = SEL[:m_sel[k_sel], k_sel].astype(np.int)

        # 最后消去变量

        # 第 3.1 步 计算相关指数
        Xcal2 = np.hstack([np.ones((N, 1)), Xcal[:, var_sel_phase2]])
        b = np.linalg.lstsq(Xcal2, ycal, rcond=None)[0]
        std_deviation = np.std(Xcal2, ddof=1, axis=0)

        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]

        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        PRESS_scree = np.empty(len(var_sel_phase2))
        yhat = e = None
        for i in range(len(var_sel_phase2)):
            var_sel = var_sel_phase2[index_decreasing_relev[:i + 1]]
            _, e = self._validation(Xcal, ycal, var_sel, Xval, yval)

            PRESS_scree[i] = np.conj(e).T.dot(e)

        RMSEP_scree = np.sqrt(PRESS_scree / len(e))

        # 第 3.3： F-test 验证
        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.25
        dof = len(e)
        fcrit = scipy.stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit

        # 找到不明显比 PRESS_scree_min 大的最小变量

        i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))
        i_crit = max(m_min, i_crit)

        var_sel = var_sel_phase2[index_decreasing_relev[:i_crit]]

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        fig1 = plt.figure()
        plt.xlabel('Number of variables included in the model')
        plt.ylabel('RMSE')
        # plt.title('Final number of selected variables:{}(RMSE={})'.format(len(var_sel), RMSEP_scree[i_crit]))
        plt.plot(RMSEP_scree)
        # plt.scatter(i_crit, RMSEP_scree[i_crit], marker='s', color='r')
        plt.grid(True)

        fig2 = plt.figure()
        plt.plot(Xcal[0, :])
        plt.scatter(var_sel, Xcal[0, var_sel], marker='s', color='r')
        plt.legend(['First calibration object', 'Selected variables'])
        plt.xlabel('Variable index')
        plt.grid(True)
        plt.show()

        return var_sel, var_sel_phase2, RMSEP_scree,Xcal[0, :]

    def __repr__(self):
        return "SPA()"


# In[3]:


# 2、GA(遗传算法)

class GA_algorithm():

    def GA(self, x_train, y_train, x_test, y_test, base, size):

        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
 
        # 2. 优化超参数
        wavelengths_save, wavelengths_size, r2_test_save, mse_test_save = [], [], [], []
        for hidden_size in range(base, base+size):
            print('隐含层神经元数量: ', hidden_size)
            estimator = MLPRegressor(hidden_layer_sizes=hidden_size,
                                    activation='relu',
                                    solver='adam',
                                    alpha=0.0001,
                                    batch_size='auto',
                                    learning_rate='constant',
                                    learning_rate_init=0.001,
                                    power_t=0.5,
                                    max_iter=1000,
                                    shuffle=True,
                                    random_state=1,
                                    tol=0.0001,
                                    verbose=False,
                                    warm_start=False,
                                    momentum=0.9,
                                    nesterovs_momentum=True,
                                    early_stopping=False,
                                    validation_fraction=0.1,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=1e-08)
 
            selector = GeneticSelectionCV(estimator, 
                                    #所使用的分类器
                                    cv=10,  
                                    #交叉验证
                                    verbose=1,
                                    scoring="neg_mean_squared_error", 
                                    #指标，用于衡量每一条染色体（超参数组合）的适应度；
                                    max_features=50,
                                    n_population=200, 
                                    #保持整个种群的染色体数目为50个超参数组合
                                    crossover_proba=0.5,
                                    #表示我们会选择每一条“父母”染色体（超参数组合）中的50%的基因（超参数）进行相互交叉，
                                    #具体的做法就是是否交换某个超参数设置为bool，然后随机取0，1即可；
                                    #（我们所谓的轮盘赌算法是应用在父母的选择上，和这里的交叉概率无关）
                                    mutation_proba=0.2,
                                    #意味着我们的“孩子”超参数组合中每次会大概选择出10%的超参数进行随机取值，
                                    #这个随机取值不是乱取值而是根据我们之前定义的超参数搜索空间进行随机搜索；
                                    #（注意，变异具体的做法是取0~1之间均匀随机采样数然后和我们设定的概率进行比较，
                                    #这里用的就是常规的采样而不是轮盘赌）                                    
                                    n_generations=200,
                                    crossover_independent_proba=0.5,
                                    mutation_independent_proba=0.05,
                                    tournament_size=3, 
                                    #意味着我们每次从上一代中选择出适应度最好的3个超参数组合直接进行 “复制”，
                                    #不进行变异和交叉进入下一代的繁衍中，在50个超参数组合中我们进行交叉和变异生成47组新的超参数，
                                    #这也意味着上一代排名靠后的47个超参数组合被淘汰；
                                    n_gen_no_change=10,
                                    caching=True,
                                    n_jobs=-1)
            selector = selector.fit(x_train, y_train)
            print('有效变量的数量：', selector.n_features_)
            #print(np.array(selector.population_).shape)
            print(selector.generation_scores_)
 
            x_train_s, x_test_s = x_train[:, selector.support_], x_test[:, selector.support_]
            estimator.fit(x_train_s, y_train.ravel())
 
            # y_train_pred = estimator.predict(x_train_s)
            y_test_pred = estimator.predict(x_test_s)
            # y_train_pred = y_scale.inverse_transform(y_train_pred)
            # y_test_pred = y_scale.inverse_transform(y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
 
            wavelengths_save.append(list(selector.support_))  
            wavelengths_size.append(selector.n_features_) 
            r2_test_save.append(r2_test)
            mse_test_save.append(mse_test)
            print('决定系数：', r2_test, '均方误差：', mse_test)
            print('有效变量数量', wavelengths_size)
 
            # 3.保存过程数据
            dict_name = {'wavelengths_size': wavelengths_size, 'r2_test_save': r2_test_save,
                'mse_test_save': mse_test_save, 'wavelengths_save': wavelengths_save}
            f = open('bpnn_ga.txt', 'w')
            f.write(str(dict_name))
            f.close()
    
        return r2_test_save, mse_test_save, wavelengths_save


# In[4]:


# 3、CARS(竞争性自适应重加权)

class CARS:
    
    def PC_Cross_Validation(self, X, y, pc, cv):
        '''
            x :光谱矩阵 nxm
            y :浓度阵 （化学值）
            pc:最大主成分数
            cv:交叉验证数量
        return :
            RMSECV:各主成分数对应的RMSECV
            PRESS :各主成分数对应的PRESS
            rindex:最佳主成分数
        '''
        kf = KFold(n_splits=cv)
        RMSECV = []
        for i in range(pc):
            RMSE = []
            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                pls = PLSRegression(n_components=i + 1)
                pls.fit(x_train, y_train)
                y_predict = pls.predict(x_test)
                RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
            RMSE_mean = np.mean(RMSE)
            RMSECV.append(RMSE_mean)
        rindex = np.argmin(RMSECV)
        return RMSECV, rindex

    def Cross_Validation(self, X, y, pc, cv):
        '''
         x :光谱矩阵 nxm
         y :浓度阵 （化学值）
         pc:最大主成分数
         cv:交叉验证数量
         return :
                RMSECV:各主成分数对应的RMSECV
        '''
        kf = KFold(n_splits=cv)
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=pc)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        return RMSE_mean
    
    def CARS_Cloud(self, X, y, N=50, f=20, cv=10):
        p = 0.8
        m, n = X.shape
        u = np.power((n/2), (1/(N-1)))
        k = (1/(N-1)) * np.log(n/2)
        cal_num = np.round(m * p)
        # val_num = m - cal_num
        b2 = np.arange(n)
        x = copy.deepcopy(X)
        D = np.vstack((np.array(b2).reshape(1, -1), X))
        WaveData = []
        # Coeff = []
        WaveNum =[]
        RMSECV = []
        r = []
        for i in range(1, N+1):
            r.append(u*np.exp(-1*k*i))
            wave_num = int(np.round(r[i-1]*n))
            WaveNum = np.hstack((WaveNum, wave_num))
            cal_index = np.random.choice                    (np.arange(m), size=int(cal_num), replace=False)
            wave_index = b2[:wave_num].reshape(1, -1)[0]
            xcal = x[np.ix_(list(cal_index), list(wave_index))]
            #xcal = xcal[:,wave_index].reshape(-1,wave_num)
            ycal = y[cal_index]
            x = x[:, wave_index]
            D = D[:, wave_index]
            d = D[0, :].reshape(1,-1)
            wnum = n - wave_num
            if wnum > 0:
                d = np.hstack((d, np.full((1, wnum), -1)))
            if len(WaveData) == 0:
                WaveData = d
            else:
                WaveData  = np.vstack((WaveData, d.reshape(1, -1)))

            if wave_num < f:
                f = wave_num

            pls = PLSRegression(n_components=f)
            pls.fit(xcal, ycal)
            beta = pls.coef_
            b = np.abs(beta)
            b2 = np.argsort(-b, axis=0)
            coef = copy.deepcopy(beta)
            coeff = coef[b2, :].reshape(len(b2), -1)
            # cb = coeff[:wave_num]
            #
            # if wnum > 0:
            #     cb = np.vstack((cb, np.full((wnum, 1), -1)))
            # if len(Coeff) == 0:
            #     Coeff = copy.deepcopy(cb)
            # else:
            #     Coeff = np.hstack((Coeff, cb))
            rmsecv, rindex = self.PC_Cross_Validation(xcal, ycal, f, cv)
            RMSECV.append(self.Cross_Validation(xcal, ycal, rindex+1, cv))
        # CoeffData = Coeff.T

        WAVE = []
        # COEFF = []

        for i in range(WaveData.shape[0]):
            wd = WaveData[i, :]
            # cd = CoeffData[i, :]
            WD = np.ones((len(wd)))
            # CO = np.ones((len(wd)))
            for j in range(len(wd)):
                ind = np.where(wd == j)
                if len(ind[0]) == 0:
                    WD[j] = 0
                    # CO[j] = 0
                else:
                    WD[j] = wd[ind[0]]
                    # CO[j] = cd[ind[0]]
            if len(WAVE) == 0:
                WAVE = copy.deepcopy(WD)
            else:
                WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
            # if len(COEFF) == 0:
            #     COEFF = copy.deepcopy(CO)
            # else:
            #     COEFF = np.vstack((WAVE, CO.reshape(1, -1)))

        MinIndex = np.argmin(RMSECV)
        Optimal = WAVE[MinIndex, :]
        boindex = np.where(Optimal != 0)
        OptWave = boindex[0]

        fig = plt.figure(figsize=(8,15))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fonts = 16
        plt.subplot(311)
        plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
        plt.ylabel('被选择的波长数量', fontsize=fonts)
        plt.title('最佳迭代次数：' + str(MinIndex) + '次', fontsize=fonts)
        plt.plot(np.arange(N), WaveNum)

        plt.subplot(312)
        plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
        plt.ylabel('RMSECV', fontsize=fonts)
        plt.plot(np.arange(N), RMSECV)

        plt.subplot(313)
        plt.plot(X[0, :])
        plt.scatter(OptWave, X[0, OptWave], marker='s', color='r')
        plt.legend(['First calibration object', 'Selected variables'])
        plt.xlabel('Variable index')
        plt.grid(True)
        plt.show()
        
        return OptWave

