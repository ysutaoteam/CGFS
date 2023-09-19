#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
from sklearn.neural_network import BernoulliRBM

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#忽略警告
warnings.filterwarnings("ignore")
# 使用自带的样式进行美化
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import math
from sklearn.model_selection import train_test_split
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
from sklearn import datasets,ensemble

def train(trainx,testx,trainy,testy):
    from sklearn.preprocessing import  MinMaxScaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(trainx)
    x_test = scaler.transform(testx)
    scale1 = MinMaxScaler()
    y_train = scale1.fit_transform(trainy)
    y_test = scale1.transform(testy)

    from sklearn import datasets, ensemble
    # import xgboost as xgb
    rf_model = ensemble.RandomForestRegressor(n_estimators=30, max_depth=30, criterion='mse')
    rf_model.fit(x_train, y_train)
    pre = rf_model.predict(x_test)
    dataset_pred = pre.reshape((-1, 1))
    dataset_pred = np.array(dataset_pred).reshape((-1, 1))
    dataset_pred = scale1.inverse_transform(dataset_pred)
    y_test = scale1.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test,dataset_pred))
    mae=mean_absolute_error(y_test,dataset_pred)
    return mae,rmse


def ParkinsonLoader(path1):
    parkinson_x = pd.read_csv(path1)
    columns = ['Unnamed: 0']
    parkinson_x = parkinson_x.drop(columns, axis=1)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_x, columns=columns)
    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')
    xx_train, x_test, yy_train, y_test = train_test_split(parkinson_x, parkinson_y, test_size=0.2, shuffle=True,random_state=86)
    x_train, x_val, y_train, y_val = train_test_split(xx_train, yy_train, test_size=0.25, shuffle=True,random_state=83)
    return x_train,x_val,  x_test, y_train,y_val, y_test

def res(x1):
    parkinson_x = pd.read_csv(x1)
    columns = ['Unnamed: 0']
    parkinson_x = parkinson_x.drop(columns, axis=1)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_x, columns=columns)
    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')
    return parkinson_x, parkinson_y

for i in range(42):

    mae1 = []
    rmse1 = []
    p1 = "p"
    p2 = ".csv"
    c = np.hstack((p1, i+1))
    c = np.hstack((c, p2))
    s = ""
    x = s.join(c)
    a = list(range(42))
    b = [i]
    a.remove(i)

    for j in a:
        tx_train, tx_val, tx_test, ty_train, ty_val, ty_test = ParkinsonLoader(x)
        p11 = "p"
        p22 = ".csv"
        c1 = np.hstack((p11, j + 1))
        c1 = np.hstack((c1, p22))
        s1 = ""
        x1 = s1.join(c1)
        trainx, trainy = res(x1)
        xxx = np.vstack((tx_train, trainx))
        yyy = np.vstack((ty_train, trainy))
        mae, _ = train(xxx, tx_val, yyy, ty_val)
        print(mae)
        mae1.append((mae))

    mae1=np.array(mae1).reshape((-1,1))
    a = [j for i in mae1 for j in i]
    sort1 = np.argsort(a) + 1

    sort = []
    for k in sort1:

        if k >= (i+1):
            sort.append(k + 1)
        else:
            sort.append(k)

    print("mae1:",mae1)
    print("sort:", sort)

    p1 = "motor-mae"
    p2 = ".csv"
    c = np.hstack((p1, i+1))
    c = np.hstack((c, p2))
    s = ""
    x = s.join(c)

    mae11=pd.DataFrame(sort)
    mae11.to_csv(x)







