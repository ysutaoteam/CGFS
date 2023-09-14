#!/usr/bin/python3
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

a00 = r"Parkinson.csv"
parkinson_x = pd.read_csv(a00)
# columns = ['test_time']
# parkinson_x = parkinson_x.drop(columns, axis=1)
dataset = parkinson_x.values
parkinson_x = dataset.astype('float32')
print(parkinson_x.shape)

for i in range(42):
    a = []
    for j in range(parkinson_x.shape[0]):
        if parkinson_x[j,0]==(i+1):
            a.append(parkinson_x[j,1:])
    name=['MDS',	'meanTapInter',	'medianTapInter',	'iqrTapInter',	'minTapInter',	'maxTapInter',	'skewTapInter',	'kurTapInter',	'sdTapInter',	'madTapInter',
          'cvTapInter',	'rangeTapInter',	'tkeoTapInter',	'dfaTapInter',	'ar1TapInter',	'ar2TapInter',	'fatigue10TapInter',	'fatigue25TapInter',	'fatigue50TapInter',
          'meanDriftLeft',	'medianDriftLeft',	'iqrDriftLeft',	'minDriftLeft',	'maxDriftLeft',	'skewDriftLeft',	'kurDriftLeft',	'sdDriftLeft',	'madDriftLeft',
          'cvDriftLeft',	'rangeDriftLeft',	'meanDriftRight',	'medianDriftRight',	'iqrDriftRight',	'minDriftRight',	'maxDriftRight',	'skewDriftRight',
          'kurDriftRight',	'sdDriftRight',	'madDriftRight',	'cvDriftRight',	'rangeDriftRight',	'numberTaps',	'buttonNoneFreq',	'corXY']
    pp=pd.DataFrame(columns=name,data=a)
    p1 = "p"
    p2 = ".csv"
    c = np.hstack((p1, i+1))
    c = np.hstack((c, p2))
    s = ""
    x = s.join(c)
    pp.to_csv(x)