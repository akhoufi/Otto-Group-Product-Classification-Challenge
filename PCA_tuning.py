# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:21:01 2016

@author: akhou
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA


# import data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
sample = pd.read_csv('input/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
labels = preprocessing.LabelEncoder().fit_transform(labels)
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)



train['sum']=train.sum(axis=1, numeric_only=True)
test['sum']=test.sum(axis=1, numeric_only=True)

train['var']=train.var(axis=1, numeric_only=True)
test['var']=test.var(axis=1, numeric_only=True)

train['filled']=train.astype(bool).sum(axis=1)
test['filled']=test.astype(bool).sum(axis=1)

train= scale(train, with_std = False)
test=scale(test, with_std = False)

pca = PCA(n_components=96)

pca.fit(train)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)

