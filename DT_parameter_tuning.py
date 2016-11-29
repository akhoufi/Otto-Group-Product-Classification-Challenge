# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:38:42 2016

@author: akhou
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale, LabelEncoder, normalize
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
 

from hyperopt import fmin, tpe, hp,STATUS_OK, Trials

train = pd.read_csv('input/train.csv')
sample = pd.read_csv('input/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
y = LabelEncoder().fit_transform(labels)
train = train.drop('id', axis=1)
X = train.drop('target', axis=1)

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
        del params['scale']
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X, y, cv=10).mean()

space4dt = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)
print 'best:'
print best