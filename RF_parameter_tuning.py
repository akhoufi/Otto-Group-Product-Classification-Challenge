# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:56:44 2016

@author: akhou
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale, LabelEncoder, normalize
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
 

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
        
    clf = RandomForestClassifier(**params)

    return cross_val_score(clf, X, y,cv=10).mean()

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,1000)),
    'max_features': hp.choice('max_features', range(0,93)),
    'n_estimators': hp.choice('n_estimators', range(10,1500)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

best = 0
def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc > best:
        best = acc
        print 'new best:', best, params
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)
print 'best:'
print best