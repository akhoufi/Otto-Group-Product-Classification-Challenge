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
from sklearn.svm import SVC
 

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

    clf = SVC(**params)
    return cross_val_score(clf, X_, y).mean()

space4svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
print best