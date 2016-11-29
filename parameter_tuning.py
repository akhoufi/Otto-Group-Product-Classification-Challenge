# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:42:20 2016

@author: akhou
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale, LabelEncoder, normalize
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier 
from hyperopt import fmin, tpe, hp,STATUS_OK, Trials

train = pd.read_csv('input/train.csv')
sample = pd.read_csv('input/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
y = LabelEncoder().fit_transform(labels)
train = train.drop('id', axis=1)
X = train.drop('target', axis=1)

def hyperopt_train_test(params):
    if 'normalize' in params:
        if params['normalize'] == 1:
            X = normalize(X)
        del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X = scale(X)
        del params['scale']
        
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'randomforest':
        clf = RandomForestClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X, y,scoring="log_loss").mean()

space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0)
    },
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,50))
    },

])

count = 0
best = 0
def f(params):
    global best, count
    count += 1
    acc = hyperopt_train_test(params.copy())
    if acc > best:
        print 'new best:', acc, 'using', params['type']
        best = acc
    if count % 50 == 0:
        print 'iters:', count, ', acc:', acc, 'using', params
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=1500, trials=trials)
print 'best:'
print best