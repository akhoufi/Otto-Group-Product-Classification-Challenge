# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:36:49 2016

@author: akhou
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer


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

#Feature engineering
#train= train_original.copy()
#test = test_original.copy()
#
#tfidf =TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
#train =tfidf.fit_transform(train_original.values)
#test  =tfidf.fit_transform(test_original.values)

train['sum']=train.sum(axis=1, numeric_only=True)
test['sum']=test.sum(axis=1, numeric_only=True)

train['var']=train.var(axis=1, numeric_only=True)
test['var']=test.var(axis=1, numeric_only=True)

train['filled']=train.astype(bool).sum(axis=1)
test['filled']=test.astype(bool).sum(axis=1)

#train= scale(train, with_std = False)
#test=scale(test, with_std = False)

pca = PCA(n_components=20)
pca.fit(train)
train = pca.fit_transform(train)

test=pca.fit_transform(test)

# train a random forest classifier without calibration

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(train, labels)
preds = clf.predict_proba(test)
calibrated_clf = CalibratedClassifierCV(clf, method='isotonic')
calibrated_clf.fit(train, labels)
preds = calibrated_clf.predict_proba(test)
#print cross_val_score(clf, train, labels,scoring='log_loss', cv=10).mean()

# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('output/RF_PCA_TF_IDF_sample.csv', index_label='id')

