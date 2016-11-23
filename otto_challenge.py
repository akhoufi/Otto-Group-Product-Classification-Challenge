# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 19:36:49 2016

@author: akhou
"""

import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

TARGET_LABEL='target'
ID_LABEL='id'
range_of_classes = range(1, 10)

fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set

ID_col = [ID_LABEL]
target_col = [TARGET_LABEL]
features= list(set(list(fullData.columns))-set(ID_col)-set(target_col))
other_col=['Type'] #Test and Train Data set identifier

fullData.isnull().any() #Will return the feature with True or False,True means have missing value else False




train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

x_train = Train[list(features)].values
y_train = Train[TARGET_LABEL].values
x_validate = Validate[list(features)].values
y_validate = Validate[TARGET_LABEL].values
x_test=test[list(features)].values

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

#status = rf.predict_proba(x_validate)
#fpr, tpr, _ = roc_curve(y_validate, status[:,1])
#roc_auc = auc(fpr, tpr)
#print roc_auc

final_status = rf.predict_proba(x_test)

submission = pd.DataFrame({ "id": test["id"]})

i = 0

# Create column name based on target values(see sample_submission.csv)
for num in range_of_classes:
    col_name = str("Class_{}".format(num))
    submission[col_name] = final_status[:,i]
    i = i + 1
    
submission.to_csv('otto.csv', index=False)

