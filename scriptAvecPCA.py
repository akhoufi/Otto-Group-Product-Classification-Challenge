# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:00:22 2016

@author: YoUNeS
"""

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import numpy as np
import pandas as pd


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_boundary(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    marque = ('o', 'v', '^', '8', 's', 'p', '*', 'd','H', '<', 'h', '>','D')
    couleurs = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'green', 'magenta',
                'yellow', 'darkred', 'chocolate', 'white')
    cmap = ListedColormap(couleurs[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all samples
    #X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=marque[idx], label=cl)
                        
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')



# Lecture des données
df_train = pd.read_csv("input/train.csv")
df_test = pd.read_csv("input/test.csv")



print "Liste des classes dans train data "
print  np.unique(df_train['target'])


## We can here shuffle data before gettin train data 

# creation d'un label encoder pour coder les labels
class_encoder = LabelEncoder()

X_train = df_train.iloc[:,1:-1].values
y_train = class_encoder.fit_transform(df_train["target"].values)


## Pour retrouver les labels on applique l'operation inverse comme ci-dessous : 
## A utiliser si besoin :
#y_inv = class_encoder.inverse_transform(y)


##Pour générer des données de validation 
## A utiliser si besoin :
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)


###Scaling data, comment if you don't want to use it 

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
#X_val_std = sc.fit_transform(X_val)


### Getting the componnent given by PCA decomposition 

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)

#Gettin the percentage of variance explained by each component.
perVarComp = pca.explained_variance_ratio_
cum_perVarComp = np.cumsum(perVarComp)           # cumulative sum of perVarComp

#Ploting these percentages to see their importance

plt.figure()
plt.bar(range(1,len(perVarComp)+1), perVarComp, alpha=0.7, align='center',
         label='percentage of variance of each component')
plt.step(range(1,len(perVarComp)+1), cum_perVarComp, where='mid',
         label='cumulative sum of percentages')
plt.ylabel('explained_variance_ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

### Applying PCA decomposition and SVC

## try a loop over C parameter as in TP4 ?

clf = SVC(C=10, kernel='rbf')
pca = PCA(n_components=2)
clf = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
clf.fit(X_train_pca, y_train)
plt.figure()
plot_boundary(X_train_pca, y_train, classifier=clf)
plt.xlabel('composante principale 1')
plt.ylabel('composante principale 2')
plt.legend(loc='upper left')
plt.show()

## apparement les deux composantes de PCA ne permettent pas de séparer, ça se voitsur le plot  
