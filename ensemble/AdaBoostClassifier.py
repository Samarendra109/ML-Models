# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:07:06 2020

@author: Samarendra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import base

def I(flag):
    return 1 if flag else 0

def sign(x):
    return abs(x)/x if x!=0 else 1

def indexToVector(y,k,labelDict):
    y_new = []
    for yi in y:
        i = labelDict[yi]
        v = np.ones(k)*(-1/(k-1))
        v[i] = 1
        y_new.append(v)
    return np.array(y_new)

def indexToLabel(i,clf):
    return clf.classes[i]

class AdaBoostClassifier:
    
    def __init__(self,base_estimator=None,n_estimators=50):
        self.n_estimators = n_estimators
        self.models = [None]*n_estimators
        if base_estimator == None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.base_estimator = base_estimator
        
    def fit(self,X,y):
        
        X = np.float64(X)
        N = len(y)
        w = [1/N for i in range(N)]
        
        self.createLabelDict(np.unique(y))
        k = len(self.classes)
        
        for m in range(self.n_estimators):
            
            Gm = base.clone(self.base_estimator).\
                            fit(X,y,sample_weight=w).predict
            
            errM = sum([w[i]*I(y[i]!=Gm(X[i].reshape(1,-1))) \
                        for i in range(N)])/sum(w)
            
            BetaM = (np.log((1-errM)/errM)+np.log(k-1))
            
            w = [w[i]*np.exp(BetaM*I(y[i]!=Gm(X[i].reshape(1,-1))))\
                     for i in range(N)]
            
            self.models[m] = (BetaM,Gm)
            
    def createLabelDict(self,classes):
        self.labelDict = {}
        self.classes = classes
        for i,cl in enumerate(classes):
            self.labelDict[cl] = i

    def predict(self,X):
        k = len(self.classes)
        y_pred = sum(Bm*indexToVector(Gm(X),k,self.labelDict) \
                             for Bm,Gm in self.models)
        
        iTL = np.vectorize(indexToLabel)
        return iTL(np.argmax(y_pred,axis=1),self)
    