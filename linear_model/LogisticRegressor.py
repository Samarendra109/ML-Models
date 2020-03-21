# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:46:43 2020

@author: Samarendra
"""

import numpy as np
import pandas as pd

class LogisticRegressor:
    
    def __init__(self,max_iter=100):
        self.Beta = None
        self.max_iter = max_iter
        
    def __convert(self,x):
        x = np.float64(x)
        n = len(x)
        x0 = np.ones(n).reshape(-1,1)
        x = np.append(x0,x,axis=1)
        return x
        
    def fit(self,x,y):
        
        try:
            x = self.__convert(x)
            y = np.float64(y).flatten()
            self.Beta = np.zeros(x.shape[1])
            
            for t in range(self.max_iter):
                p = self.__predictUtil(x)
                W = np.diag(p*(1-p))
                xT = x.transpose()
                
                dL_dB = xT@(y-p)
                dL2_dB2 = -xT@W@x
                dL2_dB2_inv = np.linalg.inv(dL2_dB2)
                
                Beta_New = self.Beta - dL2_dB2_inv@dL_dB
                self.Beta = Beta_New
            
        except Exception as e:
            raise e
            
    def __predictUtil(self,x):
        try:
            z = x@(self.Beta)
            return np.exp(z)/(1 + np.exp(z))
        except Exception as e:
            raise e
    
    def predict(self,x):
        
        try:
            x = self.__convert(x)
            z = x@(self.Beta)
            return np.where((np.exp(z)/(1 + np.exp(z)))<0.5,0,1)
        except Exception as e:
            raise e
            

from sklearn.datasets import make_classification
            
features = 4
X,y = make_classification(n_samples=500,n_features=features,n_classes=3,
                              n_informative=features,n_redundant=0)

y = np.where(y<1,'a','b')

X,y = pd.DataFrame(X),pd.DataFrame(y)
classifier = LogisticRegressor()
classifier.fit(X,y)
y_pred = classifier.predict(X)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='lbfgs')
classifier.fit(X,y)
y_pred1 = classifier.predict(X)






