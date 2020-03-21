# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 21:44:33 2020

@author: Samarendra
"""

import numpy as np

class LinearRegressor:
    
    def __init__(self,fit_intercept=True):
        self.Beta = None
        self.fit_intercept = fit_intercept
        
    def __convert(self,x):
        x = np.float64(x)
        if self.fit_intercept:
            n = len(x)
            x0 = np.ones(n).reshape(-1,1)
            x = np.append(x0,x,axis=1)
        return x
        
    def fit(self,x,y):
        
        try:
            x = self.__convert(x)
            y = np.float64(y).flatten()
            
            xT = x.transpose()
            xT_x_inv = np.linalg.inv(xT@x)
            self.Beta = xT_x_inv@xT@y
            
        except Exception as e:
            raise e
    
    def predict(self,x):
        
        try:
            x = self.__convert(x)
            return x@(self.Beta)
        except Exception as e:
            raise e