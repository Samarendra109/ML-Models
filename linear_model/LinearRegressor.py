"""

@author: Samarendra
"""

import numpy as np

class LinearRegressor:
    
    def __init__(self,fit_intercept=True):
        
        '''
        Basic Linear Regression without any regularizations (OLS)
        
        Parameters:
        ----------------------------------------------
        fit_intercept: Adds an intercept term to the linear equation.
        IF set to false then the line will pass through origin. ie.
        yj = 0 if xji = 0 for i from 0 tp p-1
        '''
        
        self.Beta = None
        self.fit_intercept = fit_intercept
        
    def __convert(self,x):
        
        '''
        Appends one extra column containing all 1s to the X matrix
        (Ignores this step is fit_intercept is set to false) 
        '''
        
        x = np.float64(x)
        if self.fit_intercept:
            n = len(x)
            x0 = np.ones(n).reshape(-1,1)
            x = np.append(x0,x,axis=1)
        return x
        
    def fit(self,x,y):
        
        '''
        Fits the model to the training data.
        Beta = (X^T * X)^-1 * X^T * y
        '''        
        
        try:
            x = self.__convert(x)
            y = np.float64(y).flatten()
            
            xT = x.transpose()
            xT_x_inv = np.linalg.inv(xT@x)
            self.Beta = xT_x_inv@xT@y
            
        except Exception as e:
            raise e
    
    def predict(self,x):
        
        '''
        Evaluates y_pred using the formula
        y_pred = X*Beta
        '''
        
        try:
            x = self.__convert(x)
            return x@(self.Beta)
        except Exception as e:
            raise e
