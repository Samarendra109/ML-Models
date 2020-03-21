# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:20:05 2020

@author: Samarendra
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:50:01 2020

@author: Samarendra
"""

import numpy as np
import pandas as pd

from tree import DecisionTree,SplitMethod

class MeanSquaredError(SplitMethod):
    
    def initialScore(self,y):
        y1 = np.array(y.values)
        
        self.y1_sqr = sum(np.square(y1))
        self.y1_sum = sum(y1)
        self.y1_num = len(y1)
        print(self.y1_num)
        
        self.y0_sqr = 0
        self.y0_sum = 0
        self.y0_num = 0
        
        return self.y1_sqr - ((self.y1_sum)**2)/self.y1_num
    
    def update(self,yi):
        
        self.y0_num += 1
        self.y0_sqr += yi**2
        self.y0_sum += yi
        
        y0Var = self.y0_sqr - ((self.y0_sum)**2)/self.y0_num
        
        self.y1_num -= 1
        self.y1_sqr -= yi**2
        self.y1_sum -= yi
        
        
        if self.y1_num != 0:
            y1Var = self.y1_sqr - ((self.y1_sum)**2)/self.y1_num
        else:
            y1Var = 0
        
        score = y0Var + y1Var
        
        return score
    
    def getNodeVal(self,y):
        return y.mean()

class DecisionTreeRegressor(DecisionTree):
    
    def __init__(self,criterion='mse',max_depth=np.inf,min_samples_split=2):
        super().__init__(max_depth,min_samples_split)
        self.criterion = criterion
        self.__mse = MeanSquaredError()
        self.method = {'mse':self.__mse}
        
    def fit(self,X,y):
        columns = X.columns
        methodObj = self.method[self.criterion]
        y.columns = ['y']
        df = pd.concat([X,y],axis=1)
        
        self.root.fitUtil(df,columns,methodObj,
                        0,self.max_depth,self.min_samples_split)
        

