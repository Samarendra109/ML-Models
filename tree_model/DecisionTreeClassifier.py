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

from scipy import stats
from collections import defaultdict

from tree import DecisionTree,SplitMethod

class GiniIndex(SplitMethod):
    
    def initialScore(self,y):
        
        self.count0 = defaultdict(int)
        self.count1 = defaultdict(int)
        
        for yi in y.values:
            self.count1[yi] += 1
            
        self.n0 = 0
        self.n1 = len(y)
        self.sum_sqr0 = 0
        self.sum_sqr1 = 0
        
        score = 1
        for yk in self.count1:
            score -= (self.count1[yk]/self.n1)**2 
            self.sum_sqr1 += self.count1[yk]**2
        return score
    
    def update(self,yi):
        
        self.n0+=1
        self.n1-=1
        
        self.sum_sqr0 -= self.count0[yi]**2
        self.sum_sqr1 -= self.count1[yi]**2
        
        self.count0[yi]+=1
        self.count1[yi]-=1
        
        self.sum_sqr0 += self.count0[yi]**2
        self.sum_sqr1 += self.count1[yi]**2
        
        n0,n1 = self.n0,self.n1
        scr0 = 1 - self.sum_sqr0/(n0**2)
        scr1 = 0
        if n1!=0:
            scr1 = 1 - self.sum_sqr1/(n1**2)
        
        return (n0*scr0 + n1*scr1)/(n0+n1)
    
    def getNodeVal(self,y):
        return stats.mode(y).mode[0]
    

class DecisionTreeClassifier(DecisionTree):
    
    def __init__(self,criterion='gini',max_depth=np.inf,min_samples_split=2):
        super().__init__(max_depth,min_samples_split)
        self.criterion = criterion
        self.__gini = GiniIndex()
        self.method = {'gini':self.__gini}
        
    def fit(self,X,y):
        columns = X.columns
        methodObj = self.method[self.criterion]
        y.columns = ['y']
        df = pd.concat([X,y],axis=1)
        
        self.root.fitUtil(df,columns,methodObj,
                        0,self.max_depth,self.min_samples_split)
        


df = pd.read_csv('train.csv')
df = df[['Sex','Age','Survived']].copy().dropna()
df['Sex'] = df['Sex'] == 'female'
X,y = df[['Sex','Age']],df[['Survived']]

classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X,y)
y_pred = classifier.predict(X)

from sklearn.metrics import confusion_matrix as cm
cm(y,y_pred)

from sklearn.tree import DecisionTreeClassifier as DTC

classifier = DTC(max_depth=2)
classifier.fit(X,y)
y_pred1 = classifier.predict(X)

from sklearn.metrics import confusion_matrix as cm
cm(y,y_pred1)
