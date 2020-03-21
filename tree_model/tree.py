# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:43:37 2020

@author: Samarendra
"""

import numpy as np
import pandas as pd

class Node:
    
    def __init__(self):
        self.isLeaf = False
        self.feature = None
        self.val = None
        self.left = None
        self.right = None
            
    def fitUtil(self,df,columns,methodObj,currDepth,mDepth,mSamples):
        
        if len(df) < mSamples or currDepth == mDepth:
            self.val = methodObj.getNodeVal(df['y'])
            self.isLeaf = True
            return
        
        minScore = np.inf
        prop = None
        value = None
        
        for col in columns:
            scr,val = self.evaluate(df,col,methodObj)
            if scr < minScore:
                minScore = scr
                prop = col
                value = val
            
        self.feature = prop
        self.val = value
        
        dfLeft = df[df[prop]<=value].copy()
        dfRight = df[df[prop]>value].copy()
        
        if len(dfLeft) == 0 or len(dfRight) == 0:
            self.val = methodObj.getNodeVal(df['y'])
            self.isLeaf = True
            return
        
        self.left,self.right = Node(),Node()
        
        self.left.fitUtil(dfLeft,columns,methodObj,
                                currDepth+1,mDepth,mSamples)
        self.right.fitUtil(dfRight,columns,methodObj,
                                currDepth+1,mDepth,mSamples)
        
    def evaluate(self,df,col,methodObj):
        df = df.sort_values([col])
        minScore = methodObj.initialScore(df['y'])
        val = None
        
        for i in df.index:
            score = methodObj.update(df['y'][i])
            if score < minScore:
                val = df[col][i]
                minScore = score
                
        return minScore,val
    
    
class DecisionTree:
    
    def __init__(self,max_depth=np.inf,min_samples_split=2):
        self.root = Node()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self,X,y):
        pass
    
    def predict(self,X):
        
        y_pred = []
        
        for _,row in X.iterrows():
            node = self.root
            while node.isLeaf == False:
                if row[node.feature] <= node.val:
                    node = node.left
                else:
                    node = node.right
            y_pred = np.append(y_pred,node.val)
            
        return np.array(y_pred)
    
class SplitMethod:
    def initialScore(self,y):
        pass
    def update(self,yi):
        pass