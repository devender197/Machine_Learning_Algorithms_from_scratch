# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:07:46 2020

@author: HOME
"""

from linear_regression import LinearRegression
import pandas as pd

data = pd.read_csv("data.csv")
data.columns = ['X','Y']
dataX = data.X
dataY = data.Y
obj = LinearRegression(dataX,dataY,True)
#obj.gradient_decent_method(0.0001,1000)
print(obj.leastSquareMethod())