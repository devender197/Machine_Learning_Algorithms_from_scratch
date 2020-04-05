# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:07:53 2020

@author: HOME
"""

from K_Mean import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("iris.csv")


dataX = data[{'sepal_length', 'sepal_width', 'petal_length', 'petal_width'}]

dataY = data[{'class'}]

plt.scatter(dataX.sepal_length,dataX.sepal_width)
plt.show()

x = KMeans(np.array(dataX),[],3)


centroids, closest_centroids_ids = x.train(50)

dataX['centroid'] = closest_centroids_ids

plt.scatter(dataX[dataX.centroid == 2].sepal_length,dataX[dataX.centroid == 2].sepal_width,color = 'red')
plt.scatter(dataX[dataX.centroid == 1].sepal_length,dataX[dataX.centroid == 1].sepal_width,color = 'blue')
plt.scatter(dataX[dataX.centroid == 0].sepal_length,dataX[dataX.centroid == 0].sepal_width,color = 'green')
plt.scatter(data[data['class'] == 'SETOSA'].sepal_length,data[data['class'] == 'SETOSA'].sepal_width,color = 'red')
plt.scatter(data[data['class'] == 'VERSICOLOR'].sepal_length,data[data['class'] == 'VERSICOLOR'].sepal_width,color = 'blue')
plt.scatter(data[data['class'] == 'VIRGINICA'].sepal_length,data[data['class']  == 'VIRGINICA'].sepal_width,color = 'green')
plt.show()