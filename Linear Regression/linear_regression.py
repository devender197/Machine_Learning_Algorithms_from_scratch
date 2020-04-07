# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:43:06 2020

@author: Devender Singh Parihar
"""

import numpy as np

class LinearRegression:

    def __init__(self, data, labels, normalize_data=True):
        
        # if data is not normalized already then normalized data
        self.data = data
        self.labels = labels
        if(normalize_data == False):
            self.data =  self.normalize(data)
    
    def normalize(features):
    
        # Copy original array to prevent it from changes.
        features_normalized = np.copy(features).astype(float)
    
        # Get average values for each feature (column) in X.
        features_mean = np.mean(features, 0)
    
        # Calculate the standard deviation for each feature.
        features_deviation = np.std(features, 0)
    
        # Subtract mean values from each feature (column) of every example (row)
        # to make all features be spread around zero.
        if features.shape[0] > 1:
            features_normalized -= features_mean
    
        # Normalize each feature values so that all features are close to [-1:1] boundaries.
        # Also prevent division by zero error.
        features_deviation[features_deviation == 0] = 1
        features_normalized /= features_deviation
    
        return features_normalized, features_mean, features_deviation
    
    def leastSquareMethod(self):
        # x will be the input feature and y will be the response
        x = self.data
        y = self.labels
        
        # getting the mean of the data
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # calculating the least square using least square method
        slope = np.sum((x - mean_x)*(y - mean_y)) / np.sum((x - mean_x)**2)
        intercept = mean_y - slope * mean_x
        
        # return the calculated slope and intercept
        return slope, intercept
    
    def gradient_decent_method(self, learn_rate, number_iteration):
        # x will be the input feature and y will be the response
        X = self.data
        Y = self.labels
        
        # learning rate 
        learning_rate = learn_rate
        iteration = number_iteration
        #print(Y.shape, Y.values)
        expected = np.reshape(Y.values,(len(Y),1))
        data_x   = np.reshape(X.values, (len(X),1))
       
        # initial value of slope and intercept
        slope = 0
        intercept = 0
        for _ in range(iteration):
            predicted = slope * data_x + intercept
            slope = slope - learning_rate * self.partial_derivative_with_respect_to_slope(expected , predicted, data_x )
            intercept = intercept - learning_rate * self.partial_derivative_with_respect_to_intercept(expected , predicted)
        print(slope,intercept)
        
    def partial_derivative_with_respect_to_slope(self, expected, predicted, value_of_x):
        return (-2/len(expected)) * ( np.sum(np.multiply(value_of_x , np.subtract(expected , predicted))))
    
    def partial_derivative_with_respect_to_intercept(self, expected, predicted):
        return (-2/len(expected)) * ( np.sum(np.subtract(expected , predicted)))