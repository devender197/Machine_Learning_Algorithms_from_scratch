# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:24:05 2020
@author: Devender Singh Parihar
"""
import numpy as np

class KMeans:
   
    def __init__(self, trainX, trainY, k_cluster):
        
        self.trainX = trainX
        self.trainY = trainY
        
		# Number of cluster 
        self.k_cluster = k_cluster
		
		# Number of rows
        self.train_rows = trainX.shape[0]
		
		# Number of columns
        self.train_columns = trainX.shape[1]
    
    def train(self, iteration):
		# Generate centroid based on training set
        centroids  = self.initialCentroid()
		
		# default centroid array
        closest_centroids_ids = np.empty((self.num_row, 1))
		
		# find closest centroid
        for x in range(iteration):
            # Find the closest centroids for training examples.
            closest_centroids_ids = self.cal_centroid(centroids)

            # Compute means based on the closest centroids found in the previous part.
            centroids = self.computeCentroid(
                self.trainX,
                closest_centroids_ids,
                self.k_cluster
            )

        return centroids, closest_centroids_ids
    
    def computeCentroid(self,closest_centroids_ids):
	
		# number of columns in train set
        num_columns = self.train_columns
		
		# set of centroid.
        centroids = np.zeros((self.k_cluster, num_columns))
	
		
        for centroid_id in range(self.k_cluster):
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(self.trainX[closest_ids.flatten(), :], axis=0)

        return centroids
    
    def initialCentroid(self):
		
		# number of rows in train set
        num_row = self.train_rows
		
		# random ids
        random_ids = np.random.permutation(num_row)
		
		# get K random centroid 
        centroid = self.trainX[random_ids[:num_row],:]
        
        return centroid
    
    def cal_centroid(self , centroids):
		
		# number of rows
        num_row = self.train_rows

        # Get number of centroids.
        num_centroids = centroids.shape[0]

        # We need to return the following variables correctly.
        closest_centroids_ids = np.zeros((num_row, 1))

        # Calculate Centroid
        for example_index in range(num_row):
            distances = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                distance_difference = self.train[example_index, :] - centroids[centroid_index, :]
                distances[centroid_index] = np.sum(distance_difference ** 2)
            closest_centroids_ids[example_index] = np.argmin(distances)

        return closest_centroids_ids
		
		
		
		
		
		
		