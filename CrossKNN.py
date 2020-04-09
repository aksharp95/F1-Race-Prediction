#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:39:42 2019

@author: Akshar Panchal and Kalpnil Anjan

File: KNN with Cross Validation and Graphs
"""

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

new_data_temp = pd.read_csv("Preprocessed _Data.csv")
final_sample_data = new_data_temp.iloc[:, [5, 6, 7, 8, 9, 11, 12]]

resultTop10 = []
resultTop5 = []
resultPodium = []
print(new_data_temp.columns)
for val in new_data_temp['DriverPosition']:
    if(np.isnan(val) & val < 11):
        resultTop10.append(1)
    else:
        resultTop10.append(0)
    if(val < 4):
        resultPodium.append(1)
    else:
        resultPodium.append(0)
        
    if(val < 6):
        resultTop5.append(1)
    else:
        resultTop5.append(0)

new_data_temp['Top 10 Finish'] = resultTop10 
new_data_temp['Top 5 Finish'] = resultTop5
new_data_temp['Podium Finish'] = resultPodium

X = final_sample_data

Y = new_data_temp['DriverPosition']

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean()*100)
# plot to see clearly
plt.title('Accuracy of KNN for 5-fold Cross Validation for Driver Position')
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.savefig("driverpos.pdf")
plt.show()


Y = new_data_temp['Top 10 Finish']

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean()*100)
# plot to see clearly
plt.title('Accuracy of KNN for 5-fold Cross Validation for Top 10 Finish')
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.savefig("top10.pdf")
plt.show()


Y = new_data_temp['Top 5 Finish']

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean()*100)
# plot to see clearly
plt.title('Accuracy of KNN for 5-fold Cross Validation for Top 5 Position')
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.savefig("top5.pdf")
plt.show()


Y = new_data_temp['Podium Finish']

# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
# plot to see clearly
plt.title('Accuracy of KNN for 5-fold Cross Validation for Podium Finish')
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.savefig("podium.pdf")
plt.show()

