#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:37:07 2019

@author: Akshar Panchal and Kalpnil Anjan
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 


resultTop10 = []
resultTop5 = []
resultPodium = []
new_data_temp = pd.read_csv("Preprocessed _Data.csv")
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

#For Driver Position
final_sample_data = new_data_temp.iloc[:, [5, 6, 7, 8, 9, 11, 12]]
X = final_sample_data
Y = new_data_temp['DriverPosition']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y,test_size=0.2, random_state = 0) 
print("X",X_test1.shape,"and Y:",y_test1.shape)

mlp = MLPClassifier()
mlp.fit(X_train1,y_train1)
y_train_predict = mlp.predict(X_train1)
train_acc_pos = accuracy_score(y_train_predict,y_train1)
y_test_predict = mlp.predict(X_test1)
test_acc_pos = accuracy_score(y_test_predict,y_test1)


# For Top 10 Position
final_sample_data = new_data_temp.iloc[:, [5, 6, 7, 8, 9, 11, 12]]
X = final_sample_data
Y = new_data_temp['Top 10 Finish']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y,test_size=0.2, random_state = 0) 
print("X",X_test1.shape,"and Y:",y_test1.shape)

mlp = MLPClassifier()
mlp.fit(X_train1,y_train1)
y_train_predict = mlp.predict(X_train1)
train_acc_top10 = accuracy_score(y_train_predict,y_train1)
y_test_predict = mlp.predict(X_test1)
test_acc_top10 = accuracy_score(y_test_predict,y_test1)


#For Top 5 Position
final_sample_data = new_data_temp.iloc[:, [5, 6, 7, 8, 9, 11, 12]]
X = final_sample_data
Y = new_data_temp['Top 5 Finish']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y,test_size=0.2, random_state = 0) 
print("X",X_test1.shape,"and Y:",y_test1.shape)

mlp = MLPClassifier()
mlp.fit(X_train1,y_train1)
y_train_predict = mlp.predict(X_train1)
train_acc_Top5 = accuracy_score(y_train_predict,y_train1)
y_test_predict = mlp.predict(X_test1)
test_acc_Top5 = accuracy_score(y_test_predict,y_test1)


#For Podium
final_sample_data = new_data_temp.iloc[:, [5, 6, 7, 8, 9, 11, 12]]
X = final_sample_data
Y = new_data_temp['Podium Finish']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y,test_size=0.2, random_state = 0) 
print("X",X_test1.shape,"and Y:",y_test1.shape)

mlp = MLPClassifier()
mlp.fit(X_train1,y_train1)
y_train_predict = mlp.predict(X_train1)
train_acc_pod = accuracy_score(y_train_predict,y_train1)
y_test_predict = mlp.predict(X_test1)
test_acc_pod = accuracy_score(y_test_predict,y_test1)

print("For DriverPosition\nTrain Accuracy", train_acc_pos, "\tTest Accuracy", test_acc_pos)
print("For Top 10 Position\nTrain Accuracy", train_acc_top10, "\tTest Accuracy", test_acc_top10)
print("For Top 5 Position\nTrain Accuracy", train_acc_Top5, "\tTest Accuracy", test_acc_Top5)
print("For Podium Finish\nTrain Accuracy", train_acc_pod, "\tTest Accuracy", test_acc_pod)


print("Latest Prediction\n")
# Predict Latest Races
X_train = final_sample_data[:-20]
Y_train = new_data_temp['DriverPosition'][:-20]
print("X",X_train.shape,"and Y:",Y_train.shape)
X_test= final_sample_data[-20:]
Y_test = new_data_temp['DriverPosition'][-20:]

mlp = MLPClassifier()
mlp.fit(X_train,Y_train)
y_train_predict = mlp.predict(X_train)
train_acc_pod = accuracy_score(y_train_predict,Y_train)
y_test_predict = mlp.predict(X_test)
test_acc_pod = accuracy_score(y_test_predict,Y_test)

print("For Driver Position\nTrain Accuracy", train_acc_pod, "\tTest Accuracy", test_acc_pod)


# Predict Latest Races
X_train = final_sample_data[:-20]
Y_train = new_data_temp['Podium Finish'][:-20]
print("X",X_train.shape,"and Y:",Y_train.shape)
X_test= final_sample_data[-20:]
Y_test = new_data_temp['Podium Finish'][-20:]


mlp = MLPClassifier()
mlp.fit(X_train,Y_train)
y_train_predict = mlp.predict(X_train)
train_acc_pod = accuracy_score(y_train_predict,Y_train)
y_test_predict = mlp.predict(X_test)
test_acc_pod = accuracy_score(y_test_predict,Y_test)


print("For Podium Finish\nTrain Accuracy", train_acc_pod, "\tTest Accuracy", test_acc_pod)


# Predict Latest Races
X_train = final_sample_data[:-20]
Y_train = new_data_temp['Top 10 Finish'][:-20]
print("X",X_train.shape,"and Y:",Y_train.shape)
X_test= final_sample_data[-20:]
Y_test = new_data_temp['Top 10 Finish'][-20:]


mlp = MLPClassifier()
mlp.fit(X_train,Y_train)
y_train_predict = mlp.predict(X_train)
train_acc_pod = accuracy_score(y_train_predict,Y_train)
y_test_predict = mlp.predict(X_test)
test_acc_pod = accuracy_score(y_test_predict,Y_test)

print("For Top 10 Finish\nTrain Accuracy", train_acc_pod, "\tTest Accuracy", test_acc_pod)


# Predict Latest Races
X_train = final_sample_data[:-20]
Y_train = new_data_temp['Top 5 Finish'][:-20]
print("X",X_train.shape,"and Y:",Y_train.shape)
X_test= final_sample_data[-20:]
Y_test = new_data_temp['Top 5 Finish'][-20:]


mlp = MLPClassifier()
mlp.fit(X_train,Y_train)
y_train_predict = mlp.predict(X_train)
train_acc_pod = accuracy_score(y_train_predict,Y_train)
y_test_predict = mlp.predict(X_test)
test_acc_pod = accuracy_score(y_test_predict,Y_test)

print("For Top 5 Finish\nTrain Accuracy", train_acc_pod, "\tTest Accuracy", test_acc_pod)
