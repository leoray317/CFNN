# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:00:51 2019

@author: steve
"""
#import the modules
import numpy as np
import math
import os
import matplotlib.pyplot as plt

#define the functions 
# f2 is the function that generate forecasting data according to the variable that contain random numbers which is data_in and data_real_out
def f2(p):
    n=p.shape[0]
    y=[];y=([[]*1 for i in range(0,500)])
    for j in range(0,n):
        y[j]=math.cos(p[j,0])*math.sin(p[j,1])
    y=np.asarray(y)
    return y

#np_R2 is the coefficient correlation function which can calculate the correlation of the forecasting data and the real output data
def np_R2(output,output_pred):
    return np.square(np.corrcoef(output.reshape(np.size(output)),output_pred.reshape(np.size(output_pred)), False)[1,0])

#np_RMSE is the root mean square error function which can calculate the error between the forecasting data and the real output data
def np_RMSE(output,output_pred):
    rmse=0
    for i in range(0,len(output)):
        rmse=rmse+np.square(output[i]-output_pred[i])
    rmse=np.sqrt(rmse/len(output))
    return rmse

#generate input data and output data
data_in= np.random.rand(500,2)*4-2
data_real_out =f2(data_in)

#setting the index of the training, validation and testing
train_index=np.asarray(range(0,300))
val_index=np.asarray(range(300,400))
test_index=np.asarray(range(400,500))
 
#split data into train,validate and testing
train_in= data_in[train_index,:]
train_out= data_real_out[train_index]
val_in= data_in[val_index,:]
val_out= data_real_out[val_index]
test_in= data_in[test_index,:]
test_out= data_real_out[test_index]
train_out=np.reshape(train_out,(-1,1));val_out=np.reshape(val_out,(-1,1));test_out=np.reshape(test_out,(-1,1))

#Please reset your directory to the location of the CFNN.py folder
os.chdir('C:/Users/user/Desktop')
#CFNN is the class that contain CFNN network. In this CFNN network contains 3 functions which is newCPN, evalGaussCPN, evalTriCPN
from CFNN import CFNN_network as CFNN
delta=0.1;beta=0.5;alpha=1;
Wi,Wp=CFNN.newCPN(train_in,train_out,delta=delta,nseed=math.ceil(len(train_in)/50),beta=beta,alpha1=alpha)  
#if use Gauss remove the annotation symbol "#"
#Yh_train=CFNN.evalGaussCPN(Wi,Wp,delta=delta,P=train_in) #Gauss function
Yh_train=CFNN.evalTriCPN(Wi,Wp,delta=delta,P=train_in) #triangulate function
#Yh_val=CFNN.evalGaussCPN(Wi,Wp,delta=delta,P=val_in) #Gauss function
Yh_val=CFNN.evalTriCPN(Wi,Wp,delta=delta,P=val_in) #triangulate function
#Yh_test=CFNN.evalGaussCPN(Wi,Wp,delta=delta,P=test_in) #Gauss function
Yh_test=CFNN.evalTriCPN(Wi,Wp,delta=delta,P=test_in) #triangulate function

#calculate the error between the forecasting data and the real output
train_error= train_out-Yh_train
val_error= val_out-Yh_val
test_error= test_out-Yh_test

#plot the results
#training
plt.subplot(2,1,1)
plt.plot(train_out)
plt.plot(Yh_train,'ro')
plt.ylabel('training')
plt.subplot(2,1,2)
plt.plot(train_error)
plt.ylabel("training-error")
plt.show()

#validation
plt.subplot(2,1,1)
plt.plot(val_out)
plt.plot(Yh_val,'yo')
plt.ylabel('validation')
plt.subplot(2,1,2)
plt.plot(val_error)
plt.ylabel("validation-error")
plt.show()

#testing
plt.subplot(2,1,1)
plt.plot(test_out)
plt.plot(Yh_test,'bo')
plt.ylabel('testing')
plt.subplot(2,1,2)
plt.plot(test_error)
plt.ylabel("testing-error")
plt.show()

#calculate the R2(coefficient determination) and the root mean square error of the forecasting data
train_R2=np_R2(train_out,Yh_train)
val_R2=np_R2(val_out,Yh_val)
test_R2=np_R2(test_out,Yh_test)

train_RMSE=np_RMSE(train_out,Yh_train)
val_RMSE=np_RMSE(val_out,Yh_val)
test_RMSE=np_RMSE(test_out,Yh_test)