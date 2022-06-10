# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:39:23 2020

@author: Noah
"""
import numpy as np
from numpy import array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
def traindataload(snr):
	x = np.load('../../Dataset/train_SNR='+str(snr)+'.npy')
	y1=np.zeros([32000,1])
	y2=np.ones([52000,1])
	y3=np.ones([38000,1])*2
	y4=np.ones([32000,1])*3
	y5=np.ones([32000,1])*4
	y6=np.ones([38000,1])*5
	y7=np.ones([38000,1])*6
	y=np.vstack((y1,y2,y3,y4,y5,y6,y7))
	y = array(y)
	y = to_categorical(y)
	X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.3, random_state= 30)
	return X_train, X_val, Y_train, Y_val


def testdataload(snr):
	X_test = np.load('../../Dataset/test_SNR='+str(snr)+'.npy')
	y1=np.zeros([30000,1])
	y2=np.ones([30000,1])
	y3=np.ones([30000,1])*2
	y4=np.ones([30000,1])*3
	y5=np.ones([30000,1])*4
	y6=np.ones([30000,1])*5
	y7=np.ones([30000,1])*6
	y=np.vstack((y1,y2,y3,y4,y5,y6,y7))
	y = array(y)
	# one hot encode
	Y_test = to_categorical(y)
	return X_test, Y_test