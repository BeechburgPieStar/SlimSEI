# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:18:51 2020

@author: Noah
"""
from SparseLightComplexNN.complexnn.conv import ComplexConv1D
from SparseLightComplexNN.complexnn.bn import ComplexBatchNormalization
from SparseLightComplexNN.complexnn.dense import ComplexDense
from SparseLightComplexNN.BinaryComplexSparsityRegularizationWithFISTA import ComplexSparsityRegularization
from SparseLightComplexNN.BinarySparsityRegularizationWithFISTA import SparsityRegularization
from keras.layers import Input, Add, MaxPooling1D, Activation,Dense,Conv1D,BatchNormalization
from keras.models import Model
import keras
from keras.layers.core import Dropout,Flatten
# from tensorflow.train import AdamOptimizer
# PGD = keras.optimizers.TFOptimizer(AdamOptimizer)

def SparseComplexCNNSameChannel(C1, C2, C3, C4, C5, C6, C7, C8, C9, D1, sf, threshold):  
	x_input = Input(shape=(1000, 2))
	x = ComplexConv1D(C1, 3, activation='relu', padding='same', name = 'conv1')(x_input)
	x = ComplexBatchNormalization(name = 'bn1')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr1')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C2, 3, activation='relu', padding='same', name = 'conv2')(x)
	x = ComplexBatchNormalization(name = 'bn2')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr2')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C3, 3, activation='relu', padding='same', name = 'conv3')(x)
	x = ComplexBatchNormalization(name = 'bn3')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr3')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C4, 3, activation='relu', padding='same', name = 'conv4')(x)
	x = ComplexBatchNormalization(name = 'bn4')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr4')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C5, 3, activation='relu', padding='same', name = 'conv5')(x)
	x = ComplexBatchNormalization(name = 'bn5')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr5')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C6, 3, activation='relu', padding='same', name = 'conv6')(x)
	x = ComplexBatchNormalization(name = 'bn6')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr6')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C7, 3, activation='relu', padding='same', name = 'conv7')(x)
	x = ComplexBatchNormalization(name = 'bn7')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr7')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C8, 3, activation='relu', padding='same', name = 'conv8')(x)
	x = ComplexBatchNormalization(name = 'bn8')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr8')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = ComplexConv1D(C9, 3, activation='relu', padding='same', name = 'conv9')(x)
	x = ComplexBatchNormalization(name = 'bn9')(x)
	x = ComplexSparsityRegularization(sf, threshold, name = 'sr9')(x)
	x = MaxPooling1D(pool_size= 2)(x)
	x = Flatten()(x)
	x = Dense(D1, activation='relu', name = 'dense1')(x)
	x = SparsityRegularization(sf, threshold, name = 'sr10')(x)
	x = Dropout(0.5)(x)
	x_output = Dense(7, activation='softmax', name = 'dense2')(x)
	model = Model(inputs=x_input, 
              outputs=x_output)
	model.compile(loss='categorical_crossentropy',
	              optimizer= 'sgd',
	              metrics=['accuracy'])
	model.summary()
	return model