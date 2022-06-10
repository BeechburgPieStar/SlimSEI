# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:40:18 2020

@author: Noah
"""

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

snr = 35
[sf, threshold] = [1e-3, 0.5]
epochs = 400

from NetworkForPruning import SparseComplexCNNSameChannel
from DataLoad import traindataload, testdataload
from SparseLightComplexNN.CustomLayersDictionary import customLayerCallbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np

def FirstComplexConv1DPruning(layer, shortNowConv, shortNowConvHalf):
    weight=layer.get_weights()
    w = weight[0]
    b = weight[1]

    w_ = np.delete(w, shortNowConv, axis=2)
    b_ = np.delete(b, shortNowConv, axis=0)

    new_weight = []
    new_weight.append(w_)
    new_weight.append(b_)
    return new_weight

def ComplexConv1DPruning(layer, shortLastConv, shortLastConvHalf, shortNowConv, shortNowConvHalf):
    weight=layer.get_weights()
    w = weight[0]
    b = weight[1]

    w_2 = np.delete(w, shortNowConv, axis=2)
    w_1 = np.delete(w_2, shortLastConvHalf, axis=1)

    b_ = np.delete(b, shortNowConv, axis=0)

    new_weight = []
    new_weight.append(w_1)
    new_weight.append(b_)
    return new_weight

def FirstDensePruning(layer, shortLastConv, shortNowDense):
    weight=layer.get_weights()
    w = weight[0]
    w = np.reshape(w,[1, 128, 1024])#62是上一层pooling的数量, 256是上一层神经元的数量，1024是Dense的神经元数量
    
    b = weight[1]
    
    w_1 = np.delete(w, shortNowDense, axis=2)
    w_2 = np.delete(w_1, shortLastConv, axis=1)
    dim = np.shape(w_2)

    w_ = np.reshape(w_2, [dim[0]*dim[1],-1])
    
    b_ = np.delete(b, shortNowDense, axis=0)

    new_weight = []
    new_weight.append(w_)
    new_weight.append(b_)
    return new_weight

def LastDensePruning(layer, shortLastDense):
    weight=layer.get_weights()
    w = weight[0]
    b = weight[1]   

    w_ = np.delete(w, shortLastDense, axis=0)
    
    new_weight = []
    new_weight.append(w_)
    new_weight.append(b)
    return new_weight

def BNPruning(layer, shortLastLayer, shortLastLayerHalf):
    new_weight = []
    weight=layer.get_weights()
    for i in [0, 1, 2]:
        w = weight[i]
        w_ = np.delete(w, shortLastLayerHalf, axis=0)    
        new_weight.append(w_)

    w = weight[3]
    w_ = np.delete(w, shortLastLayer, axis=0)    
    new_weight.append(w_)

    for i in [4, 5, 6]:
        w = weight[i]
        w_ = np.delete(w, shortLastLayerHalf, axis=0)    
        new_weight.append(w_)

    w = weight[7]
    w_ = np.delete(w, shortLastLayer, axis=0)    
    new_weight.append(w_)

    return new_weight

def SparseRegularizationPruning(layer, shortLastLayer):
    weight=layer.get_weights()
    w = weight[0]

    w_ = np.delete(w, shortLastLayer, axis=0)    

    new_weight = []
    new_weight.append(w_)
    return new_weight

# model create
[C1, C2, C3, C4, C5, C6, C7, C8, C9, D1] = [64, 64, 64, 64, 64, 64, 64, 64, 64, 1024]
model = SparseComplexCNNSameChannel(C1, C2, C3, C4, C5, C6, C7, C8, C9, D1, sf, threshold)

#load training data

X_train, X_val, Y_train, Y_val = traindataload(snr)

#training
modelname = 'SparseComplexCNNSameChannel'
checkpoint = ModelCheckpoint(modelname + str(snr) + ".hdf5",
 	                          verbose=1,
 	                          save_best_only=True)
tensorboard = TensorBoard(modelname + str(snr) + ".log", 0)
earlystopping = EarlyStopping(monitor='val_loss', patience=500, verbose=0)

model.fit(X_train,
          Y_train,
          batch_size=128,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, Y_val),
          callbacks=[checkpoint, tensorboard, earlystopping] + customLayerCallbacks)

X_test, Y_test = testdataload(snr)
model.load_weights(modelname + str(snr) + ".hdf5")
[loss, acc] = model.evaluate(X_test, Y_test, batch_size = 100, verbose=0)

#pruning
layer_name = ['sr1', 'sr2', 'sr3', 'sr4', 'sr5', 'sr6', 'sr7', 'sr8', 'sr9', 'sr10']#SF
gamma_threshold = 0
for l in layer_name:
    layer=model.get_layer(l)
    weight=layer.get_weights()
    gamma=weight[0]
    dim = np.shape(gamma)[0]//2
    low_gamma_half = np.where(gamma[:dim] == gamma_threshold)[0]
    low_gamma = np.where(gamma == gamma_threshold)[0]
    if l == 'sr1':
        C1 = C1 - np.shape(low_gamma_half)[0]#complex conv
        shortConv1 = low_gamma
        shortConv1Half = low_gamma_half        
    elif l == 'sr2':
        C2 = C2 - np.shape(low_gamma_half)[0]#complex conv
        shortConv2 = low_gamma
        shortConv2Half = low_gamma_half 
    elif l == 'sr3':
        C3 = C3 - np.shape(low_gamma_half)[0]#complex conv
        shortConv3 = low_gamma
        shortConv3Half = low_gamma_half
    elif l == 'sr4':
        C4 = C4 - np.shape(low_gamma_half)[0]#complex conv
        shortConv4 = low_gamma
        shortConv4Half = low_gamma_half
    if l == 'sr5':
        C5 = C5 - np.shape(low_gamma_half)[0]#complex conv
        shortConv5 = low_gamma
        shortConv5Half = low_gamma_half        
    elif l == 'sr6':
        C6 = C6 - np.shape(low_gamma_half)[0]#complex conv
        shortConv6 = low_gamma
        shortConv6Half = low_gamma_half 
    elif l == 'sr7':
        C7 = C7 - np.shape(low_gamma_half)[0]#complex conv
        shortConv7 = low_gamma
        shortConv7Half = low_gamma_half
    elif l == 'sr8':
        C8 = C8 - np.shape(low_gamma_half)[0]#complex conv
        shortConv8 = low_gamma
        shortConv8Half = low_gamma_half
    elif l == 'sr9':
        C9 = C9 - np.shape(low_gamma_half)[0]#complex conv
        shortConv9 = low_gamma
        shortConv9Half = low_gamma_half
    elif l == 'sr10':
        #D1 = D1 - np.shape(low_gamma_half)[0]#complex dense
        D1 = D1 - np.shape(low_gamma)[0]#real dense
        shortDense1 = low_gamma
        shortDense1Half = low_gamma_half

#light model create
lightmodel = SparseComplexCNNSameChannel(C1, C2, C3, C4, C5, C6, C7, C8, C9, D1, 0, 0)

#load light model weight
layer = model.get_layer('conv1')
lightweight_Conv1 = FirstComplexConv1DPruning(layer, shortConv1, shortConv1Half)
lightlayer = lightmodel.get_layer('conv1')
lightlayer.set_weights(lightweight_Conv1)
 
layer = model.get_layer('conv2')
lightweight_Conv2 = ComplexConv1DPruning(layer, shortConv1, shortConv1Half, shortConv2, shortConv2Half)
lightlayer = lightmodel.get_layer('conv2')
lightlayer.set_weights(lightweight_Conv2)

layer = model.get_layer('conv3')
lightweight_Conv3 = ComplexConv1DPruning(layer, shortConv2, shortConv2Half, shortConv3, shortConv3Half)
lightlayer = lightmodel.get_layer('conv3')
lightlayer.set_weights(lightweight_Conv3)

layer = model.get_layer('conv4')
lightweight_Conv4 = ComplexConv1DPruning(layer, shortConv3, shortConv3Half, shortConv4, shortConv4Half)
lightlayer = lightmodel.get_layer('conv4')
lightlayer.set_weights(lightweight_Conv4)

layer = model.get_layer('conv5')
lightweight_Conv5 = ComplexConv1DPruning(layer, shortConv4, shortConv4Half, shortConv5, shortConv5Half)
lightlayer = lightmodel.get_layer('conv5')
lightlayer.set_weights(lightweight_Conv5)

layer = model.get_layer('conv6')
lightweight_Conv6 = ComplexConv1DPruning(layer, shortConv5, shortConv5Half, shortConv6, shortConv6Half)
lightlayer = lightmodel.get_layer('conv6')
lightlayer.set_weights(lightweight_Conv6)

layer = model.get_layer('conv7')
lightweight_Conv7 = ComplexConv1DPruning(layer, shortConv6, shortConv6Half, shortConv7, shortConv7Half)
lightlayer = lightmodel.get_layer('conv7')
lightlayer.set_weights(lightweight_Conv7)

layer = model.get_layer('conv8')
lightweight_Conv8 = ComplexConv1DPruning(layer, shortConv7, shortConv7Half, shortConv8, shortConv8Half)
lightlayer = lightmodel.get_layer('conv8')
lightlayer.set_weights(lightweight_Conv8)

layer = model.get_layer('conv9')
lightweight_Conv9 = ComplexConv1DPruning(layer, shortConv8, shortConv8Half, shortConv9, shortConv9Half)
lightlayer = lightmodel.get_layer('conv9')
lightlayer.set_weights(lightweight_Conv9)

layer = model.get_layer('dense1')
lightweight_Dense1 = FirstDensePruning(layer, shortConv9, shortDense1)
lightlayer = lightmodel.get_layer('dense1')
lightlayer.set_weights(lightweight_Dense1)

layer = model.get_layer('dense2')
lightweight_Dense2 = LastDensePruning(layer, shortDense1)
lightlayer = lightmodel.get_layer('dense2')
lightlayer.set_weights(lightweight_Dense2)

layer = model.get_layer('bn1')
lightweight_BN = BNPruning(layer, shortConv1, shortConv1Half)
lightlayer = lightmodel.get_layer('bn1')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn2')
lightweight_BN = BNPruning(layer, shortConv2, shortConv2Half)
lightlayer = lightmodel.get_layer('bn2')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn3')
lightweight_BN = BNPruning(layer, shortConv3, shortConv3Half)
lightlayer = lightmodel.get_layer('bn3')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn4')
lightweight_BN = BNPruning(layer, shortConv4, shortConv4Half)
lightlayer = lightmodel.get_layer('bn4')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn5')
lightweight_BN = BNPruning(layer, shortConv5, shortConv5Half)
lightlayer = lightmodel.get_layer('bn5')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn6')
lightweight_BN = BNPruning(layer, shortConv6, shortConv6Half)
lightlayer = lightmodel.get_layer('bn6')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn7')
lightweight_BN = BNPruning(layer, shortConv7, shortConv7Half)
lightlayer = lightmodel.get_layer('bn7')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn8')
lightweight_BN = BNPruning(layer, shortConv8, shortConv8Half)
lightlayer = lightmodel.get_layer('bn8')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('bn9')
lightweight_BN = BNPruning(layer, shortConv9, shortConv9Half)
lightlayer = lightmodel.get_layer('bn9')
lightlayer.set_weights(lightweight_BN)

layer = model.get_layer('sr1')
lightweight_SR = SparseRegularizationPruning(layer, shortConv1)
lightlayer = lightmodel.get_layer('sr1')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr2')
lightweight_SR = SparseRegularizationPruning(layer, shortConv2)
lightlayer = lightmodel.get_layer('sr2')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr3')
lightweight_SR = SparseRegularizationPruning(layer, shortConv3)
lightlayer = lightmodel.get_layer('sr3')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr4')
lightweight_SR = SparseRegularizationPruning(layer, shortConv4)
lightlayer = lightmodel.get_layer('sr4')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr5')
lightweight_SR = SparseRegularizationPruning(layer, shortConv5)
lightlayer = lightmodel.get_layer('sr5')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr6')
lightweight_SR = SparseRegularizationPruning(layer, shortConv6)
lightlayer = lightmodel.get_layer('sr6')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr7')
lightweight_SR = SparseRegularizationPruning(layer, shortConv7)
lightlayer = lightmodel.get_layer('sr7')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr8')
lightweight_SR = SparseRegularizationPruning(layer, shortConv8)
lightlayer = lightmodel.get_layer('sr8')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr9')
lightweight_SR = SparseRegularizationPruning(layer, shortConv9)
lightlayer = lightmodel.get_layer('sr9')
lightlayer.set_weights(lightweight_SR)

layer = model.get_layer('sr10')
lightweight_SR = SparseRegularizationPruning(layer, shortDense1)
lightlayer = lightmodel.get_layer('sr10')
lightlayer.set_weights(lightweight_SR)

lightmodelname = 'LightSparseComplexCNNSameChannel'
lightmodel.save_weights(lightmodelname + str(snr) + ".hdf5")

lightmodel.load_weights(lightmodelname + str(snr) + ".hdf5")
[loss, lightacc] = lightmodel.evaluate(X_test, Y_test, batch_size = 100, verbose=0)

print(acc)
print(lightacc)
print([C1, C2, C3, C4, C5, C6, C7, C8, C9, D1])

