# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:47:55 2021

@author: zhy
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import math
import numpy as np 
import tensorflow as tf
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


xx = x_train[0:2,0:2,0:2,:]
print(xx)
xx_mean = np.mean(xx, axis=(1, 2, 3), keepdims=True)
xx_std = np.mean(xx ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5
print('---')
print(xx_mean)
print(xx_std)

