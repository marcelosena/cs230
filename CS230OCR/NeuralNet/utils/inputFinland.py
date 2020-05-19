#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:38:46 2020

@author: jonasmg
"""


import cv2
import numpy as np
from tensorflow import keras
import pickle

# path = '/home/jonasmg/Documents/OCRProject/'

from utils.BasicConfig import (IMGWIDTH, IMGHEIGHT, CHARDICT, DICTLENGTH, MAXCHARLENGTH)

path = 'cs230/CS230OCR/'

def loadImgsWithLabels(full=False):
    if full:
        name = 'ProcessedData/KerasReadyFull'
    else:
        name = 'ProcessedData/KerasReady'
    X       = np.load(path + name + '0.npy')
    Y_label = np.load(path + name + '1.npy')
    Y_len   = np.load(path + name + '2.npy')
    N       = np.load(path + name + '3.npy')[0]
    return X, Y_label,  Y_len,  N


class DataGen(keras.callbacks.Callback):
    def __init__(self, minibatch_size, SeqLength, X, Y_label, Y_len, N, currTrainIndex):
        self.minibatch_size = minibatch_size
        self.SeqLength = SeqLength
        self.X = X
        self.N = N
        self.Y_label = Y_label
        self.Y_len = Y_len
        self.currTrainIndex = currTrainIndex

    def get_batch(self, r=False):
        while 1:
            if not r:
                curr = self.currTrainIndex
                new = (self.currTrainIndex +self.minibatch_size)
                if new >= self.N:
                    curr = 0
                    new = self.minibatch_size
                    randomOrder = np.random.permutation(np.array(range(self.N)))
                    self.X = self.X[randomOrder]
                    self.Y_len = self.Y_len[randomOrder]
                    self.Y_label = self.Y_label[randomOrder]
                X_use = self.X[curr:new, :, :, :]
                Y_label_use = self.Y_label[curr:new, :]
                Y_len_use = self.Y_len[curr:new]
                self.currTrainIndex = new
            else:
                random = np.random.randint(self.N - self.minibatch_size)
                X_use = self.X[random:(random+self.minibatch_size), :, :, :]
                Y_label_use = self.Y_label[random:(random+self.minibatch_size), :]
                Y_len_use = self.Y_len[random:(random+self.minibatch_size)]

            inputs = {'the_input': X_use,
                      'the_labels': Y_label_use,
                      'input_length': np.ones([ self.minibatch_size,1])*self.SeqLength,
                      'label_length': Y_len_use,
                      }
            outputs = {'ctc': np.zeros([self.minibatch_size])} # dummy data for dummy loss function
            yield (inputs, outputs)
