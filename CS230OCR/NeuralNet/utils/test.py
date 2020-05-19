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
import pandas as pd

# path = '/home/jonasmg/Documents/OCRProject/'
bw1 = path + 'ScannedCensus/ExtractedBW_V1/'
bw2 = path + 'ScannedCensus/ExtractedBW_V2/'
from utils.BasicConfig import (IMGWIDTH, IMGHEIGHT, CHARDICT, DICTLENGTH, MAXCHARLENGTH)


def preproc(fn):
    img = cv2.imread(fn, 0)
    img = cv2.resize(img, (IMGWIDTH,IMGHEIGHT), interpolation = cv2.INTER_AREA)
    img =  (255 - img)/255
    return img.T

def labler(x):
    ret = []
    for c in x:
        ret.append(CHARDICT[str(c)])
    return ret

def loadImgsWithLabels():
    labels = pd.read_csv(path + 'ScannedCensus/FullData1940.csv')
    labels = labels.fillna(DICTLENGTH)
    labels['V'] = labels['V'].apply(lambda x: str(x))
    N = len(labels)

    X = np.zeros([N, IMGWIDTH, IMGHEIGHT])
    Y_len = np.zeros([N,1])
    Y_label = np.ones([N , MAXCHARLENGTH]) * -1
    for k,f in enumerate(labels.Cell.tolist()):
        X[k, :, :] = preproc(bw2 + f +'.png')
        Y_len[k] = len(labels.iloc[k, 0])
        Y_label[k, 0:int(Y_len[k])] = labler(labels.iloc[k, 0])
    X = X.reshape([N, IMGWIDTH, IMGHEIGHT,1])
    # randomOrder = np.random.permutation(np.array(range(N)))
    # X = X[randomOrder]
    # Y_len = Y_len[randomOrder]
    # Y_label = Y_label[randomOrder]
    return X, Y_label,  Y_len,  N


# def loadImgsWithLabels():
#     X       = np.load(path + 'ProcessedData/KerasReady0.npy')
#     Y_label = np.load(path + 'ProcessedData/KerasReady1.npy')
#     Y_len   = np.load(path + 'ProcessedData/KerasReady2.npy')
#     N       = np.load(path + 'ProcessedData/KerasReady3.npy')[0]
#     return X, Y_label,  Y_len,  N


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
