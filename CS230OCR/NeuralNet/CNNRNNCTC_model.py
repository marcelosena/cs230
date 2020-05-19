#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:35:54 2020

@author: jonasmg
"""


import sys, os, pickle, time
# sys.path.append('/home/jonasmg/Documents/OCRProject/NeuralNet/')
# path  = '/home/jonasmg/Documents/OCRProject/'
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Flatten, Dense, Conv2D, MaxPool2D,
                                GlobalAveragePooling2D, Activation, BatchNormalization,
                                Dropout,GRU,Bidirectional, Reshape, Lambda)
from tensorflow.keras import backend as K

from utils.inputFinland import loadImgsWithLabels, DataGen
from utils.Callbacks import VizCallback
from utils.BasicConfig import (BATCH_SIZE, IMGWIDTH, IMGHEIGHT, CHARDICT, DICTLENGTH,
                               MAXCHARLENGTH, INPUTSHAPE, LOGDIR, SAVEDIR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


run_name = str(int(time.time()))
if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)
if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
if not os.path.exists(os.path.join(SAVEDIR, run_name)):
            os.makedirs(os.path.join(SAVEDIR, run_name))
if not os.path.exists(os.path.join(LOGDIR, run_name)):
            os.makedirs(os.path.join(LOGDIR, run_name))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


################################################
#%%
X, Y_label, Y_len,  N = loadImgsWithLabels()
randomOrder = pickle.load(open(path + f'ProcessedData/orderRestr.p', 'rb'))
X = X[randomOrder]
Y_label = Y_label[randomOrder]
Y_len = Y_len[randomOrder]

trainID = int(np.ceil(N*0.33))
devID = trainID + int(np.ceil((N - trainID)/2))

X_train = X[:trainID]
Y_label_train = Y_label[:trainID]
Y_len_train = Y_len[:trainID]

X_dev = X[trainID:devID]
Y_label_dev = Y_label[trainID:devID]
Y_len_dev = Y_len[trainID:devID]

X_test = X[devID:]
Y_label_test = Y_label[devID:]
Y_len_test = Y_len[devID:]

################################################
#%%
# Layer params:   Filts K  Padding  Name
layer_params = [ [  64, 7, 'same',  'conv1'],
                 [  64, 5, 'same',  'conv2'],
                 [ 128, 3, 'same',  'conv3'],
                 [ 128, 3, 'same',  'conv4'],
                 [ 256, 3, 'same',  'conv5'],
                 [ 256, 3, 'same',  'conv6'],
                 # [ 512, 3, 'same',  'conv7'],
                 # [ 512, 3, 'same',  'conv8'],
                 # [ 1024, 3, 'same',  'conv9'],
                 # [ 1024, 3, 'same',  'conv10'],
                 # [ 1024, 3, 'same',  'conv11'],
                 ]
wherePool = [1,3,5]


# Network parameters
kernel_size = (3, 3)
pool_size = (2,2)
denseSize = 128
rnn_size = 256
act = 'relu'
minibatch_size = BATCH_SIZE


inputs = Input(name='the_input', shape=INPUTSHAPE)
layer = layer_params[0]
inner = Conv2D(layer[0], layer[1], strides=(1,1), padding=layer[2], activation=act,
                     use_bias=False, kernel_initializer="he_normal", name=layer[3] )(inputs)
inner = BatchNormalization(momentum=0.95, epsilon=1e-05, center=True, scale=True)(inner)
for i in range(1,len(layer_params)):
    layer = layer_params[i]
    inner = Conv2D(layer[0], layer[1], strides=(1,1), padding=layer[2], activation=act,
                     use_bias=False, kernel_initializer="he_normal", name=layer[3] )(inner)
    inner = BatchNormalization(momentum=0.95, epsilon=1e-05, center=True, scale=True)(inner)

    if i in wherePool:
       inner = MaxPool2D(pool_size=pool_size, strides=(2, 2), padding='same', name=f'pool{i+1}')(inner)

conv_to_rnn_dims = ( IMGWIDTH// (pool_size[0] ** len(wherePool))
                    ,(IMGHEIGHT  // (pool_size[0] ** len(wherePool))) * layer_params[-1][0]
                   )
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
inner = Dense(denseSize, activation=act, name='dense1')(inner)
inner = BatchNormalization(momentum=0.95, epsilon=1e-05, center=True, scale=True)(inner)

# Two layers of bidirectional GRUs
gru_1 = Bidirectional(GRU(rnn_size, return_sequences=True, recurrent_dropout=0.5, dropout=0.5,
            kernel_initializer='he_normal', name='gru1'), merge_mode='sum')(inner)
gru_2 = Bidirectional(GRU(rnn_size, return_sequences=True, recurrent_dropout=0.5, dropout=0.5,
            kernel_initializer='he_normal', name='gru2'))(gru_1)
y_pred = Dense(DICTLENGTH, kernel_initializer='he_normal', name='densePredict', activation='softmax')(gru_2)
keras.Model(inputs=inputs, outputs=y_pred).summary()

#%%
labels = Input(name='the_labels', shape=[MAXCHARLENGTH], dtype='float32')
input_length = Input(name='input_length',
                     shape=[1], #IMGWIDTH// (pool_size[0] ** len(wherePool)) - 2
                     dtype='int64')
label_length = Input(name='label_length',
                     shape=[1],
                     dtype='int64')

#CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
    name='ctc')([y_pred, labels, input_length, label_length])

sgd = keras.optimizers.SGD(learning_rate=0.02,
          decay=1e-6,
          momentum=0.9,
          nesterov=True)
model = keras.Model(inputs=[inputs, labels, input_length, label_length],
              outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=["accuracy"])
test_func = K.function([inputs], [y_pred])

model.summary()



#%%
##############################################################################



generatorTrain = DataGen(BATCH_SIZE,  (IMGWIDTH// (pool_size[0] ** len(wherePool))-2),
                    X_train, Y_label_train, Y_len_train,  len(X_train), 0)
generatorVal = DataGen(BATCH_SIZE,  (IMGWIDTH// (pool_size[0] ** len(wherePool))-2),
                    X_dev, Y_label_dev, Y_len_dev,  len(X_dev), 0)

viz_cb = VizCallback(run_name, test_func, generatorVal.get_batch())
early_stopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(LOGDIR, run_name) + "/bestRunning_model.hdf5",
                                             monitor='loss', verbose=1,
                                             save_best_only=True, mode='auto')

#%%
history = model.fit(generatorTrain.get_batch(), epochs=55,
                    steps_per_epoch = len(X_train)//(BATCH_SIZE),
                    validation_data = generatorVal.get_batch(),
                    validation_steps = len(X_dev)//BATCH_SIZE,
                    callbacks=[viz_cb, early_stopping, checkpoint],)
model.save(os.path.join(SAVEDIR, run_name) + "/last_model.h5")
