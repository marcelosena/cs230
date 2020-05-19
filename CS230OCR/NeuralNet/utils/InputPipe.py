#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:27:54 2020

@author: jonasmg
"""
import os
import cv2
import tensorflow as tf
import numpy as np

MNIST_IMG_SIZE = 28
MNIST_CLASSES = 10
PARALLEL_INPUT_CALLS = 12

from tensorflow.keras.datasets import mnist
from extra_keras_datasets import emnist

def load_data(justMNIST = False):
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    if not justMNIST:       
        (emnist_train, emnist_train_target), (emnist_test, emnist_test_target) = emnist.load_data(type='digits')
        
        X_train_full = np.append(X_train_full, emnist_train, axis=0)
        X_test = np.append(X_test, emnist_test, axis=0)
        y_train_full = np.append(y_train_full, emnist_train_target, axis=0)
        y_test = np.append(y_test, emnist_test_target, axis=0)
        
    N = len(X_train_full)
    NTest = len(X_test)
    X_test = X_test.reshape(NTest,MNIST_IMG_SIZE,MNIST_IMG_SIZE,1)

    randomOrder = np.random.permutation(np.array(range(N)))
    X_train_full = X_train_full[randomOrder].reshape(N,MNIST_IMG_SIZE,MNIST_IMG_SIZE,1)
    y_train_full = y_train_full[randomOrder]
    
    split = int(np.ceil(N*0.9))
    X_valid = X_train_full[split:]
    X_train = X_train_full[:split]
    y_valid = y_train_full[split:]
    y_train = y_train_full[:split]
    
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train = tf.data.Dataset.zip((X_train, y_train))
    
    X_valid = tf.data.Dataset.from_tensor_slices(X_valid)
    y_valid = tf.data.Dataset.from_tensor_slices(y_valid)
    valid = tf.data.Dataset.zip((X_valid, y_valid))

    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    y_test = tf.data.Dataset.from_tensor_slices(y_test) 
    test = tf.data.Dataset.zip((X_test, y_test))
    
    return train, valid, test, N
    

class pipelineTrain(object):
    def __init__(self,DataLength):
        self.DataLength = DataLength

    @staticmethod
    def image_read(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label        
    

    # @staticmethod
    # def image_center(image, label):
    #     mean  = tf.reduce_mean(image)
    #     image = tf.maximum(tf.subtract(image, mean),mean)
    #     # std  = tf.math.reduce_std(image)
    #     # image = tf.divide(image, std)
    #     return image, label
        
    @staticmethod
    def image_shift_rand(image, label):
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
        nonzero_x_cols = tf.cast(tf.where(tf.greater(
            tf.reduce_sum(image, axis=0), 0)), tf.int32)
        nonzero_y_rows = tf.cast(tf.where(tf.greater(
            tf.reduce_sum(image, axis=1), 0)), tf.int32)
        left_margin = tf.reduce_min(nonzero_x_cols)
        right_margin = MNIST_IMG_SIZE - tf.reduce_max(nonzero_x_cols) - 1
        top_margin = tf.reduce_min(nonzero_y_rows)
        bot_margin = MNIST_IMG_SIZE - tf.reduce_max(nonzero_y_rows) - 1
        rand_dirs = tf.random.uniform([2])
        dir_idxs = tf.cast(tf.floor(rand_dirs * 2), tf.int32)
        rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
        x_amts = [tf.floor(-1.0 * rand_amts[0] *
                  tf.cast(left_margin, tf.float32)), tf.floor(rand_amts[0] *
                  tf.cast(1 + right_margin, tf.float32))]
        y_amts = [tf.floor(-1.0 * rand_amts[1] *
                  tf.cast(top_margin, tf.float32)), tf.floor(rand_amts[1] *
                  tf.cast(1 + bot_margin, tf.float32))]
        x_amt = tf.cast(tf.gather(x_amts, dir_idxs[1], axis=0), tf.int32)
        y_amt = tf.cast(tf.gather(y_amts, dir_idxs[0], axis=0), tf.int32)
        image = tf.reshape(image, [MNIST_IMG_SIZE * MNIST_IMG_SIZE])
        image = tf.roll(image, y_amt * MNIST_IMG_SIZE, axis=0)
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
        image = tf.transpose(image)
        image = tf.reshape(image, [MNIST_IMG_SIZE * MNIST_IMG_SIZE])
        image = tf.roll(image, x_amt * MNIST_IMG_SIZE, axis=0)
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE])
        image = tf.transpose(image)
        image = tf.reshape(image, [MNIST_IMG_SIZE, MNIST_IMG_SIZE, 1])
        return image, label
    
    @staticmethod
    def image_squish_random(image, label):
        rand_amts = tf.minimum(tf.abs(tf.random.normal([2], 0, .33)), .9999)
        width_mod = tf.cast(tf.floor(
            (rand_amts[0] * (MNIST_IMG_SIZE / 4)) + 1), tf.int32)
        offset_mod = tf.cast(tf.floor(rand_amts[1] * 2.0), tf.int32)
        offset = (width_mod // 2) + offset_mod
        image = tf.image.resize(image,
            [MNIST_IMG_SIZE, MNIST_IMG_SIZE - width_mod],
            method=tf.image.ResizeMethod.LANCZOS3,
            preserve_aspect_ratio=False,
            antialias=True)
        image = tf.image.pad_to_bounding_box(
            image, 0, offset, MNIST_IMG_SIZE, MNIST_IMG_SIZE + offset_mod)
        image = tf.image.crop_to_bounding_box(
            image, 0, 0, MNIST_IMG_SIZE, MNIST_IMG_SIZE)
        return image, label
    
        
    @staticmethod
    def image_rotate_random_py_func(image, angle):
        rot_mat = cv2.getRotationMatrix2D(
            (MNIST_IMG_SIZE/2, MNIST_IMG_SIZE/2), angle, 1.0)
        rotated = cv2.warpAffine(image.numpy(), rot_mat, (MNIST_IMG_SIZE, MNIST_IMG_SIZE))
        return rotated.reshape(MNIST_IMG_SIZE,MNIST_IMG_SIZE,1)

    @staticmethod
    def image_rotate_random(image, label):
        rand_amts = tf.maximum(tf.minimum(tf.random.normal([2], 0, .33), .9999), -.9999)
        angle = rand_amts[0] * 30  # degrees
        new_image = tf.py_function(pipelineTrain.image_rotate_random_py_func,
            (image, angle), tf.float32)
        new_image = tf.cond(rand_amts[1] > 0, lambda: image, lambda: new_image)
        return new_image, label
    
    @staticmethod
    def image_erase_random(image, label):
        sess = tf.compat.v1.Session()
        with sess.as_default():
            rand_amts = tf.random.uniform([2])
            x = tf.cast(tf.floor(rand_amts[0]*19)+4, tf.int32)
            y = tf.cast(tf.floor(rand_amts[1]*19)+4, tf.int32)
            patch = tf.zeros([4, 4])
            mask = tf.pad(patch, [[x, MNIST_IMG_SIZE-x-4],
                [y, MNIST_IMG_SIZE-y-4]],
                mode='CONSTANT', constant_values=1)
            image = tf.multiply(image, tf.expand_dims(mask, -1))
            return image, label

    @staticmethod        
    def image_add_random(image, label):
        sess = tf.compat.v1.Session()
        with sess.as_default():
            rand_amts = tf.random.uniform([2])
            x = tf.cast(tf.floor(rand_amts[0]*19)+4, tf.int32)
            y = tf.cast(tf.floor(rand_amts[1]*19)+4, tf.int32)
            # image= image[x:x+2, y:y+1].assign(tf.random.uniform([2]))
            # image= image[x:x+1, y:y+2].assign(tf.random.uniform([2]))
            patch = tf.multiply(tf.subtract(tf.ones([1,2]),
                                tf.random.uniform([1,2])/3),tf.cast(tf.random.uniform([1], 
                        minval=0, maxval=2, dtype =tf.dtypes.int32), tf.dtypes.float32))
            mask = tf.pad(patch, [[x, MNIST_IMG_SIZE-x-1],[y, MNIST_IMG_SIZE-y-2]],
                                  mode='CONSTANT', constant_values=0)
            image = tf.add(image, tf.expand_dims(mask, -1))
            image = tf.minimum(image, 1)
            return image, label

    @staticmethod
    def image_add_line(image, label):
        sess = tf.compat.v1.Session()
        with sess.as_default():
            decider = tf.cast(tf.random.uniform([3], minval=0, maxval=2, dtype=tf.dtypes.int32)
                              ,tf.dtypes.float32)
            
            line1a = decider[0]*(tf.subtract(tf.ones(MNIST_IMG_SIZE,1),
                                            tf.random.uniform([MNIST_IMG_SIZE])/2))
            line1a = tf.reshape(line1a, [MNIST_IMG_SIZE,1,1])
            line1b = decider[1]*(tf.subtract(tf.ones(MNIST_IMG_SIZE,1),
                                            tf.random.uniform([MNIST_IMG_SIZE])/2))
            line1b = tf.reshape(line1b, [MNIST_IMG_SIZE,1,1])            
            
            line2 = decider[2]*(tf.subtract(tf.ones(MNIST_IMG_SIZE,1),
                                            tf.random.uniform([MNIST_IMG_SIZE])/2))
            line2 = tf.reshape(line2, [1,MNIST_IMG_SIZE,1])
            
            xSa = tf.random.uniform(shape =[1], minval=0, 
                            maxval=5,dtype=tf.dtypes.int32)[0]
            xSb = tf.random.uniform(shape =[1], minval=0, 
                            maxval=5,dtype=tf.dtypes.int32)[0]            
            yS = tf.random.uniform(shape =[1], minval=MNIST_IMG_SIZE-6, 
                            maxval=MNIST_IMG_SIZE-1,dtype=tf.dtypes.int32)[0]
            
            mask1a = tf.pad(line1a, [[0,0],[xSa, MNIST_IMG_SIZE-xSa-1], [0,0]],
                           mode='CONSTANT', constant_values=0)
            mask1b = tf.pad(line1b, [[0,0],[MNIST_IMG_SIZE-xSb-1, xSb], [0,0]],
                           mode='CONSTANT', constant_values=0)            
            mask2 = tf.pad(line2, [[MNIST_IMG_SIZE-yS-1, yS],[0,0], [0,0]],
                           mode='CONSTANT', constant_values=0) 
            image = tf.add(image, mask1a)
            image = tf.add(image, mask1b)
            image = tf.add(image, mask2)
            image = tf.minimum(image, 1)          
            return image, label
            
            
    def get_training_dataset(self, dataset, batch_size):
        dataset = dataset.shuffle(buffer_size=self.DataLength)
        dataset = dataset.map(self.image_read,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self.image_rotate_random,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self.image_shift_rand,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self.image_squish_random,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self.image_erase_random,
           num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.map(self.image_add_random,
            num_parallel_calls=PARALLEL_INPUT_CALLS)        
        dataset = dataset.map(self.image_add_line,
            num_parallel_calls=PARALLEL_INPUT_CALLS)  
        # dataset = dataset.map(self.image_center,
        #     num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(-1)
        return dataset
    
    def get_validation_dataset(self, dataset, batch_size):
        dataset = dataset.cache()
        dataset = dataset.map(self.image_read,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        # dataset = dataset.map(self.image_center,
        #     num_parallel_calls=PARALLEL_INPUT_CALLS)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(-1)
        return dataset

    def get_test_dataset(self, dataset, batch_size):
        # dataset = dataset.cache()
        dataset = dataset.map(self.image_read,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        # dataset = dataset.map(self.image_center,
        #     num_parallel_calls=PARALLEL_INPUT_CALLS)
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(-1)
        return dataset