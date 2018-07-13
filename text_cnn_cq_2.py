#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:57:19 2018

@author: xavier.qiu
"""

import numpy as np
import tensorflow as tf

class TextCNN(object):
    def __init__(self,
                 sequence_length,
                 embedding_size,
                 vocab_size,
                 num_classes,
                 num_filters,
                 filter_sizes,
                 l2_reg_lambda
                 ):
        
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") # type, shape, name
        self.input_y = tf.placeholder(tf.float32,[None, num_classes]   , name="input_y") # type, shape, name
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # type, shape, name
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.Variable(tf.constant( 0.0 ))
        
       
        # embedding layer
        with tf.name_scope("embedding"), tf.device("/cpu:0"): 
            self.W = tf.Variable(tf.random_normal([sequence_length, embedding_size], -1.0,1.0) ,name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, axis=-1)
        
        # conv-maxpool-relu layer
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-relu-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
                b = tf.Variable(tf.constant(0.1, [num_filters]) ,name="b")
                conv =  tf.nn.conv2d(self.input_x, W, stride=[1,1,1,1], padding="VALID", name="conv")
                
                h = tf.nn.relu(tf.nn.bias_add(conv,b ), name="relu")
                # pool
                h_pooled = tf.nn.maxpool(h,
                                         ksize=[1,sequence_length - filter_size+1,1,1],
                                         stride = [1,1,1,1],
                                         padding = "VALID",
                                         name="pool")
                pooled_outputs.append(h_pooled)
                
        
        #. how to debug tensorflow?
        # outputs, scores and predictions
        
        # loss
        
        # accuracy
        
        
        pass
    pass