#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:23:58 2018

@author: xavier.qiu
"""

import tensorflow as tf
import numpy as np


class TextCNN(object):
    
    def __init__(self, 
                 sequence_length, # the length of the sequence
                 num_classes, # the number of classes, here just two, pos and neg
                 num_filters, # number of each kind of filter
                 filter_sizes, # for example[3,4,5]
                 embedding_size, # for example, 300 
                 vocab_size, # this model didn't use a pre-trained word vector, so it need to initialize a word vector matrix weigth to train
                 l2_reg_lambda=0.0 ):
        
        glove_location= "/Users/xavier.qiu/Documents/GitHub/keras_nlp/glove.6B/glove.6B.100d.txt"

        # Placeholders for input, output and dropout
        # input_x store the indices of those words
        self.input_x = tf.placeholder(tf.int32,   [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes]    , name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        
        
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.Variable(tf.constant( 0.0 ))
        
        # Embedding layer
        with tf.name_scope("embedding"), tf.device("/cpu:0"): 
            def loadGloVe(filename):
                vocab = []
                embd = []
                file = open(filename,'r')
                for line in file.readlines():
                    row = line.strip().split(' ')
                    vocab.append(row[0])
                    embd.append(row[1:])
                print('Loaded GloVe!')
                file.close()
                return vocab,embd
            vocab,embd = loadGloVe(glove_location)
            vocab_size = len(vocab)
            embedding_dim = len(embd[0])
            embedding = np.asarray(embd)
            embedding_size = embedding_dim
            
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name="W")
            embedding_init = W.assign(embedding)
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # before expand dim, the shape should be n * sq_len * embedding_size 
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # after expand, it will have 4 dimension, the last one is for filter size
        
        pooled_outputs = []
        # Create a convolution + maxpool layer for each filter size
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size): 
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1) ,name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]) ,name="b")
                
                # Convolution Layer
                conv = tf.nn.conv2d(
                            self.embedded_chars_expanded,
                            W, 
                            strides=[1,1,1,1],
                            padding="VALID",
                            name="conv"
                        )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                        h,
                        ksize = [1, sequence_length- filter_size+1,1,1],
                        strides=[1,1,1,1],
                        padding='VALID',
                        name="pool"
                        )
                pooled_outputs.append(pooled)
                
        # Combine all the pooled features
        num_filter_total = len(filter_sizes) * num_filters
        self.h_pooled = tf.concat(pooled_outputs,3)
        self.h_pooled_flat = tf.reshape(self.h_pooled,[-1,num_filter_total])
       
        # Add dropout
        with tf.name_scope("dropout"): 
            self.h_drop = tf.nn.dropout(self.h_pooled_flat, self.dropout_keep_prob)
       
        # Final (unnormalized) scores and predictions
        with tf.name_scope("outputs"): 
            W = tf.get_variable("W",
                                [num_filter_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer()
                                )
            b = tf.Variable(tf.constant( 0.1,shape=[num_classes]) ,name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
            self.predictions = tf.argmax(self.scores,1,name="predictions")
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"): 
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(loss) + l2_loss * l2_reg_lambda
            pass
        # Accuracy
        with tf.name_scope("accuracy"): 
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            pass
        pass
    