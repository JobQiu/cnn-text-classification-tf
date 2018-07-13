#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:01:24 2018

@author: xavier.qiu
"""

import numpy as np
import tensorflow as tf

class TextCNN(object):
    
    def __init__(self,
                 vocab_size,
                 filter_sizes,
                 num_classes,
                 num_filters,
                 sequence_length,
                 embedding_size,
                 l2_reg_lambda)
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x" ) # type, shape, name
        self.input_y = tf.placeholder(tf.float32, [None, num_classes]) # type, shape, name
        pass