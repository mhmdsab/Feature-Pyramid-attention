# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:34:47 2019

@author: MSabry
"""

import tensorflow as tf


def conv2d(layer, ksize, in_depth, out_depth, padding = 'SAME', strides = [1,1,1,1], bias_add = True):
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    
    kernel = tf.Variable(conv_initializer(shape = [ksize, ksize, in_depth, out_depth]))
    conv = tf.nn.conv2d(layer, filter = kernel, strides = strides, padding = padding)
    if bias_add:
        bias = tf.Variable(tf.zeros(shape = [out_depth]))
        conv = tf.nn.bias_add(conv, bias)
    return conv
    


def Batch_Norm(x, is_training, decay = 0.9, scale = True, zero_debias = False):
    return tf.contrib.layers.batch_norm(x,
                                        decay = decay, 
                                        scale = scale, 
                                        is_training = is_training, 
                                        zero_debias_moving_mean = zero_debias) 



def convolve(layer, ksize, in_depth, out_depth, padding = 'SAME', strides = [1,1,1,1], is_training = True, 
             decay = 0.9, scale = True, zero_debias = False, batch_normalize = True, do_relu = True, bias_add = True):
    
    convolved = conv2d(layer = layer, ksize = ksize, in_depth = in_depth, 
                       out_depth = out_depth, padding = padding, strides = strides, bias_add = bias_add)
    
    if batch_normalize:
        convolved = Batch_Norm(x = convolved, is_training = is_training, decay = decay, scale = scale, zero_debias = zero_debias)
    
    if do_relu:
        convolved = tf.nn.relu(convolved)
    
    return convolved





def convolve_T(layer, ksize, in_depth, out_depth, output_shape, strides = [1,2,2,1], padding = 'VALID'):
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
    kernel = tf.Variable(conv_initializer(shape = [ksize, ksize, out_depth, in_depth]))
    return tf.nn.conv2d_transpose(layer, filter = kernel, strides = strides, 
                                  output_shape = output_shape, padding = padding)


def MaxPool2d(layer, ksize = [1,2,2,1] ,strides = [1,2,2,1], padding = 'VALID'):
    return tf.nn.max_pool(layer, ksize = ksize, strides = strides, padding = padding)


#def residual_block(input_layer):
#    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
#    
#    conv3_1 = convolve(layer = input_layer, kernel = kernel1)
#    conv3_1 = tf.nn.relu(conv3_1)
#    conv3_2 = convolve(layer = conv3_1, kernel = kernel2)
#    conv3_2 = tf.nn.relu(conv3_2)
#    
#    scaled_conv3_2 = conv3_2 * 0.5
#    scaled_input = input_layer * 0.5
#    
#    added = tf.add(scaled_input, scaled_conv3_2)
#    relued = tf.nn.relu(added)
#    
#    return relued
    
