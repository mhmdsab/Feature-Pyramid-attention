# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:29:24 2019

@author: MSabry
"""

import tensorflow as tf
from utils_fpa import convolve, MaxPool2d, convolve_T

class Feature_Pyramid_Attention:
    
    def __init__(self, layer):
        self.layer = layer
        self.layer_shape = self.layer.get_shape().as_list()
              
        
    def downsample(self):
                
        max_pool_1 = MaxPool2d(layer = self.layer)
        conv7_1 = convolve(layer = max_pool_1, ksize = 7, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1])
        conv7_2 = convolve(layer = conv7_1, ksize = 7, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1])
        
        max_pool_2 = MaxPool2d(layer = conv7_1)
        conv5_1 = convolve(layer = max_pool_2, ksize = 5, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1])
        conv5_2 = convolve(layer = conv5_1, ksize = 5, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1])
        
        max_pool_3 = MaxPool2d(layer = conv5_1)
        conv3_1 = convolve(layer = max_pool_3, ksize = 3, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1])
        conv3_2 = convolve(layer = conv3_1, ksize = 3, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1])
        
        upsampled_8 = convolve_T(layer = conv3_2, ksize = 2, in_depth = self.layer_shape[-1], out_depth = self.layer_shape[-1], 
                                 output_shape = conv5_2.get_shape().as_list())
        added_1 = tf.add(upsampled_8, conv5_2)
        upsampled_16 = convolve_T(layer = added_1, ksize = 2, in_depth = self.layer_shape[-1], out_depth = self.layer_shape[-1], 
                                 output_shape = conv7_2.get_shape().as_list())
        added_2 = tf.add(upsampled_16, conv7_2)
        upsampled_32 = convolve_T(layer = added_2, ksize = 2, in_depth = self.layer_shape[-1], out_depth = self.layer_shape[-1], 
                                 output_shape = self.layer.get_shape().as_list())
        
        return upsampled_32
    
    
    def direct_branch(self):
        
        conv1_1 = convolve(layer = self.layer, ksize = 1, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1], padding = 'VALID')    
        return conv1_1
        
#
    def global_pooling_branch(self):
        
        global_pool = tf.nn.avg_pool(self.layer, ksize = [1,self.layer_shape[1],self.layer_shape[1],1],
                                     strides = [1, 1, 1, 1], padding = "VALID")
        
        conv1_2 = convolve(layer = global_pool, ksize = 1, in_depth = self.layer_shape[-1], 
                           out_depth = self.layer_shape[-1], padding = 'VALID') 
        
        upsampled = convolve_T(layer = conv1_2, ksize = self.layer_shape[1], strides = [1,1,1,1], 
                               in_depth = self.layer_shape[-1], out_depth = self.layer_shape[-1], 
                               output_shape = self.layer_shape, padding = 'VALID')
#        
        return upsampled
    
    def FPA(self):
        down_up_conved = self.downsample()
        direct_conved = self.direct_branch()
        gpb = self.global_pooling_branch()
        
        multiplied = tf.multiply(down_up_conved, direct_conved)
        added_fpa = tf.add(multiplied, gpb)
        
        return added_fpa
        