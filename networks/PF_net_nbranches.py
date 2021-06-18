# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:56:58 2021

@author: bchan
"""

from keras.models import *
from keras.layers import Input, Conv3D, UpSampling3D, BatchNormalization, \
                         Activation, add, concatenate
from networks.PF_net_4branches import res_block, encoder, decoder


def build_n_branch_PF_net(num_branches, input_shapes, filters_1=5):
    
    # Create the placeholders for the inputs
    inputs = [ Input( shape=input_shape ) for input_shape in input_shapes ]
    
    # Create the encoder branches
    branches = [ encoder( input_n, filters_1 ) for input_n in inputs ]
    
    # Concatenate the residual units of encoder branches for the skip connections
    if num_branches > 1:
        branch_sum_n = []
        for j in range(3):
            branches2concat = [branch[j] for branch in branches]
            branch_sum = [concatenate( branches2concat, axis=4 )]
            branch_sum_n.append(branch_sum)
    else:
        branch_sum_n = branches[0]
        
    # Create bridge between encoder and decoder
    path = res_block(branch_sum_n[2][0], [filters_1*8, filters_1*8, filters_1*8], 
                      [(2, 2, 2), (1, 1, 1)])
    
    # Create decoder branch
    path = decoder(path, branch_sum_n[0][0], branch_sum_n[1][0], branch_sum_n[2][0], filters_1)
    
    # Last filter, this outputs the velocity in Z-direction
    # for pressure or the full velocity tensor, one could change the 
    # number of filters to > 1
    path = Conv3D(filters=1, kernel_size=(1,1,1), activation='selu')(path)
    
    return Model(inputs=inputs, outputs=path)
    