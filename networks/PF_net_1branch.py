# -*- coding: utf-8 -*-
"""
Modification of PF_net_4branches using 1 branch
"""

from keras.models import *
from keras.layers import Input, Conv3D, UpSampling3D, BatchNormalization, \
                         Activation, add, concatenate
from networks.PF_net_4branches import *


def build_1_branch_PF_net(input_shape, filters_1=5):
    """
    Creates the PoreFlow-Net
    - input_shapeX is an array of 4-dims (i.e. (None,None,None,1))
    - filters_1 is an integer with the number of filters in the first layer
      this is doubled after each residual unit
    """
    
    # Create the placeholders for the inputs
    inputs0 = Input( shape=input_shape )
    
    # Create the encoder branches 
    branch0 = encoder( inputs0, filters_1 )
  
    # Create bridge between encoder and decoder
    path = res_block(branch0[2], [filters_1*8, filters_1*8, filters_1*8], 
                     [(2, 2, 2), (1, 1, 1)])

    # Create decoder branch
    path = decoder(path, branch0[0], branch0[1], branch0[2], filters_1)

    # Last filter, this outputs the velocity in Z-direction
    # for pressure or the full velocity tensor, one could change the 
    # number of filters to > 1
    path = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='selu')(path)

    return Model(inputs=[inputs0], outputs=path)
