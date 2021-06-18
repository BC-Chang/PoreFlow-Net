#  if executing in spyder, run:
#       %matplotlib inline
# to get LiveLossPlots in the console


"""
Import the libraries
"""
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import time
from itertools import repeat
from argparser import *

import keras
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.models import save_model
from livelossplot import PlotLossesKeras #liveloss plot has to be installed
from livelossplot.plot_losses import MatplotlibPlot
from networks.PF_net_4branches import * #our proposed 3D CNN
from networks.PF_net_1branch import *   # Adapted 3D CNN with 1 branch
from networks.PF_net_nbranches import *
import pore_utils #my library

K.set_epsilon(1e-2) #fixes crazy high numbers when using MAPE

from numpy.random import seed
#from tensorflow import set_random_seed
#import sys

"""
Main inputs
"""
num_gpus      = args.n_gpus       #number of graphic-processing units to train model
mem_fraction  = args.mem_frac   #fraction of GPU to use
use_generator = args.use_generator    #option to use data-generator instead of loading all the data to the RAM
#num_features  = 1
net_branches  = args.n_branches
num_filters   = args.n_filters      #number of conv filters in the first layer
batch_size    = args.batch_size  
epochs        = args.epochs
rnd_num       = args.rnd_num  #rnd seed to initialize the model
dir_data      = 'G:/My Drive/Documents/Research/Solid_Full_X/finney'  #location of the training data
#data_input    = sys.argv[1]
features      = args.features
print(args.name)

"""
Set the random number seeds
"""
seed(rnd_num)
#set_random_seed(rnd_num)
#os.environ['PYTHONHASHSEED'] = '0'


"""
Data options
"""
input_size        = args.input_size #lenght of the side of a training cube (pixels)
train_on_sets     = [21,25]#, 22, 23, 24, 25] #training sets to use (each int is a domain)
validation_split  = args.val_split    #splits the last x %
patience_training = args.patience_training #epochs before it stops training
total_samples     = args.total_samples #for data generator
use_customloss    = args.use_customloss


"""
This is where we select the data transform that will be applied to our features
    'minMax_2' sets the bounds of the distribution to [-1,1]

"""
data_transform  = 'minMax_2'

"""
Set a name for the model
"""
model_name = f'PoreFlow_minMax2_branches_{net_branches}_filters_{num_filters}_{rnd_num}'#_{data_input}'
pore_utils.create_dir( model_name ) #creates a folder to store the output

"""
Load Data for training
"""

if use_generator == False:

    """
    Loading the selected training Data: This gives us a dictionary 
    with all the inputs and outputs
    (i.e. 'vx' : vx, 'vy' :vy, 'vz' : vz, 'p' : pressure,
    'e_poreZ' : eDist in Z dirr, 'e_pore' : E_dist, 
    'e_total':E_dist pore and solid,'tof_L':tof_L,'tof_R':tof_R)
    
    if the user has its own data, this step can be skipped
    """
    train_set = pore_utils.load_data( sets = train_on_sets, features=features, path=dir_data,
                                      split=True, input_size = input_size, 
                                      overlap=0 )

    binary_mask = train_set['binary']  #binary mask for the custom loss
    # e_dist = train_set['e_pore']
    # e_dist[e_dist <= 0] = 0
    # e_dist[e_dist > 0] = 1
    
    
    """
    Now, we can select and transform our inputs. 
    The summary stats are saved in a file for later use
    """
    for count, feat in enumerate(features[2:]):
        feat_train, _         = pore_utils.transform( train_set[feat],
                                                      data_transform,
                                                      model_name,
                                                      fileName=f'{feat}_stats')
        
        if count == 0:
            X_train = np.expand_dims(feat_train,axis=4)
        
        else: 
            X_train = np.concatenate( (X_train,
                                       np.expand_dims(feat_train,axis=4)
                                       ), axis=4)
        
        del feat_train
    
    vz , vz_stats     = pore_utils.transform( train_set['vz'], 
                                             data_transform, 
                                             model_name, 
                                             fileName='Vz_trainStats')
    vz[vz==-1] = 0
    
    y_train =  np.expand_dims(vz,axis=4) 
    
    if X_train.ndim <= 4:
            X_train = np.expand_dims( X_train , axis=4 ) 
            
    if y_train.ndim <= 4:
            y_train = np.expand_dims( y_train , axis=4 ) 
     
    
    """
    Shuffles data
    """
    mask = np.arange( X_train.shape[0] ) 
    np.random.shuffle( mask ) #get a mask to shuffle data
    
    X_train = X_train[ mask ,:,:,:,: ]
    y_train = y_train[ mask ,:,:,:,: ]
    
    binary_mask = binary_mask[ mask ,:,:,: ]
    
    
else:
    ## Data-generator
    IDs = np.arange(0,total_samples)
    np.random.shuffle( IDs ) #get a mask to shuffle data
    val_split = int( total_samples*validation_split )
    train_IDs =IDs[:-val_split]
    val_IDs   =IDs[-val_split:]
    ###


"""
Callbacks and model internals
"""
# Mean absolute percentage error (MAPE), Mean absolute error (MAE)
metrics=['MAPE','MAE'] 

#Custom loss as described in the paper
if use_customloss == True:
    loss    = pore_utils.custom_loss #imports custom loss
    metrics = [keras.losses.MAPE_c]  #custom MAPE
    # concatenates the porosity mask. This way, tf has access to it during training
    y_train = np.concatenate( (np.expand_dims(binary_mask,4),
                                                    y_train),axis=4)
else:
    loss = keras.losses.mean_absolute_error
        


optimizer     = keras.optimizers.Adam() # the default LR does the job
plot_losses   = PlotLossesKeras(outputs=[MatplotlibPlot(figpath=f'savedModels/{model_name}/metrics.png')])    
nan_terminate = keras.callbacks.TerminateOnNaN()
early_stop    = keras.callbacks.EarlyStopping(monitor ='val_loss', min_delta=0, 
                                              patience=patience_training, 
                                              verbose=2, mode='auto', baseline=None,
                                              restore_best_weights=False)


# TF internals
#config        = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = mem_fraction 
#config.gpu_options.per_process_gpu_memory_fraction = 1 
#config.gpu_options.allow_growth = True
#set_session( tf.Session(config=config) )


csv_logger = keras.callbacks.CSVLogger("savedModels/%s/training_log.csv" % model_name)

#with tf.device('/cpu:0'):
# model = build_PF_net(   input_shape0  = ( None, None, None, 1 ), 
#                         input_shape1  = ( None, None, None, 1 ), 
#                         input_shape2  = ( None, None, None, 1 ), 
#                         input_shape3  = ( None, None, None, 1 ), 
#                         filters_1     = num_filters )
asdf
#%%
model = build_n_branch_PF_net(   net_branches, 
                                  list(repeat((None, None, None, 1), net_branches)),
                                  filters_1 = num_filters )

# model = build_1_branch_PF_net(   input_shape  = ( None, None, None, 1 ), 
#                                   filters_1     = num_filters )

model.summary()

if num_gpus > 1:
    model = keras.utils.multi_gpu_model(model,gpus=num_gpus)

model.compile( loss=loss, optimizer=optimizer, metrics=metrics[:] )


checkpoint = ModelCheckpoint('savedModels/%s/%s.h5' % (model_name,model_name), 
                             monitor='val_loss', verbose=1, save_best_only=True, 
                             mode='min',save_weights_only=False)

callbacks_list = [early_stop,checkpoint,plot_losses,csv_logger]

"""
Train the model
"""
start_time = time.time()

if use_generator == True:
    data_name  = '21-26_minMax2_etrain_tofLR_misZF_eZ'
    dir_loc = "D:/SPLBM_output/chunks/" + data_name
    training_generator   = pore_utils.DataGenerator(dir_loc, train_IDs, 
                                                    branches=net_branches, 
                                                    batch_size=batch_size)
    validation_generator = pore_utils.DataGenerator(dir_loc,   val_IDs, 
                                                    branches=net_branches, 
                                                    batch_size=batch_size)
    
    hist_model = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    callbacks=callbacks_list,
                    epochs = epochs,
                    use_multiprocessing=False,
                    #max_queue_size = 1,
                    workers=8)
    
else:
    
    x0 = X_train[:,:,:,:,0]; x0 = x0[:,:,:,:,np.newaxis]
    X = [x0]
    if net_branches > 1:
        x1 = X_train[:,:,:,:,1]; x1 = x1[:,:,:,:,np.newaxis]
        X.append(x1)
    if net_branches > 2:
        x2 = X_train[:,:,:,:,2]; x2 = x2[:,:,:,:,np.newaxis]
        X.append(x2)
    if net_branches > 3:
        x3 = X_train[:,:,:,:,3]; x3 = x3[:,:,:,:,np.newaxis]
        X.append(x3)
    if net_branches > 4:
        x4 = X_train[:,:,:,:,4]; x4 = x4[:,:,:,:,np.newaxis]
        X.append(x4)
    if net_branches > 5:
        x5 = X_train[:,:,:,:,5]; x5 = x5[:,:,:,:,np.newaxis]
        X.append(x5)
        
    hist_model = model.fit( x=X, y=y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=validation_split, verbose=2, 
                            callbacks=callbacks_list, shuffle=True )

elapsed_time = time.time() - start_time
print('Training time [hrs]: ', elapsed_time/3600)

np.savetxt(("savedModels/%s/training_time.txt" % model_name),
           (np.expand_dims(elapsed_time/3600,0),np.expand_dims(elapsed_time,0)),
           delimiter=",", header="t [hrs], t[s]")

"""
Convert to single GPU (does not seem to be necessary)
"""
#best_model = load_model('savedModels/%s/%s.h5' % (model_name,model_name)) #load the best model
#best_model = best_model.layers[-2]
#best_model.save('savedModels/%s/%s_singleGPU.h5' % (model_name,model_name))


#del model
#K.clear_session()
#import gc
#gc.collect()
