import numpy as np
import pore_utils #my library
from matplotlib import pyplot as plt
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import scipy
from scipy import stats 
import matplotlib.cm as cm
import keras.backend as K
import os
import pandas as pd



K.set_learning_phase(0)


model_name = 'PoreFlow_minMax2_branches_1_filters_10_42721_edist_detrendedpot2'
dir_data   = 'G:/My Drive/Documents/Research/Solid_Full_X/SS'  #location of the data



best_model  = keras.models.load_model('savedModels/%s/%s.h5' % (model_name,model_name)) #load the best model

# #if the custom loss is utilized
#best_model = keras.models.load_model('savedModels/%s/%s.h5' % (model_name,model_name), 
#                                     custom_objects={'mean_absolute_percentage_error_custom': 
#                                         pore_utils.mean_absolute_percentage_error_custom})

input_size = 80

data_transform_pore     = 'minMax_2'
data_transform_tof      = 'minMax_2'
data_transform_vel      = 'minMax_2'
data_transform_MIS      = 'minMax_2'
data_transform_perm     = 'minMax_2'


samples = [11]



for ii in range( 0, np.size(samples) ):
    
    test_on = [samples[ii]] 
    
    print('-'*10)
    print(f'Sample: {test_on}')
    print('-'*10)
    

    test_set  = pore_utils.load_data(sets = test_on, path=dir_data, split=True, 
                                     input_size = input_size, overlap=0 )


    e_test,  _       = pore_utils.transform( test_set['e_pore'],   data_transform_pore,  model_name, fileName='e_stats',       isTraining=False )
    #MIS_z_test, _    = pore_utils.transform( test_set['mis_z'],    data_transform_MIS,   model_name, fileName='mis_z_stats',   isTraining=False )
    #MIS_f_test, _    = pore_utils.transform( test_set['mis_f'],    data_transform_MIS,   model_name, fileName='mis_f_stats',   isTraining=False )
    #tof_L_test,  _   = pore_utils.transform( test_set['tof_L'],    data_transform_tof,   model_name, fileName='tof_L_stats',   isTraining=False )
    #tof_R_test,  _   = pore_utils.transform( test_set['tof_R'],    data_transform_tof,   model_name, fileName='tof_R_stats',   isTraining=False )
    vel_t_true , _   = pore_utils.transform( test_set['vz'],       data_transform_vel,   model_name, fileName='Vz_trainStats', isTraining=False )
    #eZ_test , _      = pore_utils.transform( test_set['e_poreZ'],  data_transform_vel,   model_name, fileName='eZ_stats',      isTraining=False )
    vel_t_true[vel_t_true==-1] = 0
    #X_test = np.concatenate( ( 
    #                            np.expand_dims(e_test,      axis=4), 
    #                            np.expand_dims(tof_L_test,  axis=4), 
    #                            np.expand_dims(tof_R_test,  axis=4),
    #                            np.expand_dims(MIS_z_test,  axis=4),
                                #np.expand_dims(MIS_f_test,  axis=4),
                                #np.expand_dims(eZ_test,     axis=4), 
    #                            ),                          axis=4)
    X_test = np.expand_dims(e_test,axis=4)
    #del e_test, MIS_z_test, MIS_f_test, tof_L_test, tof_R_test
    del e_test
    if X_test.ndim <= 4:
            X_test  = np.expand_dims( X_test  , axis=0 ) 

         
    x1=np.expand_dims( X_test[:,:,:,:,0], axis=4)
    #x2=np.expand_dims( X_test[:,:,:,:,1], axis=4)
    #x3=np.expand_dims( X_test[:,:,:,:,2], axis=4)
    #x4=np.expand_dims( X_test[:,:,:,:,3], axis=4)
    #x5=np.expand_dims( X_test[:,:,:,:,4], axis=4)
    #x6=np.expand_dims( X_test[:,:,:,:,5], axis=4)
     
    vel_t_pred  = np.float64( np.squeeze(best_model.predict( x=[x1,
                                                                #x2,
                                                                #x3,
                                                                #x4,
                                                                #x5,
                                                                #x6,
                                                                ],batch_size=5 )) )#make prediction in batches
    
    
    ########### Mean velocity calc
    vel_t_pred_mean = vel_t_pred.mean()
    vel_t_true_mean = vel_t_true.mean()

    ########### Perm error
    kt_error = np.abs( (vel_t_true_mean-vel_t_pred_mean)/vel_t_true_mean )*100
    print(f'The error in electric potential is {kt_error:0.4f} %')
    

    vel_t_pred_full  = pore_utils.unsplit_matrix( vel_t_pred )
    vel_t_true_full  = pore_utils.unsplit_matrix( vel_t_true )

#%%
    """
    Plotting cross-sections
    """
    
    slice_true = vel_t_true_full[:,:,250]
    slice_pred = vel_t_pred_full[:,:,250]
   
    max_v = slice_true.max()
    try:
        min_v = slice_true.min()   
    except:
        min_v = -.1
            
    
    fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(20,4) )
    im=axs[0].imshow(slice_true, cmap=plt.cm.inferno)
    axs[0].set_title('Direct simulation results')
    fig.colorbar(im,ax=axs[0])
    axs[0].axis('off')
     
    
    im=axs[1].imshow(slice_pred, clim=(min_v, max_v), cmap=plt.cm.inferno)
    fig.colorbar(im,ax=axs[1])
    axs[1].axis('off')
    axs[1].set_title('PoreFlow-net predictions')
      
    # im=axs[2].imshow((np.abs((slice_true-slice_pred))),#clim=(-10,50),
    #                         cmap=plt.cm.inferno)#,norm=LogNorm(1,100)
    im=axs[2].imshow((np.abs((slice_true-slice_pred))/np.abs(slice_true)),
                     clim=(0,1),
                            cmap=plt.cm.inferno)#,norm=LogNorm(1,100)
    fig.colorbar(im,ax=axs[2])
    axs[2].set_title('Absolute error')
    axs[2].axis('off')


    #fig.savefig(f'savedModels/{model_name}/{samples[ii]}_prediction.png', dpi=600, transparent=True)
#%%
# import matplotlib as mpl
# slice_true = vel_t_true_full[250,:,:]
# slice_pred = vel_t_pred_full[250,:,:]
# plt.figure()
# plt.imshow(slice_pred, clim=(0.25,0.5), cmap='turbo')
# cbar = plt.colorbar()
# plt.title('PoreFlow-Net Predictions')
# #%%
# plt.figure()
# #slice_true[slice_true==-1] = np.nan
# plt.imshow(slice_true, cmap='turbo')
# plt.colorbar()
# plt.title('Direct Simulation Results')
# #%%
# plt.figure()
# # slice_true[slice_true==-1] = np.nan
# # slice_pred[slice_pred==-1] = np.nan
# plt.imshow(np.abs(slice_true-slice_pred), cmap='inferno')
# plt.colorbar()
# plt.title('Absolute Error')

#%%
    # results_dir = 'G:/My Drive/Documents/Research/Solid_Full_X/Results/Detrended_TOFL'
    # bin_test = vel_t_true_full.copy()
    # bin_test[bin_test != 0] = 1
    # bin_test = bin_test.astype('int8', copy=False )
    # vel_t_pred_full.tofile(f'{results_dir}/{samples[ii]}_prediction.raw')
    # bin_test.tofile(f'{results_dir}/{samples[ii]}_test_binary.raw')
