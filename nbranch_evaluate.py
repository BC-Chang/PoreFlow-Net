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
import warnings

warnings.filterwarnings('ignore')

K.set_learning_phase(0)


model_name = 'PoreFlow_minMax2_branches_2_filters_20_61521_edist_tofl1'
parent_dir_data   = 'G:/My Drive/Documents/Research/Solid_Full_X'  #location of the data
sample_directories = ['finney-raw', 'ss-raw']
features = ['e_pore', 'tof_L', 'linear']
num_features = 2

best_model  = keras.models.load_model('syncModels/%s/%s.h5' % (model_name,model_name)) #load the best model

input_size = 80
data_transform  = 'minMax_2'

# samples = [21, 22, 23, 24, 25]
for sample_dir in sample_directories:
    dir_data = f'{parent_dir_data}/{sample_dir}'
    if sample_dir == 'finney-raw':
        samples = [21, 22, 23, 24, 25]
    elif sample_dir == 'ss-raw':
        samples = [10, 11, 12, 13, 14, 15]
    for ii in range( 0, np.size(samples) ):
        
        test_on = [samples[ii]] 
        
        print('-'*10)
        print(f'Sample: {test_on}')
        print('-'*10)
        
        
        test_set, data_shape  = pore_utils.load_data(sets = test_on, features=features, path=dir_data, split=True, 
                                         input_size = input_size, overlap=0 )
        
        for count, feat in enumerate(features[2:-1]):
            feat_test, _ = pore_utils.transform( test_set[f'{feat}'], data_transform, model_name, fileName=f'{feat}_stats', isTraining=False)
            
            if count == 0:
                X_test = np.expand_dims(feat_test, axis=4)
            else:
                X_test = np.concatenate( (X_test,
                                          np.expand_dims(feat_test, axis=4)),
                                            axis=4)    
            del feat_test
    
        vel_t_true , _   = pore_utils.transform( test_set['vz'],       data_transform,   model_name, fileName='Vz_trainStats', isTraining=False )
        vel_t_true[vel_t_true == -1] = 0
        
        if X_test.ndim <= 4:
                X_test  = np.expand_dims( X_test  , axis=0 ) 
    
        xn = [np.expand_dims( X_test[:,:,:,:,i], axis=4) for i in range(num_features)]     
    
        vel_t_pred  = np.float64( np.squeeze(best_model.predict( x=xn, batch_size=5 )) )#make prediction in batches
        
        vel_t_pred = vel_t_pred + test_set['linear']
        vel_t_true = vel_t_true + test_set['linear']
        
        ########### Mean velocity calc
        vel_t_pred_mean = vel_t_pred.mean()
        vel_t_true_mean = vel_t_true.mean()
    
        ########### Perm error
        # kt_error = np.abs( (vel_t_true_mean-vel_t_pred_mean)/vel_t_true_mean )*100
        # print(f'The permeability error is {kt_error:0.4f} %')
        
    
        vel_t_pred_full  = pore_utils.unsplit_matrix( vel_t_pred )
        vel_t_true_full  = pore_utils.unsplit_matrix( vel_t_true )
        
        vel_t_pred_full  = pore_utils.crop_matrix( vel_t_pred_full, data_shape )
        vel_t_true_full  = pore_utils.crop_matrix( vel_t_true_full, data_shape )
    
        vel_t_true_full[vel_t_true_full < 0] = 0
        vel_t_pred_full[vel_t_true_full == 0] = 0
        rel_error_full = np.abs((vel_t_true_full-vel_t_pred_full))/np.abs(vel_t_true_full)
        mean_error = np.mean(rel_error_full[~np.isnan(rel_error_full)])
        del test_set
        
        # Plot stuff
        """
        Plotting cross-sections
        """
        
        slice_true = vel_t_true_full[0,:,:]
        slice_pred = vel_t_pred_full[0,:,:]
       
        max_v = slice_true.max()
        try:
            min_v = slice_true[slice_true > 0].min()   
        except:
            min_v = -.1
                
        
        fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(20,4) )
        im=axs[0].imshow(slice_true, clim=(min_v, max_v), cmap=plt.cm.inferno)
        axs[0].set_title('Direct Simulation Results')
        fig.colorbar(im,ax=axs[0])
        axs[0].axis('off')
         
        
        im=axs[1].imshow(slice_pred, clim=(min_v, max_v), cmap=plt.cm.inferno)
        fig.colorbar(im,ax=axs[1])
        axs[1].axis('off')
        axs[1].set_title('PoreFlow-net Prediction')
          
        # im=axs[2].imshow((np.abs((slice_true-slice_pred))),#clim=(-10,50),
        #                         cmap=plt.cm.inferno)#,norm=LogNorm(1,100)
        rel_error = np.abs((slice_true-slice_pred))/np.abs(slice_true)
        rel_error[np.isnan(rel_error)] = 0
        im=axs[2].imshow(rel_error,
                                 clim=(0,0.2), cmap=plt.cm.inferno)#,norm=LogNorm(1,100)
        fig.colorbar(im,ax=axs[2])
        
    
        axs[2].set_title(f'Relative Error, mean error= {mean_error*100:0.1f}%')
        axs[2].axis('off')
        
        plt.savefig(f'syncModels/%s/prediction_images/Dataset_{samples[ii]}.png' % (model_name))
        
        # print(f'Mean relative error in pore space: {mean_error*100:0.1f}%')

    
 #%%   
    # Calculate Conductivity
    # img_shape = vel_t_true_full.shape
    # p = {'nx': img_shape[0],
    #   'ny': img_shape[1],
    #   'nz': img_shape[2],
    #   'dz': 1}
    # p['nxy'] = (p['nx'])*(p['ny'])
    
    # true_elec = np.transpose(vel_t_true_full, (2,0,1)).ravel()
    # pred_elec = np.transpose(vel_t_pred_full, (2,0,1)).ravel()
    # cond = np.copy(true_elec).ravel()
    # cond[cond != 0] = 1
    
    # true_currZ, true_currZmean, true_currZstd = pore_utils.calc_elec_currz(cond, true_elec, p)
    # pred_currZ, pred_currZmean, pred_currZstd = pore_utils.calc_elec_currz(cond, pred_elec, p)
    
    # true_currZ = true_currZ.reshape((p['nz'], p['ny'], p['nx']))
    # true_currZ_slice = np.sum(true_currZ, axis=(1,2))
    # print(f'Mean Curr. Z (simulation) = {true_currZmean}')
    # print(f'Std. Curr. Z (simulation) = {true_currZstd} ({100*true_currZstd/true_currZmean})')
    # print('-'*10)

    # z_plot = np.arange(0,p['nz'], 1)
    # fig = plt.figure(figsize=(8,5), dpi=300)
    # plt.plot(z_plot, true_currZ_slice)

    
    # pred_currZ = pred_currZ.reshape((p['nz'], p['ny'], p['nx']))
    # pred_currZ_slice = np.sum(pred_currZ, axis=(1,2))
    # print(f'Mean Curr. Z (prediction) = {pred_currZmean}')
    # print(f'Std. Curr. Z (prediction) = {pred_currZstd} ({100*pred_currZstd/pred_currZmean})')
    # print('-'*10)

    # z_plot = np.arange(0,p['nz'], 1)
    # fig = plt.figure(figsize=(8,5), dpi=300)
    # plt.plot(z_plot, pred_currZ_slice)
    # plt.ylim([0,])
    # Calculate Mean Current in Z of Simulation Result
    # data_dir = 'G:/My Drive/Documents/Research/PoreFlow-Net/test_data/Eroded_SP_21'
    # dataset = [21]
    
    
    # pot = np.fromfile(f'{data_dir}/elecpot.raw', dtype='float32')
    # cond = np.fromfile(f'{data_dir}/elecconnect.raw', dtype='uint8')
    # #cond = -1*cond + 1
    
    # currZ, currZmean, currZstd = calc_elec_currz(cond,pot, p)

    
#%%

# true_pot = np.transpose(vel_t_true_full, (2,0,1))
# pred_pot = np.transpose(vel_t_pred_full, (2,0,1))

# # plt.figure(figsize=(8,5), dpi=300)
# # plt.imshow(true_pot[250,:,:])

# # plt.figure(figsize=(8,5), dpi=300)
# # plt.imshow(vel_t_true_full[250,:,:])

# plt.figure(figsize=(8,5), dpi=300)
# plt.imshow(pred_pot[250,:,:])
# plt.figure(figsize=(8,5), dpi=300)
# plt.imshow(vel_t_pred_full[250,:,:])

# vel_t_true_full.tofile('True21.raw')
# vel_t_pred_full.tofile('Prediction21.raw')

