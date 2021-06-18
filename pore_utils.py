import numpy as np
import os
import shutil # copying files
from hdf5storage import loadmat#load matrices
import keras
import keras.backend as K
import scipy
import itertools as it
import timeit


def create_dir(name):
    """
    Creates the folder to save the model and makes a copy of the python
    training script (this is useful after training 100+ models :)   ) 
    """

    if not os.path.exists('savedModels/%s' % name):
        os.mkdir('savedModels/%s' % name)
        print("Creating " , name ,  " directory ")
    else:    
        print("Directory: " , name ,  "Folder already exists!")
    
    try:
        shutil.copy2('train.py', ( 'savedModels/%s' % name )) #created a copy of training file
    except:
        print("File: train.py already exists! ")
    
    
def load_data( sets, features, path, split=False , input_size = np.nan, 
               overlap = np.nan ):
    
    """
    Loads the data (for train or test)
    
    The data was saved in matlab matrix format because of ... <- insert good reason here
    
    sets : tuple of integers indicating the domain number: 1 is a finneypack, etc
    feature: list of strings indicating desired features to load, do not include binary or vz
        feature options: ['e_pore', 'e_total', 'e_poreZ', 'tof_L', 'tof_R', 'mis_f', 'mis_z']    
    path: location of the training data
    split: bool indicating wheter to split the domains in subdomains
    input_size: subdomain size
    overlap: indicates if these subdomains should be sample with overlap
    """
    

    dom = {'vz': ['detrended_pot','detrended_dom'], 
           #'vz': ['elecpot', 'pot_domain'],
           'binary':['solid_full','domain'], 'linear':['linear_trend','linear_dom'],
           'e_pore':['euclidean_pore','e_domain'], 
           'e_total':['euclidean_total','e_full'],
           'e_poreZ':['euclidean_poreZ', 'e_z'],
           'poros':['detrended_porosimetry','pore_domain'],
           'tof_L':['ToF_l', 'tOf_L'],
           'tof_R':['ToF_r', 'tOf_R'],
           'mis_f':['MIS_full', 'MIS_3D'],
           'mis_z':['MIS_z_inlet', 'MIS_3D']}
    
    if not 'vz' in features: features.insert(0,'vz')
    if not 'binary' in features: features.insert(0,'binary'); 
    
    t_set = {}
    if np.isnan(overlap) == True:
        overlap = input_size/2 - 1
            
    for i in range( 0 , np.size( sets ) ) :
        load_set = sets[i]
        print('Loading set no. %d' % load_set)
        
        for feat in features:
            # feat_data = loadmat(f'{path}/{dom[feat][0]}_{load_set}')

            # p_feat_data = feat_data[dom[feat][1]].astype('float32')
            # print(p_feat_data.shape)
            if feat == 'binary': 
                data_type = 'uint8'
            else:
                data_type = 'float32'
                
            p_feat_data = np.fromfile(f'{path}/{dom[feat][0]}_{load_set}.raw', dtype=data_type)
            p_feat_data.astype('float32')
            

            data_shape = int(np.round(p_feat_data.shape[0] ** (1./3.)))
            p_feat_data = p_feat_data.reshape((data_shape, data_shape, data_shape))
            

            
            if feat == 'binary':
                phi = np.sum(p_feat_data < 1) / p_feat_data.size
                print(f'The porosity of this domain is {phi}')
                p_feat_data = calculate_weighted_mask(p_feat_data)
                
            if feat == features[-1]:
                data_shape = p_feat_data.shape
                print(data_shape)

            if split == True:
                data_tmp = split_matrix(p_feat_data, input_size, overlap)
                print(data_tmp.shape)
            else:
                data_tmp = p_feat_data
   
            if i == 0: 
                t_set[f'{feat}'] = data_tmp 
            else:
                t_set[f'{feat}'] = np.concatenate((t_set[f'{feat}'], data_tmp), axis=0)
            
    
    return t_set, [data_shape[0]-2, data_shape[1]-2, data_shape[2]-2]

def calculate_weighted_mask(solid_mask):
    
    """
    Calculates the porosity weighted mask
    """
    for i in range( 0,solid_mask.shape[2] ):
        porosity = 1 - np.sum(solid_mask[:,:,i])/np.size(solid_mask[:,:,i])
        solid_mask[:,:,i][ solid_mask[:,:,i] == 0 ] = 1/porosity
        solid_mask[:,:,i] = solid_mask[:,:,i]/np.sum(solid_mask[:,:,i])*np.size(solid_mask[:,:,i])
    return solid_mask
'''
def split_matrix(m, w_size,  erase_bcs=True):
    """
    Splits the 3D domain into smaller subdomains
    
    m: 3D domain
    w_size: size of the subsamples
    w_stride: stride lenght
    erase_bcs: bool. if true erases the boundary layers (to avoid noise)
    """
    
    if erase_bcs==True:
        m=np.delete(m,-1,0) #get rid of the boundaries
        m=np.delete(m,0 ,0)
        
        m=np.delete(m,-1,1)
        m=np.delete(m,0 ,1)
        
        m=np.delete(m,-1,2)
        m=np.delete(m,0 ,2)
        
    # Pad matrix with 0s to be divisible by w_size
    if m.shape[0] % w_size != 0:
        target_size = m.shape[0] + w_size - (m.shape[0] % w_size)
        m = np.pad(m, (target_size - m.shape[0])//2, constant_values=(0))
    
    p, q, r = m.shape
    
    m = m.reshape((-1, q//w_size, w_size, r//w_size, w_size)).transpose(1,3,0,2,4).reshape(-1,p, w_size, w_size)    
    
    return m 

'''
def split_matrix(m, w_size, w_stride=0, erase_bcs=True, pad=True):
    """
    Splits the 3D domain into smaller subdomains
    
    m: 3D domain
    w_size: size of the subsamples
    w_stride: stride lenght
    erase_bcs: bool. if true erases the boundary layers (to avoid noise)
    """
    
    w_stride=int(w_stride)
    
    if erase_bcs==True:
        m=np.delete(m,-1,0) #get rid of the boundaries
        m=np.delete(m,0 ,0)
        
        m=np.delete(m,-1,1)
        m=np.delete(m,0 ,1)
        
        m=np.delete(m,-1,2)
        m=np.delete(m,0 ,2)
    
        
    # Pad matrix with 0s to be divisible by w_size
    if m.shape[0] % w_size != 0:
        target_size = m.shape[0] + w_size - (m.shape[0] % w_size)
        m = np.pad(m, (target_size - m.shape[0])//2, constant_values=(0))

    p, q, r = m.shape
    
    mt = m.reshape((-1, q//w_size, w_size, r//w_size, w_size)).transpose(1,3,0,2,4).reshape((-1,w_size, w_size, w_size))         
    # # Indices to split matrix
    # sample_start=np.arange(0,m.shape[0],w_size)
    # sample_start=sample_start[sample_start<(m.shape[0]-w_size+1)]
    
    # # Indices to split matrix with overlap
    # sub_sample_start=sample_start+w_stride
    # sub_sample_start=sub_sample_start[sub_sample_start<(m.shape[0]-w_size+1)]
    
    # if w_stride == 0: #if no overlap is requested
    #     mt=np.zeros((sample_start.size**3,w_size,w_size,w_size))
    # else: #subsamples + overlap
    #     mt=np.zeros((sample_start.size**3+sub_sample_start.size**3,
    #                   w_size,w_size,w_size))
    
    # ii=0
    # for j,k,i in it.product(sample_start, repeat=3):
    #     mt[ii,:,:,:]=np.expand_dims(m[k:k+w_size,
    #                                   j:j+w_size,
    #                                   i:i+w_size],axis=0)
        
    #     ii=ii+1        
                 
    # if w_stride!=0:
    #     for i,j,k in it.product(sub_sample_start, repeat=3):
    #         mt[ii,:,:,:]=np.expand_dims(m[k:k+w_size, \
    #                                       j:j+w_size, \
    #                                       i:i+w_size],axis=0)
                            
    #         ii=ii+1

    return mt


def transform( x, tName, modelName, fileName='tmp', isTraining=True):
    """
    Performs the desired data transform
    x: array w/data
    tName: name of desired transformation
    modelName: name of the model (to save the summary stats)
    isTraining: bool. If true overwirtes existing file w/sum stats
    
    """
    
    
    if isTraining == True:
        x_stats = calculate_stats( x, modelName, fileName )
        x_mean    = x_stats['mean']
        x_min     = x_stats['min']
        x_max     = x_stats['max']
        x_maxAbs  = x_stats['maxAbs']
        x_minAbs  = x_stats['minAbs']
        x_std     = x_stats['std']
        x_range   = x_stats['range']
        x_p95     = x_stats['p95']
        x_new_min = x_stats['x_new_min']
        
    else:
        x_stats = np.loadtxt( 'syncModels/%s/%s.txt' % (modelName, fileName) , 
                             delimiter = ',' )
        
        x_mean    = x_stats[0]
        x_min     = x_stats[1] 
        x_range   = x_stats[2]
        x_std     = x_stats[3]
        x_max     = x_stats[4]
        x_maxAbs  = x_stats[5]
        x_minAbs  = x_stats[6]
        x_p95     = x_stats[7]
        x_new_min = x_stats[8]
    
    
    if tName == 'Constant':
        print( 'Dividing by 6e-6 Transform' )
        xt =  x/6e-6
    
    if tName == 'minMax_abs':
        print('minMax_abs')
        xt = (x-x_minAbs)/(x_maxAbs-x_minAbs)
    
    if tName == 'minMax_eps_2':
        print( 'minMax EPS 2 Transform' )
        xt = (   ( (x     - x_min)*(x_max - x_new_min)/x_range )/
                 ( (x_max - x_min)*(x_max - x_new_min)/x_range )   )*2-1    
    
    if tName == 'minMax':
        print( 'minMax Transform' )
        xt = ( x - x_min ) / x_range
        
    if tName == 'minMax_2':
        print( 'minMax 2 Transform' )
        xt = ( x - x_min )*2 / x_range - 1
        
    if tName == 'minMax_8':
        print( 'minMax 8 Transform' )
        xt = ( x - x_min )*8 / x_range - 2
        
    if tName == 'mMP95_2':
        print( 'mMP95 2 Transform' )
        xt = ( x )*2 / x_p95 - 1
        
    if tName == 'minMax_4':
        print( 'minMax 4 Transform' )
        xt = ( x - x_min )*4 / x_range - 2
        
    if tName == 'minMax_noZ':
        print( 'minMax Transform' )
        xt = ( x - x_min ) / x_range
        xt[ x==0 ] = 0
        
    if tName == 'normal':
        print('normal Transform')
        xt = ( x - x_mean ) / x_std
        #xt[ x==0 ] = 0
        
    if tName == 'normal_range2':
        print('normal Transform')
        xt = ( x - x_mean ) / x_std
        max_x = np.max( np.abs(xt) )
        print(max_x)
        xt = xt/max_x
        
    if tName == 'range':
        print('range Transform')
        xt = x / x_range
        
    if tName == 'max':
        print('max Transform')
        xt = x / x_maxAbs
        
    if tName == 'logCNN': 
        print('logCNN Transform')
        tmp  = np.abs(x) #absolute value
        tmp  = tmp / 3e-18 #divides by the min value
        tmp  = np.log10( tmp + 1 ) #plus one to eliminate the zeros
        xt   = tmp # this dist goes from 0 to ~13
        xt[ x<0 ] = ( -1 )*xt[ x<0 ] #adds the negative sign back
        xt = xt/13
        
    if tName == 'log_tmp': 
        print('logTMP Transform')
        x[ x==0 ]  = 1 #absolute value
        xt  = x/np.abs(x)*np.log10( np.abs(x) ) #plus one to eliminate the zeros

    if tName == 'log_tmp2': 
        print('logTMP2 Transform')
        x[ x==0 ]  = 1
        xt  = -x/np.abs(x)*np.log10( np.abs(x) ) - x/np.abs(x)*4.5
        xt[ np.abs(xt) == 4.5 ] = 0
        #xt = xt/6

    if tName == 'logCNN_test': 
        print('logCNN_test Transform')
        tmp  = np.abs(x) #absolute value
        tmp  = tmp / x_minAbs #divides by the min value
        tmp  = np.log10( tmp + 1 ) #plus one to eliminate the zeros
        xt   = tmp # this dist goes from 0 to ~13
        xt[ x<0 ] = ( -1 )*xt[ x<0 ] #adds the negative sign back
        xt = xt/13
                        
    if tName=='none':
        print('no Transform')
        xt = x
        
    summary_stats = { 'mean':x_mean, 'min':x_min, 'range':x_range, 'std':x_std, 
                'max':x_max, 'maxAbs':x_maxAbs, 'minAbs':x_minAbs }
    
    return xt , summary_stats

def calculate_stats( x, modelName, fileName ):
    
    x_mean  = x.mean()
    x_min   = x.min()
    x_max   = x.max()
    x_std   = x.std()
    x_range = x_max - x_min
    
    eps         = np.finfo(np.float32).eps
    x_new_min   = eps/(1/x_max) 
    
    x_maxAbs = np.max( np.abs(x) )
    x_minAbs = np.min( np.abs( x[ x>0 ] ) )
    
    x_p95 = np.percentile( x[x!=0], 95 )
    
    x_stats = { 'mean':x_mean, 'min':x_min, 'range':x_range, 'std':x_std, 
                'max':x_max, 'maxAbs':x_maxAbs, 'minAbs':x_minAbs, 'p95': x_p95,
                'x_new_min': x_new_min}
    
    np.savetxt( ('savedModels/%s/%s.txt' % (modelName, fileName) ),
               (x_mean, x_min, x_range, x_std, x_max, x_maxAbs, x_minAbs,
                x_p95, x_new_min), 
               delimiter=",", header="mean, min, range, std, max, maxAbs, minAbs, \
               P95, new_min")
    
    return x_stats

def custom_loss(y_true_weights, y_pred):
    
    weights = y_true_weights[:,:,:,:,0]
    y_true1 = y_true_weights[:,:,:,:,1]
    
    y_pred = K.squeeze(y_pred, axis = 4)
    
    y_true1 = y_true1*weights
    y_pred1 = y_pred*weights
    
    print(K.int_shape(y_pred1))
    print(K.int_shape(y_true1))
    
    return K.mean(K.square(y_pred1 - y_true1), axis=-1)

def mean_absolute_percentage_error_custom(y_true_weights, y_pred):
    
    y_true1 = y_true_weights[:,:,:,:,1]
    y_pred1 = K.squeeze(y_pred, axis = 4)
    diff    = K.abs( (y_true1 - y_pred1) / K.clip(K.abs(y_true1),
                                           K.epsilon(),
                                           None))
    return 100. * K.mean(diff, axis=-1)

def ijk2m(i, j, k, Nx, Ny):

    m = Nx*Ny*k + Nx*j + i
    
    return m

def flux_stats(currZ, p):
    
    ncurrZ = currZ.copy()

    ncurrZ = ncurrZ.reshape((p['nx'], p['ny'], p['nz']))
    fluxZ = np.sum(np.sum(ncurrZ,axis=1),axis=1)
    fluxZ[np.abs(fluxZ) > 150] = np.nan
    mean_flux = np.nanmean(fluxZ)
    std_flux = np.nanstd(fluxZ)
    
    return mean_flux, std_flux

def calc_elec_currz(cond, pot, p):
    
    y = np.zeros((p['nx']*p['ny']*p['nz']))
    EPS = 1e-1
    print('-'*10)
    print('Calculating current...')
    for k in range(p['nz']):
        print('Slice ', k)
        K = ijk2m(0,0,k,p['nx'],p['ny'])
        for j in range(p['ny']):
            for i in range(p['nx']):
                if(cond[K] < EPS):
                    y[K] = 0.
                elif k < p['nz']-1 and cond[K + p['nxy']]>EPS:
                    y[K] = -(pot[K+p['nxy']] - pot[K])*cond[K]/p['dz']
                elif k == p['nz']-1 and cond[K-p['nxy']] > EPS:
                    y[K] = -(pot[K] - pot[K- p['nxy']])*cond[K]/p['dz']
                else:
                    y[K] = 0.
                K+=1
                
    mean_currZ, std_currZ = flux_stats(y, p)

    return y, mean_currZ, std_currZ

def unsplit_matrix_overlapped(mt1, w_stride=39, method='Max'):
   
    mt = np.copy(mt1)
   
    if mt.shape[0] == 250:
        m_side      = 480
        m  = np.zeros((m_side,m_side,m_side))
        m1 = np.ones((m_side,m_side,m_side))
        m2 = np.ones((m_side,m_side,m_side))
       
    if mt.shape[0] == 341:
        m_side      = 500
        m  = np.zeros((m_side,m_side,m_side))
        m1 = np.ones((m_side,m_side,m_side))
        m2 = np.ones((m_side,m_side,m_side))
     
    if method =='Max':
        m1 = m1*(-np.inf)
        m2 = m2*(-np.inf)
    if method =='Min':
        m1 = m1*(np.inf)
        m2 = m2*(np.inf)
    if method =='Mean':
        m1 = m1*(np.nan)
        m2 = m2*(np.nan)
    if method =='Mid':
        mask = np.ones((40,40,40))
        mask = np.pad(mask, 20, 'constant')
       
        for i in range(0,mt.shape[0]):
            mt[i,:,:,:] = mask*mt[i,:,:,:]
           
   
    #mt[mt==0] = (-np.inf)
       
    sample_start=np.arange(0,m1.shape[0],mt.shape[1])
    sample_start=sample_start[sample_start<(m1.shape[0]-(mt.shape[1]+1))]
    sample_start = sample_start
   
   
    sub_sample_start = sample_start+w_stride
    sub_sample_start = sub_sample_start[sub_sample_start<(m1.shape[0]-(mt.shape[1]+1))]
    sub_sample_start = sub_sample_start
         
   
    m_side_cubes = sample_start.size
    ii=0
    for j in range(0,m_side_cubes):
        for k in range(0,m_side_cubes):
            for i in range(0,m_side_cubes):
                m1[sample_start[k]:sample_start[k]+mt.shape[1],
                   sample_start[j]:sample_start[j]+mt.shape[1],
                   sample_start[i]:sample_start[i]+mt.shape[1] ] = np.squeeze( mt[ii,:,:,:] )
                ii = ii+1
               
    m_side_cubes = sub_sample_start.size
    for i in range(0,m_side_cubes):
        for j in range(0,m_side_cubes):
            for k in range(0,m_side_cubes):
                m2[sub_sample_start[k]:sub_sample_start[k]+mt.shape[1],
                   sub_sample_start[j]:sub_sample_start[j]+mt.shape[1],
                   sub_sample_start[i]:sub_sample_start[i]+mt.shape[1] ] = np.squeeze( mt[ii,:,:,:] )
                ii = ii+1
   
   
    if method=='Max':
        m = np.maximum(m1,m2)
    if method=='Min':
        m = np.minimum(m1,m2)
    if method=='Mean':
        m1[np.isnan(m1)] = m2[np.isnan(m1)]
        m2[np.isnan(m2)] = m1[np.isnan(m2)]
        m1[np.isnan(m1)] = 0
        m2[np.isnan(m2)] = 0
        m = (m1+m2)/2.0
       
   
    if m_side == 500:
        mask = np.arange(480,500)
        m = np.delete(m,mask,axis=0)
        m = np.delete(m,mask,axis=1)
        m = np.delete(m,mask,axis=2)
       
    if m_side == 480:
        mask = np.arange(400,480)
        m = np.delete(m,mask,axis=0)
        m = np.delete(m,mask,axis=1)
        m = np.delete(m,mask,axis=2)
       
    return m


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_loc, list_IDs, branches, 
                 batch_size=5, dim=(80,80,80), 
                 n_channels_in=6,n_channels_out=1,shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.branches = branches
        
        self.file_loc = file_loc
        
       
        
        self.list_IDs = list_IDs
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        
        if self.branches == 0:
            pass
        if self.branches > 0:
            Xc = X
            x0 = Xc[:,:,:,:,0]; x0 = x0[:,:,:,:,np.newaxis]
            X = [x0]
        if self.branches > 1:
            x1 = Xc[:,:,:,:,1]; x1 = x1[:,:,:,:,np.newaxis]
            X.append(x1)
        if self.branches > 2:
            x2 = Xc[:,:,:,:,2]; x2 = x2[:,:,:,:,np.newaxis]
            X.append(x2)
        if self.branches > 3:
            x3 = Xc[:,:,:,:,3]; x3 = x3[:,:,:,:,np.newaxis]
            X.append(x3)
        if self.branches > 4:
            x4 = Xc[:,:,:,:,4]; x4 = x4[:,:,:,:,np.newaxis]
            X.append(x4)
        if self.branches > 5:
            x5 = Xc[:,:,:,:,5]; x5 = x5[:,:,:,:,np.newaxis]
            X.append(x5)
                
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels_in ))
        y = np.empty((self.batch_size, *self.dim, self.n_channels_out))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = X_full[ID,:,:,:,:]
         
            X[i,] = np.load((self.file_loc + '/X/' + str(ID)+'.npy'))
            
            y[i,] = np.load((self.file_loc + '/y/' + str(ID)+'.npy'))

        return X, y

def write_data_chunks(X,y,name='tmp'):
    #folder name should be samples(21_26)_transform_features
    #sample name from 0 to len(X)
    dir_2write = "D:/SPLBM_output/chunks/"
    #dir_2write = ('../chunks/') #Darwin
    
    dir_name = dir_2write + name + '/'
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.mkdir(dir_name + 'X')
        os.mkdir(dir_name + 'y')
    
    for i in range(0,X.shape[0]):
        np.save( (dir_name+'X/'+str(i)), X[i,:,:,:,:] )
        np.save( (dir_name+'y/'+str(i)), y[i,:,:,:,:] )

def calculate_DarcyPerm(v_avg,d_size=500):
    mu = 1/3
    dp = 0.0000001*(d_size/500)
    dpdx = dp/d_size
    k = v_avg*mu/dpdx
    return k   

    

def crop_sample(y, crop_size):
    
    m = np.copy(y)
    
    for i in range(0,crop_size):
        m=np.delete(m,-1,0) #get rid of the boundaries
        m=np.delete(m,0 ,0)
        
        m=np.delete(m,-1,1)
        m=np.delete(m,0 ,1)
        
        m=np.delete(m,-1,2)
        m=np.delete(m,0 ,2)
    
    return m

def remove_solid(y1, solid_val=0):
    y = np.copy(y1)
    tmp = y.flatten()
    tmp = tmp[1:1000]
    solid_value = scipy.stats.mode(tmp) #find the mode
    solid_value = solid_value[0] #value
    y[ y==solid_value ] = solid_val
    return y

    
def unsplit_matrix(mt, w_stride=0):
    m_side       = np.int(np.round(mt.size**(1/3)))
    block_size   = mt.shape[-1]
    m = mt.reshape((m_side//block_size, m_side//block_size, -1, block_size, block_size)).transpose(2,0,3,1,4).reshape((m_side, m_side, m_side))

    
    # m = mt.reshape((-1, q//w_size, w_size, r//w_size, w_size)).transpose(1,3,0,2,4).reshape((-1,w_size, w_size, w_size))     
    
    # m_side_cubes = np.int(m_side/mt.shape[1])
    # m = np.zeros((m_side,m_side,m_side))
    
    # sample_start=np.arange(0,m.shape[0]+1,mt.shape[1])
    
    # ii=0
    # for j in range(0,m_side_cubes):
    #     for k in range(0,m_side_cubes):
    #         for i in range(0,m_side_cubes):
    #             m[sample_start[k]:sample_start[k+1],
    #               sample_start[j]:sample_start[j+1],
    #               sample_start[i]:sample_start[i+1]] = np.squeeze( mt[ii,:,:,:] )
    #             ii = ii+1
    
    return m

def crop_matrix(mt, data_shape):
    
    m = mt.copy()
    pad_size = m.shape
    
    m = m[(pad_size[0] - data_shape[0])//2 : -(pad_size[0] - data_shape[0])//2,
          (pad_size[1] - data_shape[1])//2 : -(pad_size[1] - data_shape[1])//2,
          (pad_size[2] - data_shape[2])//2 : -(pad_size[2] - data_shape[2])//2]
    
    return m


