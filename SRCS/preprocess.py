import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def data_preprocessing(args):
    dset, mesh, _, _, datetime = read_data(args)
    #mesh = mesh[ mesh['hgt'] > 0 ] ## Only Land
    nvalid= int(len(dset['X_train']) * args.train_val_split)
    #### Save Dataset
    #np.savetxt("%s/Xtest.txt" %(args.outf), X_test, delimiter=",")
    #np.savetxt("%s/Xtrain.txt" %(args.outf), X_train.iloc[:nvalid,:], delimiter=",")
    ## Decomposition
    u, s, v, nS, time_mean, n_eofs = decomposition(args, dset['X_train'], dset['y_train'])
    ## Modelling the coefficient maps and reconstruction of the full spatio-temporal field
    X_train, X_valid = dset['X_train'].values[:nvalid,:], dset['X_train'].values[nvalid:,:]
    u_train, u_valid = u[:nvalid,:n_eofs], u[nvalid:,:n_eofs]
    s = s[:n_eofs]
    v = v[:,:n_eofs]
    print(X_train)
    print(mesh)
    scaler = StandardScaler()
    scaler.fit(mesh.iloc[:,:3])
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(dset['X_test'])
    X_valid_scaled = scaler.transform(X_valid)
    X_mesh_scaled = scaler.transform(mesh.iloc[:,:3])

    Trainset = TemporalSpatialDataset(X_train_scaled,dset['y_train'][:nvalid,:],u_train)
    Validset = TemporalSpatialDataset(X_valid_scaled,dset['y_train'][nvalid:,:],u_valid)

    train_loader = DataLoader(dataset=Trainset,
                              batch_size=args.batchSize,
                              shuffle=True,
                              num_workers=args.workers)
    valid_loader = DataLoader(dataset=Validset,
                              batch_size=len(Validset),
                              shuffle=False,
                              num_workers=args.workers)
    n_in = X_train_scaled.shape[1]
    print(X_train_scaled)
    print(X_mesh_scaled)
    return train_loader, valid_loader, X_test_scaled, X_mesh_scaled, n_in,\
           u, s, v, nS, time_mean, n_eofs

def minmax(array, min_value, max_value):
    minmax_array = (array-min_value) / (max_value - min_value)
    return(minmax_array)


def read_data(args):
    ### Data Preprocessing
    data = pd.read_csv('%s' %(args.dataf),index_col=False)
    info = pd.read_csv('%s' %(args.infof))
    mesh = pd.read_csv('%s' %(args.meshf))
    #info = info.iloc[np.random.permutation(info.index)].reset_index(drop=True)
    # Height to logcale
    #info['hgt'] = info['hgt'].apply(lambda x: x if x == 0 else np.log(x))
    #mesh['hgt'] = mesh['hgt'].apply(lambda x: x if x == 0 else np.log(x))

    timesteps = 0
    data = data.iloc[timesteps:,:]
    dtime= pd.to_datetime(data['datetime'])
    dtime= dtime.dt.strftime("%Y%m%d%H").values.astype(int)
    data.set_index('datetime',inplace=True)  ## set datetime index
    ### All Dataset
    field = data.iloc[:,:]
    #field = np.transpose(field.values) ## space-wide format (where space varies along the columns and time varies along the rows)
    field_coords = info[['lon', 'lat', 'hgt']]
    ### drop stnid not in info
    info_id = info['stnid'].tolist()
    data_id = list(map(int,data.columns.tolist()))
    if len(info_id) < len(data_id):
        remove_id = set(data_id) - set(info_id)
        remove_id = list(map(str,remove_id))
        data.drop(remove_id,axis=1,inplace=True)
    ### Sampling station ID for Trainging
    sampl = data.sample(frac=1, random_state=args.manualSeed, axis=1)
    stn_sampl = list(map(int,sampl.columns.tolist()))
    sampl = np.transpose(sampl.values[timesteps:,:])

    ### Extract Dataset for Training
    sampl_coords = info[ info['stnid'].isin(stn_sampl) ]
    ### Make Dataset
    sampl_coords_T = sampl_coords.set_index('stnid').T
    sampl_coords_T = sampl_coords_T[ stn_sampl ]  ##  Matching Order of STN
    sampl_coords = sampl_coords_T.T
    sampl_coords = sampl_coords[['lon', 'lat', 'hgt']]

    # split datasets into trainset(+valid) & testset
    ndata = sampl.shape[0]
    ntest = int(ndata * args.ptest)
    ntrain= ndata - ntest

    # % split
    dset = {}
    datetime = {}
    coords = {}
    dset['X_train'] = sampl_coords.iloc[:ntrain, :]
    dset['X_test'] = sampl_coords.iloc[ntrain:,:]
    dset['y_train'] = sampl[:ntrain, :]
    dset['y_test'] = sampl[ntrain:,:]

    coords['field_coords'] = field_coords
    coords['test_coords'] = sampl_coords.iloc[ntrain:,:]

    # random sampling
    #train_idx = np.random.choice(range(ndata), ntrain, replace=False)
    #test_idx  = np.delete(np.array(range(ndata)), train_idx)
    #X_train = sampl_coords.iloc[train_idx, :]
    #X_test = sampl_coords.iloc[test_idx,:]
    #y_train = sampl[train_idx, :]
    #y_test = sampl[test_idx,:]
    #test_coords = sampl_coords.iloc[test_idx,:]
    return dset, mesh, field, coords, dtime

class TemporalSpatialDataset(Dataset):
    def __init__(self, X, Y, U):
        self.X = torch.from_numpy(X).type(torch.FloatTensor)
        self.Y = torch.from_numpy(Y).type(torch.FloatTensor)
        self.U = U.type(torch.FloatTensor)
        print("X: ",X.shape)
        print("Y: ",Y.shape)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx], self.U[idx]

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).type(torch.FloatTensor)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def decomposition(args, X_train, y_train):
    ### SVD decomposition of training data
    Z = torch.Tensor(y_train)
    print('Matrix Z: \n{}\n'.format(Z))

    time_mean = torch.mean(Z, 0)
    nS = torch.Tensor([Z.shape[0]]).type(torch.FloatTensor)
    time_mean = torch.reshape(time_mean.repeat(nS.type(torch.IntTensor)),[nS.type(torch.IntTensor), time_mean.shape[0]])

    Ztime_detrend = Z - time_mean
    Ztilde = 1/torch.sqrt(nS-1)*Ztime_detrend    ### 정규화된 공간계수 산출
    print('Matrix Ztiled: \n{}\n'.format(Ztilde))

    # Using tf.linalg.svd to calculate the singular value decomposition
    u, s, v = torch.svd(Ztilde)    # u : 공간,  v : 시간,  s : 대각행렬
    u =  u * torch.sqrt(nS-1)
    print('Diagonal S: \n{} \n\nMatrix U: \n{} \n\nMatrix V^T: \n{}'.format(s, u, v))
    print('s shape :', s.shape)
    print('u shape :', u.shape)
    print('v shape :', v.shape)
    # Plot
    n_eofs = plot_svd(args, u, s, v, X_train)
    # To recompose
    v_ct = (torch.conj(v[:,:n_eofs])).t()  ## conjugated and transpose
    tf_Z_approx = torch.matmul((u[:,:n_eofs]/torch.sqrt(nS-1)), torch.matmul(torch.diag(s[:n_eofs]), v_ct))

    print('Matrix Zt, reconstructed: \n{}\n'.format(tf_Z_approx))

    reconstructed = tf_Z_approx * torch.sqrt(nS-1) + time_mean

    check = reconstructed - Z
    print('True - reconstructed data: \n{}\n'.format(check))
    return u, s, v, nS, time_mean, n_eofs

def plot_svd(args, u, s, v, X_train):
    maxsize = min(s.size()[0], args.limit_max_eofs)
    ### calcalate n_eofs
    n_eofs = maxsize
    cumsum = np.cumsum(torch.square(s)/torch.sum(torch.square(s)))[:maxsize]
    '''
    for i, val in enumerate(cumsum):
        if i >= min(maxsize, args.min_n_eofs) and val > args.exvar_thres:
            n_eofs = i
            break
    print(f"n_eofs : {n_eofs}")
    ### Plot SVD
    plt.figure(figsize=(6,2.25))
    plt.plot(np.arange(1, maxsize+1), (np.cumsum(torch.square(s)/torch.sum(torch.square(s))))[:maxsize], c='black')
    plt.axvline(n_eofs, color='black', linestyle='--', linewidth =1)
    #plt.axvline(565, color='black', linestyle='--', linewidth =1)
    #plt.title('Cumulative sum of explained varicane by component')
    ### 설명된 총분산 - 요소가 전체 분산 중 얼마만큼을 설명하는지 나타내기 위한 값
    plt.ylabel('Cumulative \n explained variance')
    plt.xlabel('Number of components')
    plt.savefig("%s/Simulated_screeplot_noise.pdf" %(args.outf), dpi=300)
    plt.close()
    #print((np.cumsum(torch.square(s)/torch.sum(torch.square(s))))[:25])
    ### Plot Decomposition
    plt.figure(figsize=(20,8))
    plt.subplot(2,3,1)
    plt.plot(np.arange(len(v[:,0])), v[:,0], color='black', alpha=0.5)
    plt.title('EOF 1')
    plt.subplot(2,3,2)
    plt.plot(np.arange(len(v[:,1])), v[:,1], color='black', alpha=0.5)
    plt.title('EOF 2')
    plt.subplot(2,3,3)
    plt.plot(np.arange(len(v[:,2])), v[:,2], color='black', alpha=0.5)
    plt.title('EOF 3')

    plt.subplot(2,3,4)
    plt.scatter(X_train['lon'], X_train['lat'],c=u[:,0])
    plt.colorbar()
    plt.title('EOF 1 (spatial coefficients)')
    plt.subplot(2,3,5)
    plt.scatter(X_train['lon'], X_train['lat'],c=u[:,1])
    plt.colorbar()
    plt.title('EOF 2 (spatial coefficients)')
    plt.subplot(2,3,6)
    plt.scatter(X_train['lon'], X_train['lat'],c=u[:,2])
    plt.colorbar()
    plt.title('EOF 3 (spatial coefficients)')
    plt.savefig("%s/Components_noise.pdf" %(args.outf), dpi=300)
    plt.close()
    '''
    return n_eofs
