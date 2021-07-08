import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import os, sys
from utils import get_argument_parser, create_folder
import warnings
import result_plot

def Separate_Yhat(args):
    print('SEPARATION BEGINS ALL STDL DATA WITH DATE')
    var     = args.var
    pred    = np.load('%s/y_hat_mesh.npy' %(args.outf))
    lsm     = np.load('%s/noaa_lsm1km.npy' %(args.dainf))
    land    = np.where(lsm==1)
    all_dir = '%s/all/'%(args.outf)
    cat_dir = '%s/cat/'%(args.outf)

    create_folder(all_dir)
    create_folder(cat_dir)

    ldaps_dir = args.ldapsdir

    sdate = datetime.strptime(args.sdate, args.fmt)
    edate = datetime.strptime(args.edate, args.fmt)
    now = sdate
    k     = 0
    while now <= edate:
        dtime = now.strftime(args.fmt)
        fname = '%s/%s_%s.npy'%(args.ldapsdir, args.var, dtime)
        data  = np.load(fname)
        inte  = np.reshape(pred[:,k], (745,1265))
        data[land] = inte[land]
        np.save('%s%s_%s.npy'%(cat_dir, args.var, dtime), data)
        np.save('%s%s_%s.npy'%(all_dir, args.var, dtime), inte)
        now = now + timedelta(hours = 3)
        k = k +1


def RMSE_Calculator(args):
    print('RMSE IS CALCULATING....')
    lsm      = np.load('%s/noaa_lsm1km.npy' %(args.dainf))
    grid     = np.load('%s/grid_info1km.npy' %(args.dainf))
    mlon, mlat   = grid[:,:,0], grid[:,:,1]

    obs_data = pd.read_csv('%s/%s_obs_data.csv'%(args.dainf, args.var))
    obs_info = pd.read_csv('%s/%s_obs_info.csv'%(args.dainf, args.var))
    obs_data = obs_data[obs_data.columns[1:]].values
    obs_info_pos = '%s/%s_obs_info_pos.csv'%(args.dainf, args.var)

    if os.path.exists(obs_info_pos) == False:
        pos_info = pos_stn(obs_info, mlat, mlon, args.var)
    else:
        pos_info = pd.read_csv(obs_info_pos, index_col=False)

    land_len, sea_len  = len(np.where(lsm==1)[0]), len(np.where(lsm==0)[0])
    all_len  = lsm.shape[0] * lsm.shape[1]
    aws_len  = obs_data.shape[0]


    sdate = datetime.strptime(args.sdate, args.fmt)
    edate = datetime.strptime(args.edate, args.fmt)
    now   = sdate
    k     = 0

    while now <= edate:
        st_now = datetime.strftime(now, args.fmt)
        hat    = np.load('%s/all/%s_%s.npy' %(args.outf, args.var, st_now))
        answer = np.load('%s/%s_%s.npy' %(args.ldapsdir, args.var, st_now))

        differ = hat-answer

        pos_answer = obs_data[k]
        pos_pred   = hat[pos_info['xpos'].values, pos_info['ypos'].values]
        pos_pred   = answer[pos_info['xpos'].values, pos_info['ypos'].values]

        if  k == 0:
            all_rm = np.sum(differ**2)
            land   = np.sum(differ[np.where(lsm==1)]**2)
            sea    = np.sum(differ[np.where(lsm==0)]**2)
            aws    = np.sum((pos_pred-pos_answer)**2)
        else:
            all_rm = all_rm +  np.sum(differ**2)
            land   = land + np.sum(differ[np.where(lsm==1)]**2)
            sea    = sea + np.sum(differ[np.where(lsm==0)]**2)
            aws    = aws + np.sum((pos_pred-pos_answer)**2)
        k = k+1
        now = now+timedelta(hours=3)
    save_file = pd.DataFrame(data = np.reshape(np.array((math.sqrt(all_rm/(k*all_len)), \
                            math.sqrt(land/(k*land_len)), math.sqrt(sea/(k*sea_len)), \
                            math.sqrt(aws/(k*aws_len)))), (1,4)), columns = ['ALL', 'LAND','SEA','OBS'])
    save_file.to_csv('%s/rmse.csv'%(args.outf))
    
def cal_dist(mlat, mlon, olat, olon):
    return np.sqrt((mlat - olat)**2 + (mlon - olon)**2)

def find_nearest(mlat, mlon, olat, olon):
    dist = cal_dist(mlat, mlon, olat, olon)
    nx, ny = np.unravel_index(dist.argmin(), dist.shape)
    return nx, ny

def pos_stn(stn_info, mlat, mlon, var):
    olons = stn_info['lon'].values
    olats = stn_info['lat'].values
    stn_info['xpos'] = 0
    stn_info['ypos'] = 0
    for i in range(len(olons)):
        nx, ny = find_nearest(mlat, mlon, olats[i], olons[i])
        stn_info['xpos'][i] = nx
        stn_info['ypos'][i] = ny
    stn_info.to_csv('%s/%s_obs_info_pos.csv'%(args.dainf, args.var))
    return(stn_info)



def main():
    parser = get_argument_parser()
    args   = parser.parse_args()
    #Separate all ydata
    #Separate_Yhat(args)
    #RMES calculator
    pd.set_option('mode.chained_assignment', None)
    #RMSE_Calculator(args)
    warnings.filterwarnings(action='ignore')
    #Make IMAGE
    result_plot.Make_Image(args)


if __name__ == '__main__':
    main()
