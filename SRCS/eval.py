import numpy as np
import math
from datetime import datetime, timedelta
import pandas as pd
import os

def rmse_calculator(hat_dir, answer_dir, var, lsm, sdate, edate, pos_info, obs_data):
  land_len = len(np.where(lsm==1)[0])
  sea_len  = len(np.where(lsm==0)[0])
  all_len  = lsm.shape[0] * lsm.shape[1]
  aws_len  = obs_data.shape[0]
  dt_sdate = datetime.strptime(sdate, "%Y%m%d%H")
  dt_edate = datetime.strptime(edate, "%Y%m%d%H")
  now = dt_sdate
  k = 0
  while now <= dt_edate:
    st_now = datetime.strftime(now, "%Y%m%d%H")
    hat    = np.load('%s/%s_%s.npy' %(hat_dir, var, st_now))
    answer = np.load('%s/%s_%s.npy' %(answer_dir, var, st_now))
    
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

  return(math.sqrt(all_rm/(k*all_len)), math.sqrt(land/(k*land_len)), math.sqrt(sea/(k*sea_len)),\
          math.sqrt(aws/(k*aws_len)))
  

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
  stn_info.to_csv('/home/ubuntu/pkw/PKW_STDL/DAIN/%s_obs_info_pos.csv'%(var))
  return(stn_info)  
    
def main():
  pd.set_option('mode.chained_assignment',  None)
  var          = 'T3H'
  lsm          = np.load('../DAIN/noaa_lsm1km.npy')
  model_name   = '/LDAPS_OBS_ALL'
  hat_dir      = '/home/ubuntu/pkw/PKW_STDL/DAOU/%s/%s/all'%(model_name, var)
  answer_dir   = '/home/ubuntu/pkw/DATA/KOR1KM/%s'%(var)
  sdate, edate = '2019010100', '2020010100'
  stn_info     = pd.read_csv('/home/ubuntu/pkw/PKW_STDL/DAIN/%s_obs_info.csv'%(var))
  obs_data     = pd.read_csv('/home/ubuntu/pkw/PKW_STDL/DAIN/%s_obs_data.csv'%(var))
  obs_data     = obs_data[obs_data.columns[1:]].values
  grid         = np.load('/home/ubuntu/pkw/PKW_STDL/DAIN/grid_info1km.npy')
  mlon, mlat   = grid[:,:,0], grid[:,:,1]
  obs_info_pos = '/home/ubuntu/pkw/PKW_STDL/DAIN/%s_obs_info_pos.csv'%(var)

  if os.path.exists(obs_info_pos) == False:
    pos_info = pos_stn(stn_info, mlat, mlon, var)
  else:
    pos_info = pd.read_csv(obs_info_pos, index_col=False)

  
  all_rmse, land_rmse, sea_rmse, aws_rmse = rmse_calculator(hat_dir, answer_dir, var, lsm, sdate, edate, pos_info, obs_data)
  print('%s %s RMSE RESULT'%(model_name, var))
  print('ALL SPOT RMSE = ', all_rmse)
  print('LAND SPOT RMSE = ', land_rmse)
  print('SEA SPOT RMSE = ', sea_rmse)
  print('OBS SPOT RMSE = ', aws_rmse)

if __name__ == '__main__':
  main()
