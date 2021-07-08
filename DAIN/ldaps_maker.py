import pandas as pd
import numpy as np
import random
import sys, os
from datetime import datetime, timedelta
from scipy.io import FortranFile


def isExists(fname):
  if not os.path.exists(fname):
   print("Can't find : %s" %(fname))
   return False
  else:
   return True


def ran_ldaps_height(elev, lsm, point_num):
  print('Random sampling %s LDAPS points with Height'%(str(point_num)))

  '''
  Find RANDOM POINT with height
  =============================================
  '''

  rdmsk  = np.zeros((745, 1265))

  up100  = np.where((elev < 1000) & (elev > 100) & (lsm==1))
  rd100  = random.sample(range(len(up100[0])), point_num)

  dw100  = np.where((elev < 100) & (elev > 0) & (lsm==1))
  rdw100 = random.sample(range(len(dw100[0])), point_num)

  #sea    = np.where((elev <= 0) & (lsm==0))
  #rdsea  = random.sample(range(len(sea[0])), point_num)

  up1000 = np.where((elev > 1000) & (lsm==1))
  if len(up1000[0])<point_num:
    rd1000 = up1000
    rdmsk[rd1000] =1
    for i in range(point_num):
      rdmsk[up100[0][rd100[i]], up100[1][rd100[i]]] =1
      rdmsk[dw100[0][rdw100[i]], dw100[1][rdw100[i]]] =1
      #rdmsk[sea[0][rdsea[i]], sea[1][rdsea[i]]] =1

  else:
    rd1000 = random.sample(range(len(up1000[0])), point_num)
    for i in range(point_num):
      rdmsk[up1000[0][rd1000[i]], up1000[1][rd1000[i]]] =1
      rdmsk[up100[0][rd100[i]], up100[1][rd100[i]]] =1
      rdmsk[dw100[0][rdw100[i]], dw100[1][rdw100[i]]] =1
      #rdmsk[sea[0][rdsea[i]], sea[1][rdsea[i]]] =1
  print('Random sampling is finished')

  return rdmsk

def Stratified_sampling_height(elev, lsm, point_num):
  print('Random sampling %s LDAPS points with Height'%(str(point_num)))

  '''
  Find RANDOM POINT with height
  =============================================
  '''
  high_num = int(point_num*0.1)
  mid_num = int(point_num*0.7)
  low_num = int(point_num*0.2)
  rdmsk  = np.zeros((745, 1265))

  up100  = np.where((elev < 1000) & (elev >= 100) & (lsm==1))
  rd100  = random.sample(range(len(up100[0])), mid_num)

  dw100  = np.where((elev < 100) & (lsm==1))
  rdw100 = random.sample(range(len(dw100[0])), low_num)

  #sea    = np.where((elev <= 0) & (lsm==0))
  #rdsea  = random.sample(range(len(sea[0])), point_num)

  up1000 = np.where((elev >= 1000) & (lsm==1))
  if len(up1000[0])<high_num:
    rd1000 = up1000
    rdmsk[rd1000] =1
    for i in range(mid_num):
      rdmsk[up100[0][rd100[i]], up100[1][rd100[i]]] =1
    for j in range(low_num):
      rdmsk[dw100[0][rdw100[j]], dw100[1][rdw100[j]]] =1
      #rdmsk[sea[0][rdsea[i]], sea[1][rdsea[i]]] =1

  else:
    rd1000 = random.sample(range(len(up1000[0])), high_num)
    for i in range(high_num):
      rdmsk[up1000[0][rd1000[i]], up1000[1][rd1000[i]]] =1
    for j in range(mid_num):
      rdmsk[up100[0][rd100[j]], up100[1][rd100[j]]] =1
    for k in range(low_num):
      rdmsk[dw100[0][rdw100[k]], dw100[1][rdw100[k]]] =1
      #rdmsk[sea[0][rdsea[i]], sea[1][rdsea[i]]] =1
  print('Random sampling is finished')
  np.save('sample_mask.npy', rdmsk)
  return rdmsk

def all_ldaps_obs(lsm, del_obs_info):
  rdmsk = lsm
  rdmsk[del_obs_info['xpos'].values, del_obs_info['ypos'].values] = 0
  nrdmsk = np.zeros((745, 1265))
  rnpos   = np.where(rdmsk==1)
  rnlist  = random.sample(range(len(rnpos[0])), 320000)
  for i in range(320000):
    nrdmsk[rnpos[0][rnlist[i]], rnpos[1][rnlist[i]]] = 1
  return nrdmsk

def ran_ldaps(lsm, point_num):
  print('Random sampling %s LDAPS points with Height'%(str(point_num)))

  '''
  Find RANDOM POINT with LSM
  =============================================
  '''

  rdmsk    = np.zeros((745, 1265))
  point    = np.where(lsm==1)
  se_point = random.sample(range(len(point[0])), point_num)

  for i in range(point_num):
    rdmsk[point[0][se_point[i]], point[1][se_point[i]]] = 1
  print('Random sampling is finished')

  return rdmsk

def read_obs(obs_fname, stn_fname):
  '''
  FORTRAN TO NUMPY OBS DATA (LEN(STN INFO) = LEN(DATASET))
  =============================================
  '''

  if not isExists(obs_fname): return
  with FortranFile(obs_fname,'r') as f:
    info = f.read_ints(np.int32)
    stnlist = f.read_ints(np.int32)
    data = f.read_reals(np.float32)
    data = np.reshape(data, info[:7:-1]) ### nstn, nyear, nmonth, nday, nhour
  data = np.transpose(data)[:,:,:,:,:24]  # 24hour
  stn_list   = list(stnlist)
  stn_info  = pd.read_csv(stn_fname)
  info_list = list(stn_info['stnid'])
  real_list = []
  for i in info_list:
    if i in stn_list:
      real_list.append(np.where(stnlist==i)[0][0])
    else:
      stn_info = stn_info.drop(stn_info[stn_info['stnid']==i].index)
  if len(real_list) == len(stn_info['stnid']):
    pass
  else:
    print('Something is wrong')
    sys.exit(1)
  stn_info = stn_info.reset_index(drop = True, inplace = False)
  print(data[real_list].shape, stn_info.shape)
  return data[real_list], stn_info
  


def obs_data_maker(aws_names, obs_fname, stn_fname, sdate, edate, point_num, var):
  all_data, all_stn  = None, None
  for i, aws_name in enumerate(aws_names):
    data, stnlist = read_obs(obs_fname%(aws_name, var), stn_fname%(aws_name, aws_name.lower()))
    if i == 0:
      all_data = data
      all_stn  = stnlist
    else:
      all_data = np.concatenate((all_data, data), axis=0)
      all_stn  = pd.concat([all_stn, stnlist])

  col_name    = list(range(point_num*4, point_num*4+all_data.shape[0]))
  start        = datetime.strptime(sdate,"%Y%m%d%H")
  end          = datetime.strptime(edate,"%Y%m%d%H")
  now          = start
  refined_data = []

  while now <= end:
    print(now)
    kst_now = now+timedelta(hours=9)
    year, month, day, hour = kst_now.year-2016, kst_now.month-1, kst_now.day-1, kst_now.hour
    on_data = all_data[:, year, month, day, hour]
    refined_data.append(list(on_data))    
    now = now+timedelta(hours= 3)
  dt        = pd.DataFrame(data = refined_data, columns = col_name)
  dt.index  = pd.DatetimeIndex(pd.date_range(start=start, end=end, freq='180min'))
  dt        = dt.replace(-999.0, np.NaN)
  dt        = dt.interpolate(method='time', limit_direction='both')
  dt.reset_index(drop=True, inplace=False)
  nul_list = np.where(dt.isnull().any() == True)
  stid = np.reshape(np.array(range(point_num*4, point_num*4+all_data.shape[0])), (len(col_name), 1))
  info = pd.DataFrame(data = np.concatenate((stid, all_stn[['longitude', 'latitude', 'altitude']].values), axis = 1), \
                      columns = ['stnid', 'lon', 'lat', 'hgt'])
  for i in nul_list[0]:
    info = info.drop(i)
  print(info)
  return dt, info


def ldaps_data_maker(var, point_num, sdate, edate, data_dir, ra_ldaps, elev, gis):

  '''
  LDAPS DATASET MAKE WITH RANDOM POINT
  =============================================
  '''
  start     = datetime.strptime(sdate,"%Y%m%d%H")
  end       = datetime.strptime(edate,"%Y%m%d%H")
  now       = start

  hgt       = np.reshape(elev, (elev.shape[0] * elev.shape[1], 1))
  lat       = np.reshape(gis[:,:,1], (elev.shape[0] * elev.shape[1], 1))
  lon       = np.reshape(gis[:,:,0], (elev.shape[0] * elev.shape[1], 1))
  if isExists('../DATA/mesh_1km.csv') == False:
    mesh    = pd.DataFrame(data = np.concatenate((lon, lat, hgt), axis = 1), columns = ['lon', 'lat', 'hgt'])
    mesh.to_csv('../DATA/mesh_1km.csv', index_label = False)
  ra_list = np.where(ra_ldaps==1)
  stnid_num = len(np.where(ra_ldaps==1)[0])
  data      = []

  while now <= end:
    print(now)
    stnow   = datetime.strftime(now, "%Y%m%d%H")
    ld      = np.load('%s/%s_%s.npy'%(data_dir, var, stnow))
    se_ld   = ld[ra_list[0],ra_list[1]]
    data.append(list(se_ld))
    now = now + timedelta(hours=3)
  print(len(data[0]))
  dt        = pd.DataFrame(data = data, columns = list(range(stnid_num)))
  dt.index  = pd.DatetimeIndex(pd.date_range(start=start, end=end, freq='180min'))
  dt        = dt.replace(-999.0, np.NaN)
  dt        = dt.interpolate(method='time', limit_direction='both')
  dt.reset_index(drop=True, inplace=False)

  '''
  LDAPS DATASET INFO MAKE WITH RANDOM POINT
  =============================================
  '''
  stid = np.reshape(np.array(range(stnid_num)), (len(range(stnid_num)), 1))
  hgt  = np.reshape(elev[ra_list[0], ra_list[1]], stid.shape)
  lat  = np.reshape(gis[:,:,1][ra_list[0], ra_list[1]], stid.shape)
  lon  = np.reshape(gis[:,:,0][ra_list[0], ra_list[1]], stid.shape)
  info = pd.DataFrame(data = np.concatenate((stid, lon, lat, hgt), axis = 1), \
                      columns = ['stnid', 'lon', 'lat', 'hgt'])
  print(info)
  return dt, info


def main():
  var             = 'REH'
  elev            = np.load('output_elev_1KM_Mean_SRTM.npy') #Height (window = 1km)
  lsm             = np.load('noaa_lsm1km.npy') #Land SEA mask (window = 1km, 0 = sea, 1 = land)
  gis             = np.load('grid_info1km.npy') #GIS
  point_num       = 100000
  sdate, edate    = '2019010100', '2020010100'
  data_dir        = "/home/ubuntu/pkw/DATA/KOR1KM/%s"%(var)
  obs_dir         = '/home/ubuntu/pkw/DATA/OBS/AWS_%s/QC/qc_obs_%s.2016010100-2020072223'
  stn_info        = '/home/ubuntu/pkw/DATA/OBS/AWS_%s/%s_aws_pos.csv'
  aws_names       = ['KMA', 'KFS', 'RDA']
  obs_data, obs_info = obs_data_maker(aws_names, obs_dir, stn_info, sdate, edate, point_num, var)
  '''
  if os.path.exists('%s_sample_mask.npy'%(point_num)):
    ra_list = np.load('sample_mask.npy')
  else:
    #ra_list         = ran_ldaps_height(elev, lsm, point_num)
    ra_list         = Stratified_sampling_height(elev, lsm, point_num)
    #ra_list         = ran_ldaps(lsm, point_num)
  '''
  del_obs_info    = pd.read_csv('%s_obs_info_pos.csv'%(var))
  ra_list         = all_ldaps_obs(lsm, del_obs_info)
  data, info      = ldaps_data_maker(var, point_num, sdate, edate, data_dir, ra_list, elev, gis)
  data            = pd.concat([data, obs_data], axis=1)
  info            = pd.concat([info, obs_info], ignore_index=True)
  data            = data.interpolate(method='time', limit_direction='both')
  data            = data.dropna(axis = 1)
  print(data.shape, data)
  print(info.shape, info)
  col_name        = list(range(data.shape[1]))
  data.columns    = col_name
  info.stnid      = col_name
  print(info.shape)
  data.to_csv('../DATA/%s_all_data.csv'%(var), index_label='datetime')
  #data.to_csv('../DATA/%s-%s_%s_%s_stratify_data.csv'%(sdate, edate,var, point_num), index_label='datetime')
  #info.to_csv('../DATA/%s-%s_%s_%s_stratify_info.csv'%(sdate, edate, var, point_num), index_label = False)
  info.to_csv('../DATA/%s_all_info.csv'%(var), index_label = False)
  #data.to_csv('REH_obs_data.csv', index_label = 'datetime')
  #info.to_csv('REH_obs_info.csv', index_label = False)

if __name__ == '__main__':
  main()
