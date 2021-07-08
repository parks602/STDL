import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys


    
def main():
    var      = 'T3H'
    percen   = '20000'
    sdate    = '2018010100'
    #edate    = '2018113021'
    edate    = '2020010100'
    start    = datetime.strptime(sdate,"%Y%m%d%H")
    end      = datetime.strptime(edate,"%Y%m%d%H")
    ipath    = "/home/ubuntu/pkw/DATA/KOR1KM/%s"%(var)
    ra_ldaps = np.load('/home/ubuntu/pkw/ST_DL_alpha/DAIN/ldaps_point_%s.npy'%(percen))
    ra_list  = np.where(ra_ldaps==1)
    now      = start
    elev     = np.load('/home/ubuntu/pkw/ST_DL_alpha/DAIN/output_elev_1KM_Mean_SRTM.npy')
    gis      = np.load('/home/ubuntu/pkw/ST_DL_alpha/DAIN/grid_info1km.npy')
    lats,lons= gis[:,:,1], gis[:,:,0]
    hgt      = elev
    hgt      = np.reshape(np.array(hgt),(hgt.shape[0]*hgt.shape[1],1))
    lat      = np.reshape(np.array(lats),(hgt.shape[0]*hgt.shape[1],1))
    lon      = np.reshape(np.array(lons),(hgt.shape[0]*hgt.shape[1],1))
    mesh     = pd.DataFrame(data = np.concatenate((lon,lat,hgt), axis=1), columns=['lon', 'lat', 'hgt'])
    mesh.to_csv('mesh_1km.csv', index_label =False)

    stnid_num = len(np.where(ra_ldaps==1)[0])
    data     = []
    while now <= end:
        print(now)
        stnow  = datetime.strftime(now, "%Y%m%d%H")
        ld     = np.load('%s/%s_%s.npy'%(ipath, var, stnow))
        se_ld  = ld[ra_list[0],ra_list[1]]
        data.append(list(se_ld))
        now = now + timedelta(hours=3)
    print(len(data[0]))
    dt       = pd.DataFrame(data = data, columns = list(range(stnid_num)))
    dt_idx   = pd.DatetimeIndex(pd.date_range(start=start, end=end, freq='180min'))
    dt.index = dt_idx
    dt = dt.replace(-999.0, np.NaN)
    dt = dt.interpolate(method='time', limit_direction='both')
    dt.reset_index(drop=True)
    dt.to_csv('%s-%s_%s_ldaps_%s_data.csv'%(sdate, edate,var, percen), index_label='datetime')

    elev     = np.load('/home/ubuntu/pkw/ST_DL_alpha/DAIN/output_elev_1KM_Mean_SRTM.npy')
    gis      = np.load('/home/ubuntu/pkw/ST_DL_alpha/DAIN/grid_info1km.npy')
    lats,lons= gis[:,:,1], gis[:,:,0]
    hgt      = elev[ra_list[0], ra_list[1]]
    lat      = lats[ra_list[0], ra_list[1]]
    lon      = lons[ra_list[0], ra_list[1]]
    stid     = np.reshape(np.array(range(stnid_num)), (len(range(stnid_num)),1))
    hgt      = np.reshape(np.array(hgt),(len(hgt),1))
    lat      = np.reshape(np.array(lat),(len(lat),1))
    lon      = np.reshape(np.array(lon),(len(lon),1))
    infodf   = pd.DataFrame(data = np.concatenate((stid,lon,lat,hgt),axis=1), columns = ['stnid','lon','lat', 'hgt'])
    infodf.to_csv('%s-%s_%s_ldaps_%s_info.csv'%(sdate, edate, var, percen), index_label=False)
if __name__ == "__main__":
    main()


