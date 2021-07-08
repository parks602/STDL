from scipy.io import FortranFile
from datetime import timedelta, datetime
import os, sys, json
import pandas as pd
import numpy as np

def isExists(fname):
    if not os.path.exists(fname):
       print("Can't find : %s" %(fname))
       return False
    else:
       return True

def main():
    percen = '20000'
    fmodel = "2019010100-2020010100_T3H_ldaps_%s_data.csv"%(percen)
    m_info = "2019010100-2020010100_T3H_ldaps_%s_info.csv"%(percen)
    fobs   = "all_spatio_temporal_temperature.csv"
    o_info = "all_stninfo.csv"
    if isExists(fmodel) and isExists(fobs):
        obs = pd.read_csv(fobs)
        model = pd.read_csv(fmodel)
        m_info = pd.read_csv(m_info)
        o_info = pd.read_csv(o_info)
    else:
        sys.exit(1)
    ### Merge
    print("merge...")
    df = pd.merge(model, obs ,on='datetime',how='left')
    info = o_info.append(m_info, ignore_index=True)
    ### re-order
    '''
    cols = df.columns.tolist()
    cols.remove('datetime')
    ordered_cols = ['datetime'] + cols
    df = df[ ordered_cols ]
    '''
    dfcol = ['datetime'] + list(range(0,len(info)))
    df.columns = dfcol
    info['stnid'] = list(range(0,len(info)))
    ### Save
    print("saving...")
    df = df.interpolate(limit_direction='both')
    df.to_csv("model+obs_temperature_%s.csv"%(percen))
    info.to_csv("model+obs_info_%s.csv"%(percen), index_label=False)

if __name__ == "__main__":
    main()
