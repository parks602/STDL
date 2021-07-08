import pandas as pd
import numpy as np
import random
import sys

elev   = np.load('output_elev_1KM_Mean_SRTM.npy')
lsm    = np.load('noaa_lsm1km.npy') #0 = sea, 1 = land

elevk  = np.zeros((745,1265,2))
lsmk   = np.zeros((745,1265,2))
elevk[:,:,0], elevk[:,:,1] = elev, empty
lsmk[:,:,0], lsmk[:,:,1] = lsm, empty
#with height
rdmsk  = np.zeros((745,1265))
nn  = 20000
up1000 =  np.where((elevk[:,:,0]>1000)&(lsmk[:,:,0]==1))
rd1000 =  random.sample(range(len(up1000[0])),nn)
print(len(rd1000))
for i in range(nn):
    rdmsk[up1000[0][rd1000[i]], up1000[1][rd1000[i]]]=1

up100  =  np.where((elevk[:,:,0]>100)&(elevk[:,:,0]<1000)&(lsmk[:,:,0]==1))
rd100  =  random.sample(range(len(up100[0])),nn)
print(len(rd100))
for i in range(nn):
    rdmsk[up100[0][rd100[i]], up100[1][rd100[i]]]=1

dw100  =  np.where((elevk[:,:,0]>0)&(elevk[:,:,0]<100)&(lsmk[:,:,0]==1))
rdw100 =  random.sample(range(len(dw100[0])),nn)
print(len(rdw100))
for i in range(nn):
    rdmsk[dw100[0][rdw100[i]], dw100[1][rdw100[i]]]=1

sea    =  np.where((lsmk[:,:,0]<=0)&(lsmk[:,:,1]==0))
rdsea  =  random.sample(range(len(sea[0])),nn)
print(len(rdsea))
for i in range(nn):
    rdmsk[sea[0][rdsea[i]], sea[1][rdsea[i]]]=1

np.save('ldaps_point_%s.npy'%(str(nn)), rdmsk)
'''
#random
rad   = np.where(elevk[:,:,1]==0)
print(len(rad[0]))

radrad= random.sample(range(len(rad[0])), int(len(rad[0])*0.35))
print(len(radrad))
for i in range(len(radrad)):
    rdmsk[rad[0][radrad[i]],rad[1][radrad[i]]]=1

np.save('ldaps_point_35.npy', rdmsk)
'''
'''
#nearby
rdmsk  = np.zeros((745,1265))
rdmsk[:700,:1170]=1
neby   = np.where((elevk[:,:,1]==0)&(rdmsk==0))

rneby  = random.sample(range(len(neby[0])), int(20000))

akmsk  = np.zeros((745,1265))
for i in range(len(rneby)):
  akmsk[neby[0][rneby[i]], neby[1][rneby[i]]]=1
np.save('ldaps_point_nearby.npy', akmsk)
'''
