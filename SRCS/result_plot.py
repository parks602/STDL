import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from cmap import COLORBARS
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.colors as colors


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)

        vlargest = max( abs( self.vmax - self.midpoint ), abs( self.vmin - self.midpoint ) )
        x, y = [ self.midpoint - vlargest, self.midpoint, self.midpoint + vlargest], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), mask=result.mask, copy=False)



def make_colormaps(var):
    maxc = 255.
    cmaps = COLORBARS()
    c_value, c_over, c_under= cmaps.getColorBar(var)
    colors = np.array(c_value) / maxc
    colors = colors.tolist()
    cms = LinearSegmentedColormap.from_list(var,colors,N=len(colors))
    if c_over is not None:   cms.set_over(np.array(c_over) / maxc)
    if c_under is not None : cms.set_under(np.array(c_under) / maxc)
    return cms

def set_levels(var, vmin, vmax):
    if var == "T3H":
        if vmin < -20 and vmax < 20:
            clev = np.linspace(-40,20,61,endpoint=True)
            ticks= np.linspace(-40,20,31,endpoint=True)
        elif vmax > 20 and vmin > -20:
            clev = np.linspace(-20,40,61,endpoint=True)
            ticks= np.linspace(-20,40,31,endpoint=True)
        elif vmin > -30 and vmax < 30:
            clev = np.linspace(-30,30,61,endpoint=True)
            ticks= np.linspace(-30,30,31,endpoint=True)
        else:
            clev = np.linspace(-40,20,61,endpoint=True)
            ticks= np.linspace(-40,20,31,endpoint=True)
    elif var == "REH":
        clev = np.linspace(0,100,21,endpoint=True)
        ticks= clev
    return clev, ticks


def comparision(var, ldaps_dir, stdl_ldaps_dir, stdl_dir, mloc, date, save_dir):
  fig = plt.figure(figsize = (21,12))
  ldaps        = np.load('%s/%s_%s.npy'%(ldaps_dir, var, date))
  stdl         = np.load('%s/%s_%s.npy'%(stdl_dir, var, date))
  stdl_ldaps   = np.load('%s/%s_%s.npy'%(stdl_ldaps_dir, var, date))
  diff         = stdl_ldaps -ldaps
  cms = make_colormaps(var)

  llat = np.min(mloc[:,:,1])
  rlat = np.max(mloc[:,:,1])
  llon = np.min(mloc[:,:,0])
  rlon = np.max(mloc[:,:,0])

  level        = np.hstack((ldaps, stdl))
  clev, ticks  = set_levels(var, np.min(level), np.max(level))

  ax           = fig.add_subplot(141)
  ax.set_title('LDAPS %s %s'%(var, date))
  m = Basemap(projection='lcc', lat_1=30., lat_2=60., lon_0=126.0, lat_0=38.0,
            llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=rlon,urcrnrlat=rlat,resolution='h')
  m.drawcoastlines()
  m.drawmapboundary()
  m.drawparallels(np.arange(llat, rlat, 2.), labels=[1,0,0,0], linewidth=0.1, fmt='%.1f')
  m.drawmeridians(np.arange(llon, rlon, 5.), labels=[0,0,0,1], linewidth=0.1, fmt='%.1f')
  x,y  = m(mloc[:,:,0],mloc[:,:,1])
  cs   = m.contourf(x,y, ldaps, clev, cmap=cms)
  cbar = m.colorbar(cs, location='right', pad="5%")
  cbar.set_ticks(ticks)

  ax           = fig.add_subplot(142)
  ax.set_title('STDL %s %s'%(var, date))
  m = Basemap(projection='lcc', lat_1=30., lat_2=60., lon_0=126.0, lat_0=38.0,
            llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=rlon,urcrnrlat=rlat,resolution='h')
  m.drawcoastlines()
  m.drawmapboundary()
  m.drawparallels(np.arange(llat, rlat, 2.), labels=[1,0,0,0], linewidth=0.1, fmt='%.1f')
  m.drawmeridians(np.arange(llon, rlon, 5.), labels=[0,0,0,1], linewidth=0.1, fmt='%.1f')
  x,y  = m(mloc[:,:,0],mloc[:,:,1])
  cs   = m.contourf(x,y,stdl, clev, cmap=cms)
  cbar = m.colorbar(cs, location='right', pad="5%")
  cbar.set_ticks(ticks)

  ax           = fig.add_subplot(143)
  ax.set_title('STDL & LDAPS %s %s'%(var, date))
  m = Basemap(projection='lcc', lat_1=30., lat_2=60., lon_0=126.0, lat_0=38.0,
            llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=rlon,urcrnrlat=rlat,resolution='h')
  m.drawcoastlines()
  m.drawmapboundary()
  m.drawparallels(np.arange(llat, rlat, 2.), labels=[1,0,0,0], linewidth=0.1, fmt='%.1f')
  m.drawmeridians(np.arange(llon, rlon, 5.), labels=[0,0,0,1], linewidth=0.1, fmt='%.1f')
  x,y  = m(mloc[:,:,0],mloc[:,:,1])
  cs   = m.contourf(x,y,stdl_ldaps, clev, cmap=cms)
  cbar = m.colorbar(cs, location='right', pad="5%")
  cbar.set_ticks(ticks)

  ax           = fig.add_subplot(144)
  ax.set_title('DIFF(STDL - LDAPS) %s %s'%(var, date))
  m = Basemap(projection='lcc', lat_1=30., lat_2=60., lon_0=126.0, lat_0=38.0,
            llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=rlon,urcrnrlat=rlat,resolution='h')
  m.drawcoastlines()
  m.drawmapboundary()
  m.drawparallels(np.arange(llat, rlat, 2.), labels=[1,0,0,0], linewidth=0.1, fmt='%.1f')
  m.drawmeridians(np.arange(llon, rlon, 5.), labels=[0,0,0,1], linewidth=0.1, fmt='%.1f')
  x,y  = m(mloc[:,:,0],mloc[:,:,1])
  cs   = m.pcolormesh(x,y,diff , cmap=plt.cm.seismic,norm=MidpointNormalize(midpoint=0.), vmin=np.min(diff), vmax=np.max(diff))
  cbar = m.colorbar(cs, location='right', pad="5%")
  cbar.set_ticks(ticks)

  plt.tight_layout()
  plt.savefig('%s/%s_%s.png'%(save_dir,var,date),bbox_inches='tight')
  plt.close()


def Make_Image(args):
  print("IMAGE MAIKING IS STARTED")
  var              = args.var
  stdl_dir         = '%s/all/'%(args.outf)
  stdl_ldaps_dir   = '%s/cat/'%(args.outf)
  ldaps_dir        = args.ldapsdir
  mloc             = np.load('%s/grid_info1km.npy'%(args.dainf))
  save_dir         = '%s/image/'%(args.outf)
  stime, etime     = args.img_sdate, args.img_edate
  fmt              = args.fmt

  if os.path.exists(save_dir):
    pass
  else:
    os.mkdir(save_dir)
  sdate = datetime.strptime(stime,fmt)
  edate = datetime.strptime(etime,fmt)
  now = sdate
  while now <= edate:
    print(now)
    dtime = now.strftime(fmt)
    comparision(var, ldaps_dir, stdl_ldaps_dir, stdl_dir, mloc, dtime, save_dir)
    now = now + timedelta(hours = 3)

