import matplotlib.pyplot as plt
import numpy as np
import netCDF4
from skimage import filter
from skimage.morphology import disk

nc = netCDF4.Dataset('/Volumes/HD4/dz/248/dz.nc')

A = nc.variables['dz_array'][50, :, :]


min_max = 0.01

# clip
A[A>min_max] = min_max
A[A<-min_max] = -min_max

# scale between -1 / +1
A = A / min_max

# mask the data
#A = np.ma.masked_array(A, abs(A)<0.001)

print disk(0)
Afilt = filter.rank.mean(A, 
             disk(0),
             #mask = A.mask,
             )

Afilt = Afilt/256.0  * 2 - 1

plt.subplot(2,1,1)
plt.imshow(A,
        vmin = -1,
        vmax = 1,
        interpolation='nearest',
        aspect='auto')
plt.colorbar()

plt.subplot(2,1,2)
plt.imshow(Afilt,
        vmin = -1,
        vmax = 1,
        interpolation='nearest',
        aspect='auto')
plt.colorbar()

plt.show()
