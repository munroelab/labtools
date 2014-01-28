import matplotlib.pyplot as plt
import numpy as np
import netCDF4
from skimage import filter
from skimage.morphology import disk

nc = netCDF4.Dataset('/Volumes/HD4/dz/248/dz.nc')

A = nc.variables['dz_array'][80,:,:]

min_max = 0.1
# clip
#A[A>min_max] = min_max
#A[A<-min_max] = -min_max

plt.figure()
plt.imshow(A,
        vmin=-min_max/10.0,
        vmax=min_max/10.0,
        interpolation='nearest',
        aspect='auto')
plt.colorbar()

print "max", A.max()
print "min",A.min()
print A

# mask the data
maskA = np.ma.masked_array(A, abs(A)<0.001)

# scale between -1 / +1
#A = A / min_max

# scale the number into an integer between 0 and 255
A +=1.0 #shift
A *=128.0 # scale

print disk(0)

Afilt= filter.rank.mean(np.uint16(A), 
             disk(0),
             #mask = A.mask,
             )

#Afilt = Afilt/255.0  

print "max", A.max()
print "min",A.min()
print A
print "max", Afilt.max()
print "min",Afilt.min()
print Afilt
plt.figure()
plt.subplot(2,1,1)
plt.imshow(A,
        vmin=A.min(),
        vmax=A.max(),
        interpolation='nearest',
        aspect='auto')
plt.colorbar()

plt.subplot(2,1,2)
plt.imshow(Afilt,
        vmin=Afilt.min(),
        vmax=Afilt.max(),
        interpolation='nearest',
        aspect='auto')
plt.colorbar()

#shift back
A =  A/128.0 -1.0
Afilt = Afilt/128.0 -1.0

print "max", A.max()
print "min",A.min()
print A
print "max", Afilt.max()
print "min",Afilt.min()
print Afilt
plt.figure()
plt.subplot(2,1,1)
plt.imshow(A,
        vmin=-min_max/10.0,
        vmax=min_max/10.0,
        interpolation='nearest',
        aspect='auto')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(Afilt,
        vmin=-min_max/10.0,
        vmax=min_max/10.0,
        interpolation='nearest',
        aspect='auto')
plt.colorbar()

plt.show()
