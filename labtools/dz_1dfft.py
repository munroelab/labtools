import pylab
import argparse
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

data = nc.Dataset('/Volumes/HD3/dz/27/dz.nc')
dz = data.variables['dz_array']
t = data.variables['time']
x = data.variables['column']
z = data.variables['row']

t = np.float16(t)
x = np.float16(x)
z = np.float16(z)

print(dz.shape)

fig = plt.figure()

"""
omega calculations
"""
dzfft = np.fft.fft(dz[113:342,120,700])
dzfft = np.fft.fftshift(dzfft)
tseries= t[113:342]
freq = np.fft.fftfreq(tseries.shape[0],d=0.29)
print("dt  : " ,np.mean(np.diff(t)))
freq = np.fft.fftshift(freq)

a1 = fig.add_subplot(121)
plt.plot(tseries,dz[113:342,120,700])
plt.xlabel('time')
a2 = fig.add_subplot(122)
plt.plot(freq ,abs(dzfft),'b:')
plt.xlabel('frequency')

"""
kx calculation
"""
dz_kx = np.fft.fft(dz[300,700,250:1200])
dz_kx = np.fft.fftshift(dz_kx)
x= x[250:1200]
print("dx : ", np.mean(np.diff(x)))   
dz_xwavenum = np.fft.fftfreq(x.shape[0] , d = 0.0356)
dz_xwavenum = np.fft.fftshift(dz_xwavenum)

fig = plt.figure()
b1 = fig.add_subplot(121)
plt.plot(x,dz[300,700,250:1200])
plt.xlabel('length')
b2 = fig.add_subplot(122)
plt.plot(dz_xwavenum, abs(dz_kx),'b:')
plt.xlabel('kx/(2*pi)')

"""
kz calculation
"""
dz_kz = np.fft.fft(dz[150,60:900,1200])
dz_kz = np.fft.fftshift(dz_kz)
z=z[60:900]
print("dz : ", np.mean(np.diff(z)))
dz_zwavenum = np.fft.fftfreq(z.shape[0] , d = 0.0363)
dz_zwavenum = np.fft.fftshift(dz_zwavenum)

fig = plt.figure()
c1 = fig.add_subplot(121)
plt.plot(z,dz[150,60:900,1200])
plt.plot(z,dz[165,60:900,1200])

plt.xlabel('depth')

c2 = fig.add_subplot(122)
plt.plot(dz_zwavenum, abs(dz_kz),'b:')
plt.xlabel('kz/(2*pi)')

i= np.argmax(abs(dzfft))
print("freq : " , freq[i])
i= np.argmax(abs(dz_kx))
print("kx : ", dz_xwavenum[i])
i= np.argmax(abs(dz_kz))
print("kz : ", dz_zwavenum[i])
plt.show()



def UI():
    """
    take arguments from the user: dz_id, t,z,x
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dz_id", type  = int , help = "Enter the dz_id ")



