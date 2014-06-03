# explaining every step of SyntheticSchlieren.  To understand the purpose of every single step and view the output of every step.


import Image
import numpy as np
import matplotlib.pyplot as plt
import SyntheticSchlieren as SS
import skimage.morphology, skimage.filter
from scipy import ndimage
import netCDF4 as nc

# take a look at a time series of any single vertical column of the video
im = Image.open('/Users/prajvala/Documents/Project_labtools/labtools/plots/vts757.png')
im = np.array(im)
print im.shape
# plt.figure(1)
# ax = plt.subplot(2,2,1)
# plt.imshow(im,
#            interpolation='nearest')
# plt.title('vertical time series')
# plt.colorbar()

# constants used in computing Synthetic Schlieren
min_tol = 7  # small mintol means you have more regions that are returned by getTOL function
sigma = 8
dz = 56.0/964
nz, nt = im.shape
disk_size = 10 # it will make a 9 by 9 array with a circular disk of 1's
plot_maxmin = 0.01

# getTOL returns a matrix that helps us capture only the motion that is monotonic in time.
# we only want to capture the motions of the black and white lines as that represents propagating internal waves.
C = SS.newGetTol(im, mintol = min_tol)
# take a look at the matrix returned by getTOL
#plt.figure()
# plt.subplot(2,2,2,sharex=ax, sharey=ax)
# plt.imshow(C,interpolation='nearest',
#            )
# plt.title('mintol')
# plt.colorbar()

delz = np.zeros((nz,nt),dtype=np.float32)
# the compute_dz_image function return a dz array interspersed with nan's
for i in range(1,nt):
    array1 = im[:,i-1].reshape((-1,1))
    array2 = im[:,i].reshape((-1,1))

    #print array1.shape
    temp1 = SS.compute_dz_image(array1,
                                array2,
                                dz)
    delz[:,i] = temp1.reshape((964,))
# take a look at delz with nan's
#plt.figure()
# plt.subplot(2,2,3, sharex=ax, sharey=ax)
# plt.imshow(delz,
#            vmax=plot_maxmin,vmin=-plot_maxmin,interpolation='nearest',
#
#             )
# plt.title('raw delz')
# plt.colorbar()

# remove the nan's
delz = np.nan_to_num(delz)
# take a look at delz without nan's
# plt.figure()
# plt.imshow(delz,
#            vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',
#            )
# plt.title('delz after removing nan')
# plt.colorbar()
# multiplying delz with C gives us delz that has only monotonically increasing motion which is what we want
delz = delz * C
# take a look at delz now! vmin and vmax are used to show you the actual dz values which are quite small

# plt.figure()
# plt.subplot(2,2,4, sharex=ax, sharey=ax)
# plt.imshow(delz,
#            vmax=plot_maxmin,vmin=-plot_maxmin,interpolation='nearest',
#           )
# plt.title('delz - monotonically increasing')
# plt.colorbar()
# you'll notice that the values are very very high and that is because the function nan_to_num()
# replaces nan's with zeros and -infinity with very small values and +infinity with very large values
# So now we need to cut the very large and very small values off

# implementing the skimage.filter.mean so as to apply mean filter on a
# masked array and then applying a gaussian filter to smooth the image

# step 1: clip the large values
min_max = 0.03 # 0.05 cm dz per pixel shift is plenty!
clip_min_max = 0.95 * min_max # actual clipping value is 0.0475
delz[delz > clip_min_max] = clip_min_max # clipping the very large values
delz[delz < -clip_min_max] = -clip_min_max # clipping off the very small values
# take a look at the clipped delz
# plt.figure()
# plt.imshow(delz,interpolation='nearest')
#
# plt.title('clipped delz')
# plt.colorbar()
# Step 2 : map the original data from -0.05 to +0.05 to range from 0 to 255
check = np.array((delz + min_max/(2.0 *min_max)*256) ,dtype=np.uint8)
check1 = np.array(((delz + min_max)/(2.0 *min_max)*256),dtype=np.uint8)
print check[700:750,570:580], "\n######\n", check1[700:750,570:580]

# just found a bug in SyntheticSchlieren!!! previously the expression was delz + min_max/(.5 *min_max)*256 instead of being
#(delz + min_max)/(.5 *min_max)*256). In the first case the mapped_delz has only 0 and 255 instead of having a
# range of numbers
mapped_delz = np.uint8((delz + min_max)/ (2.0* min_max) * 256)
# take a look at mapped_delz
#print mapped_delz[700:750,570:580]
# plt.figure(3)
# ax  = plt.subplot(2,1,1)
# plt.imshow(mapped_delz,interpolation='nearest')
# plt.title('mapped delz')
# plt.colorbar()
# Step 3 : prepare a mask:: 1 means use the data and 0 means ignore the
# data here within the disk
# The mask is 0 wherever the data is bad and 1 wherever the data is to be considered
mask_delz = np.uint8(mapped_delz <>128)
# take a look at the mask
# plt.figure()
# plt.imshow(mask_delz)
# plt.title('mask_delz')
# plt.colorbar()
#Step 4: apply the mean filter to compute values for the masked pixels
print "delz.shape", delz.shape
row_disk = np.ones((disk_size,1))

filtmapped_delz = np.ones((nz,nt))
# This is better than applying a spatial filter in x and z .. atleast till we verify that it makes no significant difference
#for i in range(0,nt-1):
#    mdelz = np.reshape(mapped_delz[:,i], (nz,1))
filt_delz = skimage.filter.rank.mean(mapped_delz,
                #skimage.morphology.disk(disk_size),
                row_disk,
                #mask = np.reshape(mask_delz[:,i],(nz,1)),
                mask=mask_delz,
                )
    #filtmapped_delz[:,i] = filt_delz[:,0]
filtmapped_delz = filt_delz
#setting the zeros in filtmapped_delz to 128
filtmapped_delz[filtmapped_delz ==0 ] = 128
#print skimage.morphology.disk(disk_size)
# take a look at the mask
# plt.figure(3)
# plt.subplot(2,1,2,sharex = ax, sharey=ax)
# plt.imshow(filtmapped_delz,interpolation='nearest')
# plt.title('filt_delz' )
# plt.colorbar()

# Step 5: mapping back the values from 0 to 255 to its original values of
# -0.05 to 0.05
filtered_delz = (filtmapped_delz / 256.0) * (2.0 * min_max) - min_max
# take a look at the remapped delz
# plt.figure()
# ax = plt.subplot(2,2,1)
# plt.imshow(filtered_delz, vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest')
# plt.title('remapped_delz' )
# plt.colorbar()

# Step 6: Replacing the elements that were already right in the beginning
filled_delz = (1-mask_delz) * filtered_delz + mask_delz * delz
# take a look at the filled delz
# plt.subplot(2,2,2,sharex = ax ,sharey = ax)
# plt.imshow(filled_delz, vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest')
# plt.title('filled_delz' )
# plt.colorbar()

smooth_filt_delz = skimage.filter.gaussian_filter(filled_delz,
            [sigma,1])

# take a look at the filled delz
# plt.subplot(2,2,3,sharex = ax,sharey = ax)
# plt.imshow(smooth_filt_delz,
#           vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest'
#           )
# plt.title('smoothed filled_delz')
# plt.colorbar()
final_dz = ndimage.uniform_filter(smooth_filt_delz,
                                    size=(1,6),
                                    )
# plt.subplot(2,2,4,sharex = ax,sharey = ax)
# plt.imshow(final_dz,
#           vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest'
#           )
# plt.title(' final_dz')
# plt.colorbar()


### applying HT for this column


datain = final_dz[:, :]

# take FFT in time
U_spectrum = np.fft.fft(datain, axis=1)
freq = np.fft.fftfreq(nt,d = 1.0/6.2)
print "@@@",len(freq),len(U_spectrum[400,:])

print freq
plt.figure()

plt.plot(np.fft.fftshift(freq),np.fft.fftshift(U_spectrum[400,:].real))


# only keep positive frequencies
#   could extend this to a band pass filter

# explicitly set all non-positive frequencies to zero
U_spectrum[:, nt/2:] = 0

# take inverse FFT and multiply by 2
datac = 2 * np.fft.ifft(U_spectrum, axis=1)


datain = datac[:, :]

# fft
datac_spectrum = np.fft.fft(datain, axis=0)
freq  = np.fft.fftshift(np.fft.fftfreq(964,d = 56.0/964))
print "@@@",len(freq),len(datac_spectrum[:,1000])
plt.figure()
plt.plot(freq, np.fft.fftshift(datac_spectrum[:,300]))

# make a copy
datac_R_spectrum = datac_spectrum.copy()
datac_L_spectrum = datac_spectrum # note: no copy here

# only include right ward propagating (kx > 0)
datac_R_spectrum[:nz/2, :] = 0

# only include left ward propagating (kx < 0)
datac_L_spectrum[nz/2:, :] = 0

# inverse FFT
datac_R = np.fft.ifft(datac_R_spectrum, axis=0)
datac_L = np.fft.ifft(datac_L_spectrum, axis=0)

plt.figure()
ax = plt.subplot(3,1,1)
plt.imshow(final_dz,vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',aspect='auto')
plt.title(' final_dz')
plt.colorbar()

plt.subplot(3,1,2,sharex = ax , sharey=ax)
plt.imshow(datac_R.real,vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',aspect='auto')
plt.title('upward')
plt.colorbar()

plt.subplot(3,1,3,sharex = ax , sharey=ax)
plt.imshow(datac_L.real,vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',aspect='auto')
plt.title('downward')
plt.colorbar()


# demonstrate that we can separate the waves moving left and right from a horizontal
plot_maxmin = 0.1
ncdata = '/Volumes/HD4/dz/32/dz.nc'
dz = nc.Dataset(ncdata,'r')
dzhts = dz.variables['dz_array'][:,320,40:1240]
print dzhts.shape
nt,nx = dzhts.shape

plt.figure()
plt.subplot(2,1,1)
plt.imshow(dzhts,interpolation='nearest',aspect = 'auto')
plt.subplot(2,1,2)
plt.imshow(dzhts.T[::-1,:],interpolation='nearest',aspect = 'auto')
plt.colorbar()

# hilbert transform
# step 1 : remove negative frequencies
hts = dzhts.T[::-1,:]
# take FFT in time
U_spectrum = np.fft.fft(hts, axis=1)

# only keep positive frequencies
#   could extend this to a band pass filter

# explicitly set all non-positive frequencies to zero
U_spectrum[:,nt/2:] = 0

# take inverse FFT and multiply by 2
datac = 2 * np.fft.ifft(U_spectrum, axis=1)
print datac.shape

# fft
datac_spectrum = np.fft.fft(datac, axis=0)
freq = np.fft.fftshift(np.fft.fftfreq(1200,d=73.0/1292))
print "@@@", len(freq),len(U_spectrum[:,500])
plt.figure()
plt.plot(freq,np.fft.fftshift(U_spectrum[:,100]))

# make a copy
datac_R_spectrum = datac_spectrum.copy()
datac_L_spectrum = datac_spectrum # note: no copy here

# only include right ward propagating (kx > 0)
datac_R_spectrum[nx/2:,:] = 0

# only include left ward propagating (kx < 0)
datac_L_spectrum[:nx/2,:] = 0

# inverse FFT
datac_R = np.fft.ifft(datac_R_spectrum, axis=0)
datac_L = np.fft.ifft(datac_L_spectrum, axis=0)

plt.figure()
ax = plt.subplot(3,1,1)
plt.imshow(hts,vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',aspect='auto')
plt.title(' final_dz')
plt.colorbar()

plt.subplot(3,1,2,sharex = ax , sharey=ax)
plt.imshow(datac_R.real,vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',aspect='auto')
plt.title('right')
plt.colorbar()

plt.subplot(3,1,3,sharex = ax , sharey=ax)
plt.imshow(datac_L.real,vmin = -plot_maxmin,vmax = plot_maxmin,interpolation='nearest',aspect='auto')
plt.title('left')
plt.colorbar()

plt.show()
