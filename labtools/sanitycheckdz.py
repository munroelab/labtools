# explaining every step of SyntheticSchlieren.  To understand the purpose of every single step and view the output of every step.


import Image
import numpy
import matplotlib.pyplot as plt
import SyntheticSchlieren as SS
import skimage.morphology, skimage.filter
# take a look at a time series of any single vertical column of the video
im = Image.open('/Users/prajvala/Documents/Project_labtools/labtools/plots/vtsVID764.png')
im = numpy.array(im)
plt.figure()
plt.imshow(im)
plt.colorbar()

# constants used in computing Synthetic Schlieren
min_tol = 3  # small mintol means you have more regions that are returned by getTOL function
sigma = 11
filter_size=10
video_id = 764
dz = 58.0/964
nz, nt = im.shape
disk_size = 4 # it will make a 9 by 9 array with a circular disk of 1's

# getTOL returns a matrix that helps us capture only the motion that is monotonic in time.
# we only want to capture the motions of the black and white lines as that represents propagating internal waves.
C = SS.getTol(im, mintol = min_tol)
# take a look at the matrix returned by getTOL
plt.figure()
plt.imshow(C)
plt.colorbar()

delz = numpy.zeros((nz,nt),dtype=numpy.float32)
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
plt.figure()
plt.imshow(delz,
           vmax=0.02,vmin=-0.02,
            )
plt.colorbar()

# multiplying delz with C gives us delz that has only monotonically increasing motion which is what we want
delz = delz * C
# take a look at delz now! vmin and vmax are used to show you the actual dz values which are quite small

plt.figure()
plt.imshow(delz,vmax=0.02,vmin=-0.02)
plt.colorbar()
# remove the nan's
delz = numpy.nan_to_num(delz)
# take a look at delz without nan's
plt.figure()
plt.imshow(delz)
plt.colorbar()
# you'll notice that the values are very very high and that is because the function nan_to_num()
# replaces nan's with zeros and -infinity with very small values and +infinity with very large values
# So now we need to cut the very large and very small values off

# implementing the skimage.filter.mean so as to apply mean filter on a
# masked array and then applying a gaussian filter to smooth the image

# step 1: clip the large values
min_max = 0.05 # 0.05 cm dz per pixel shift is plenty!
clip_min_max = 0.95 * min_max # actual clipping value is 0.0475
delz[delz > clip_min_max] = clip_min_max # clipping the very large values
delz[delz < -clip_min_max] = -clip_min_max # clipping off the very small values
# take a look at the clipped delz
plt.figure()
plt.imshow(delz)
plt.title('clipped delz')
plt.colorbar()
# Step 2 : map the original data from -0.05 to +0.05 to range from 0 to 255
check = numpy.array((delz + min_max/(.5 *min_max)*256) ,dtype=numpy.uint8)
check1 = numpy.array(((delz + min_max)/(.5 *min_max)*256),dtype=numpy.uint8)
print check[700:750,570:580], "\n######\n", check1[700:750,570:580]

# just found a bug in SyntheticSchlieren!!! previously the expression was delz + min_max/(.5 *min_max)*256 instead of being
#(delz + min_max)/(.5 *min_max)*256). In the first case the mapped_delz has only 0 and 255 instead of having a
# range of numbers
mapped_delz = numpy.uint8((delz + min_max)/ (0.5 * min_max) * 256)
# take a look at mapped_delz
#print mapped_delz[700:750,570:580]
plt.figure()
plt.imshow(mapped_delz)
plt.colorbar()
# Step 3 : prepare a mask:: 1 means use the data and 0 means ignore the
# data here within the disk
# The mask is 0 wherever the data is bad and 255 wherever the data is to be considered
mask_delz = numpy.uint8(mapped_delz <>128)
# take a look at the mask
plt.figure()
plt.imshow(mask_delz)
plt.colorbar()

#Step 4: apply the mean filter to compute values for the masked pixels
disk_size = 4
filt_delz = skimage.filter.rank.mean(mapped_delz,
            skimage.morphology.disk(disk_size),
            mask = mask_delz,
            )
print skimage.morphology.disk(disk_size)
# take a look at the mask
plt.figure()
plt.imshow(filt_delz)
plt.colorbar()
plt.show()
