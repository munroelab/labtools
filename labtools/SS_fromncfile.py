"""
test program for implementing line based synthetic schlieren 
using matrix based operations
"""
import spectrum_test
import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
from matplotlib import pyplot as plt
import argparse
from PIL import Image
import pylab
import numpy
import time
import os
import labdb
import netCDF4
from scipy import ndimage

def getTol(image, mintol = 10):
    """
    estimate monotonicity of data
    """

    nrows, ncols = image.shape
    # work with float point arrays
    image = image * 1.0

    # compute diference between rows
    d = image[:-1,:] - image[1:,:]
    z = numpy.zeros(ncols)
    # construct array of difference between current and upper
    A = numpy.vstack((d, z))
    # and between current and lower row
    B = numpy.vstack((z, d))
    # only estimate if the sequence is monotonic and differences
    # are at least mintol 
    C = (abs(A) >= mintol) & (abs(B) >= mintol) & (A*B>0)
    return C

def compute_dz_image(im1, im2, dz = 1.0):
    """
    Estimate dz given two images im1 and im2 (where im1 is the reference image)
    """

    # make things all float
    im1 = im1 * 1.0
    im2 = im2 * 1.0

    # assert img.shape = img0.shape
    N, M = im1.shape
    zerorow = numpy.zeros(M)

    A = im2[:] - im1[:] 
    #   0,1,2,N-1   0,1,2,3, N-1

    B = im2[:-1] - im1[1:]
    B = numpy.vstack((B, zerorow))

    #     1,2,3,..N-1   -   0,1,2,..N-2
    C = im2[1:] - im1[:-1]
    C = numpy.vstack((zerorow, C))

    D = im1[:-1] - im1[1:]
    D = numpy.vstack((zerorow, D))

    E = im1[:-2] - im1[2:]
    E = numpy.vstack((zerorow, E, zerorow))


    F = im1[1:] - im1[:-1]
    F = numpy.vstack((F, zerorow))

    """ans1 = - dz * (A*B)/(D*E)  ans2 = - dz * (A*C)/(F*E) ans = ans1 + ans2"""

    ans = -dz * A/E * (B/D + C/F)
    return ans

def compute_dz(video_id,col,min_tol,sigma,filter,max_min,override_MS=0,overridef=0):
    
    """
    > Given video_id, calculate the dz array. Output is cached on disk.
    > returns the array dz
    > skip_frames is the number of frames to jump before computing dz
    """
    diff_frames=3
    db = labdb.LabDB()
    
    # n: number of frames between 2 images we subtract 
    n=diff_frames
    # count: start from the second frame. count is the variable that tracks the
    # current frame
    count=0
    # load the video nc file and get the image column array to compute the dz 
    video_filename = '/Volumes/HD4/videoncfiles/%d/video.nc' % video_id
    print "video filename : ", video_filename

    nc=netCDF4.Dataset(video_filename,'r')
    image = nc.variables['img_array'][:,:,col]
    nt,nz = image.shape
    image = image.reshape((nt,nz,1))
    print image.shape
    z = nc.variables['row']
    x = nc.variables['column'][col]
    t = nc.variables['time']
    stopF = t.shape[0]-diff_frames
    
    dz = 50.0 / 964
    
    from scipy.ndimage import gaussian_filter
    from numpy import ma
    dz_array=[]
    # while True
    while (count <stopF):
        delz = compute_dz_image(image[count], image[count+diff_frames], dz) 
        #delz = image[count]
        print "count", count        
        C = getTol(image[count],mintol = min_tol)
        if (override_MS==0):
            delz = numpy.nan_to_num(delz) * C
        #delz_m = ma.masked_where(C==False, delz)
        #dz_array = generate_dz_array(delz,dz_array)
        # clip large values
            bound = 1.0
            delz[delz > bound] = bound
            delz[delz < -bound] = -bound

        if (override_MS==0):
            # fill in missing values
            filt_delz = ndimage.gaussian_filter(delz, (sigma,1))
            #i = abs(delz) > 1e-8
            delz = C*delz+ (1-C)*filt_delz

            # smooth
            #filt_delz = ndimage.gaussian_filter(filt_delz, 7)
        
        if (overridef ==0):
            # new smoothing
            delz =ndimage.uniform_filter(delz,size=(filter,filter))

        dz_array.append(delz)
        count=count+1
        
    
    dz_array = numpy.array(dz_array)
    print "dz shape: ", dz_array.shape
    count=0
    timeFdz=[]
    print "z shape", z.shape[0]
    while (count < z.shape[0]):
        print "count :" , count
        xx = ndimage.uniform_filter(dz_array[:,count,:],size=(6,1))
        count+=1
        timeFdz.append(xx)

    timeFdz = numpy.array(timeFdz)
    print "timeFdz.shape",timeFdz.shape

    plt.figure(figsize=(20,11))
    ax = plt.subplot(3,1,1)
    plt.imshow(dz_array[:,:,0].T,extent=[t[0],t[-1],z[-1],z[0]],vmax=max_min,\
                    vmin=-max_min,aspect='auto',interpolation='nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.colorbar()

    plt.subplot(3,1,2, sharex=ax, sharey=ax)
    plt.imshow(timeFdz[:,:,0],extent=[t[0],t[-1],z[-1],z[0]],vmax=max_min,\
                    vmin=-max_min,aspect='auto',interpolation='nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.colorbar()

    plt.subplot(3,1,3, sharex=ax, sharey=ax)
    plt.imshow(image[:,:,0].T,extent=[t[0],t[-1],z[-1],z[0]],vmax=255,\
                    vmin=0,aspect='auto',interpolation='nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.colorbar()

    plt.show()


def UI(): 
    
    """
    take arguments from the user :video id and skip frame number and call
    compute dz function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id",type = int, 
                        help = "Enter the video id of the frames on which to do Synthetic Schlieren")
    parser.add_argument("column",type = int, help = "Enter the column")
    parser.add_argument("max_min",type = float, help = "Enter the max_min \
            for the colorbar")
    parser.add_argument("mintol",type = int, help=" Helps estimate the monoticity of the data")
    parser.add_argument("sigma", type = float, help= "standard deviation for the Gaussian kernal")
    parser.add_argument("filter", type = int, help = "filter size")
    parser.add_argument("--override_MS",type=int, default = "0",\
            help="if 1 mintol is not applied to the raw dz")
    parser.add_argument("--overridef",type=int, default = "0",\
            help="if 1 uniform filter is not applied to the raw dz")
    args = parser.parse_args()
    dz_id=compute_dz(args.video_id,args.column,args.mintol,args.sigma,args.filter,args.max_min\
            ,args.override_MS,args.overridef)

if __name__ == "__main__":
    UI()
