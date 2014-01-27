"""
test program for implementing line based synthetic schlieren 
using matrix based operations
"""
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import argparse
import Image
import pylab
import numpy
import time
import os
import labdb
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

    A = im2[:,:] - im1[:,:] 
    #   0,1,2,N-1   0,1,2,3, N-1

    B = im2[:-1,:] - im1[1:,:]
    B = numpy.vstack((B, zerorow))

    #     1,2,3,..N-1   -   0,1,2,..N-2
    C = im2[1:,:] - im1[:-1,:]
    C = numpy.vstack((zerorow, C))

    D = im1[:-1,:] - im1[1:,:]
    D = numpy.vstack((zerorow, D))

    E = im1[:-2,:] - im1[2:,:]
    E = numpy.vstack((zerorow, E, zerorow))

    F = im1[1:,:] - im1[:-1,:]
    F = numpy.vstack((F, zerorow))
    
    """ans1 = - dz * (A*B)/(D*E)  ans2 = - dz * (A*C)/(F*E) ans = ans1 + ans2"""

    ans = -dz * A/E * (B/D + C/F)
    return ans


def compute_dz(video_id,min_tol,sigma,filter_size,image1,image2,win_h):
    """
    > Given video_id, calculate the dz array. Output is cached on disk.
    > returns the array dz
    > skip_frames is the number of frames to jump before computing dz
    """
    db = labdb.LabDB()
    
    

    # Set path to the two images
    im1 = "/Volumes/HD3/video_data/%d/frame%05d.png" % (video_id,image1)
    im2 = "/Volumes/HD3/video_data/%d/frame%05d.png" % (video_id,image2)
    print "im1 =",im1
    print "im2= " ,im2

    dz = win_h / 964
    
    #mintol =10
   

    from scipy.ndimage import gaussian_filter
    from numpy import ma
    

    image1 = numpy.array(Image.open(im1))
    image2 = numpy.array(Image.open(im2))
    print "image1.shape", image1.shape
    print "image2.shape", image2.shape
    C = getTol(image1, mintol = min_tol)
    delz = compute_dz_image(image1, image2, dz) 
    delz = numpy.nan_to_num(delz) * C
    bound = 1.0
    delz[delz > bound] = bound
    delz[delz < -bound] = -bound

    # fill in missing values
    #filt_delz = ndimage.gaussian_filter(delz, (7.5,7.5))
    filt_delz = ndimage.gaussian_filter(delz, (sigma,sigma))
    #i = abs(delz) > 1e-8
    filt_delz = C*delz+ (1-C)*filt_delz

    # smooth
    #filt_delz = ndimage.gaussian_filter(filt_delz, 7)
        
    # new smoothing
    smooth_filt_delz =ndimage.uniform_filter(filt_delz,size=(filter_size,filter_size))
    
    plt.figure(1)
    plt.imshow(delz[213::,:],vmax=0.01,vmin=-0.01)
    plt.colorbar()
    plt.title('raw dZ')

    plt.figure(2)
    plt.imshow(filt_delz[213::,:],vmax=0.01,vmin=-0.01)
    plt.colorbar()
    plt.title('after smoothing (no uniform filter)')

    plt.figure(3)
    plt.imshow(smooth_filt_delz[213::,:])
    plt.title('smoothed and uniform filtered')
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
    
    parser.add_argument("mintol",type = int, help=" Helps estimate the monoticity of the data")
    parser.add_argument("sigma", type = float, help= "standard deviation for the Gaussian kernal")
    parser.add_argument("filter_size", type = int, help = "filter size")
    parser.add_argument("image1",type= int, help="enter the image number of \
            IM1")
    parser.add_argument("image2",type= int, help= "enter the image number of \
            IM2")
    parser.add_argument("win_h",type = float, help = "window height")
    args = parser.parse_args()
    dz_id=compute_dz(args.video_id,args.mintol,args.sigma,args.filter_size,\
            args.image1,args.image2,args.win_h)

if __name__ == "__main__":
    UI()
