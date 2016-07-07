"""
test program for implementing line based synthetic schlieren 
using matrix based operations
"""
"""
Some details of Synthetic Schlieren Settings:
if diff_frames = None
then the Synthetic Schlieren algorithm uses the first image as the reference
image from which all the future images are subtracted from.



"""

import spectrum_test
import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import argparse
from PIL import Image
import pylab
import numpy
import time
import os
import labdb
import gc
import netCDF4
from scipy import ndimage
import progressbar
import multiprocessing
import socket
import warnings

import skimage.morphology, skimage.filter 

from matplotlib import animation

# to eliminate DivideByZero warnings
warnings.filterwarnings("ignore")

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

    ans = -dz * A/E * (B/D + C/F)
    return ans


def create_nc_file(video_id,skip_frames,skip_row,skip_col,mintol,sigma,filter_size,
        startF,stopF,diff_frames):

    """ 
        return dt,dz,dx
        Need to compute dz for the first time.
    """
    
    db = labdb.LabDB()
    # Get experiment ID
    sql = """ SELECT expt_id FROM video_experiments WHERE video_id =  %d """ % video_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print " experiment ID : ", expt_id

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]*1.0
    
    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]*1.0
    
    print "lenght" , win_l, "\nheight", win_h
    print  "video_id,skip_frames,skip_row,skip_col,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames"
    print  video_id,skip_frames,skip_row,skip_col,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames
    
    # find the number for rows and column of the frames
    Nrow = 964/skip_row
    Ncol = 1292/skip_col
    print "no of rows : %d and no of columns : %d" %(Nrow,Ncol)

    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    t=numpy.loadtxt(path2time)
    dt = numpy.mean(numpy.diff(t[:,1]))
    
    # compute dt,dz,dx
    dt = dt*skip_frames
    dz = win_h / Nrow
    dx = win_l / Ncol
    #print "timestep: " ,dt
    #print "dz : ", dz
    #print "dx : ",dx 

    return dt,dz,dx


def schlieren_lines(p):
    """
    p is a dictionary containing parameters need to 
    compute synthetic schliren on a pair of images

    returns array
    """

    IM1 = numpy.array(Image.open(p['filename1']))
    
    #loading the array according to user specification 
    image1 = IM1[::p['skip_row'],::p['skip_col']]    
    
    IM2 = numpy.array(Image.open(p['filename2']))
    
    #loading the array according to user specification 
    image2 = IM2[::p['skip_row'],::p['skip_col']]
    
    #change the filter size accordingly
    filtersize_r = p['filter_size']//p['skip_row']
    filtersize_c =  p['filter_size']//p['skip_col']
    
    # plotting everything 
    """pylab.figure()
    pylab.subplot(2,1,1)
    pylab.imshow(IM1,aspect="auto")
    pylab.title("image 1")
    pylab.colorbar()
    pylab.subplot(2,1,2)
    pylab.imshow(IM2,aspect="auto")
    pylab.title("image 2")
    pylab.colorbar()
    """

    #filtersize_r = 1
    C = getTol(image1, mintol = p['min_tol'])
    delz = compute_dz_image(image1, image2, p['dz']) 
    delz = numpy.nan_to_num(delz) * C
    
    # returning here itself to see what the raw dz looks like before we apply
    # any filtering of any sort
   # return p['i'], delz
    
    # implementing the skimage.filter.mean so as to apply mean filter on a
    # masked array and then applying a gaussian filter to smooth the image

    # step 1: clip the large values
    min_max = 0.02
    clip_min_max = 0.95 * min_max
    delz[delz > clip_min_max] = clip_min_max
    delz[delz < -clip_min_max] = -clip_min_max
    # Step 2 : map the original data from -0.1 to +0.1 to range from 0 to 255
    mapped_delz = numpy.array( (delz + min_max)/(0.5 * min_max) * 256, dtype = numpy.uint8)
    
    # Step 3 : prepare a mask:: 1 means use the data and 0 means ignore the
    # data here within the disk
    mask_delz = numpy.uint8(mapped_delz <>128)
    
    #Step 4: apply the mean filter to compute values for the masked pixels 
    disk_size = 1
    filt_delz = skimage.filter.rank.mean(mapped_delz,
            skimage.morphology.disk(disk_size),
            mask = mask_delz,
            ) 
    # Step 5: mapping back the values from 0 to 255 to its original values of
    # -0.1 to 0.1
    filtered_delz = (filt_delz / 256.0) * (0.5 * min_max) - min_max 
    
    # Step 6: Replacing the elements that were already right in the beginning
    filled_delz = (1-mask_delz) * filtered_delz + mask_delz * delz
    
    # Step 7 : applying the Gaussian filter to do a spatial smoothing of the image
    smooth_filt_delz = skimage.filter.gaussian_filter(filled_delz, 
            [p['sigma'],p['sigma']])
    # plotting  delz , delz * C , mapped_delz, mask_delz, filt_delz,
    # filled_delz(before applying uniform filter)

    pylab.figure()
    pylab.subplot(2,2,1)
    pylab.imshow(mask_delz,aspect="auto")
    pylab.title("mintol")
    pylab.colorbar()
    pylab.subplot(2,2,2)
    pylab.imshow(delz,vmax = 0.001,vmin = -0.001,aspect="auto")
    pylab.title("delz")
    pylab.colorbar()
    pylab.subplot(2,2,3)
    pylab.imshow(mapped_delz,aspect="auto")
    pylab.title("mapped_delz")
    pylab.colorbar()
    pylab.subplot(2,2,4)
    pylab.imshow(filled_delz,vmax = 0.001,vmin = -0.001,aspect="auto")
    pylab.title("filled_delz")
    pylab.colorbar()


    return  smooth_filt_delz
    
    """
    old code below 
    """

    # fill in missing values
    filt_delz = ndimage.gaussian_filter(delz, (p['sigma'],p['sigma']))
    #i = abs(delz) > 1e-8
    filt_delz = C*delz+ (1-C)*filt_delz
    return filt_delz
    # smooth
    #filt_delz = ndimage.gaussian_filter(filt_delz, 7)
    
    # spatial smoothing along x and z axis
    #smooth_filt_delz=ndimage.uniform_filter(filt_delz,size=(filtersize_c,filtersize_r))
    smooth_filt_delz = filt_delz
    return smooth_filt_delz

def compute_dz(video_id,min_tol,sigma,filter_size,skip_frames=1,skip_row=1,skip_col=1,
            startF=0,stopF=0,diff_frames=1 ):
    """
    > returns the array dz
    > skip_frames is the number of frames to jump before computing dz
    """
    db = labdb.LabDB()

    # get the number of frames if stopF is unspecified
    if (stopF == 0):
        sql = """ SELECT num_frames FROM video WHERE video_id = %d""" % video_id
        rows = db.execute(sql)
        stopF = rows[0][0]
        print "stop_frames = ", stopF 
    num_frames=stopF-startF
    print "num_frames:" ,num_frames

    # Call the function that will create the nc file to append data to
    dt,dz,dx=create_nc_file(video_id,skip_frames,skip_row,skip_col,min_tol,\
            sigma,filter_size,startF,stopF,diff_frames)

    
    # count: start from the second frame. count is the variable that tracks the
    # current frame
    if diff_frames is None:
        count = startF
    else:
        count=startF+diff_frames

    # Set path to the two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
    # path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    
    from scipy.ndimage import gaussian_filter
    from numpy import ma

    #check if path exists
    filename2 = path % (video_id, count)

    p = {}
    p['filename1'] = None
    p['filename2'] = None
    p['skip_row'] = skip_row
    p['skip_col'] = skip_col
    p['filter_size'] = filter_size
    p['min_tol'] = min_tol
    p['dz'] = dz
    p['sigma'] = sigma
    print "the dictionary P" ,p

    i = 0

    # submit tasks to perform
    while os.path.exists(filename2) & (count <=stopF):
        if diff_frames is not None:
            ref_frame = count - diff_frames        
        else:
            ref_frame = startF

        filename1 = path % (video_id, ref_frame)
        filename2 = path % (video_id, count)
            
        if not os.path.exists(filename2):
            break

        # add filename1, filename2 to list of tasks
        p['filename1'] = filename1
        p['filename2'] = filename2
        p['i'] = i
        #tasks.append( (schlieren_lines, (dict(p),)))

        count += skip_frames
        filename2 = path % (video_id, count)

        smooth_filt_delz = schlieren_lines(p)
        pylab.figure()
        pylab.imshow(smooth_filt_delz,vmax = 0.002,vmin = -0.002,aspect="auto",
                interpolation = "nearest")
        pylab.colorbar() 
        pylab.show()
        i += 1
    



    #nc=netCDF4.Dataset(dz_filename,'a')
    #DZarray = nc.variables['dz_array']
    
    # define time axis for the nc time variable
    #tl = DZarray.shape[0]
    #nz = DZarray.shape[1]
    #nx = DZarray.shape[2]
    #print "no of timesteps:", tl
    #Tm=nc.variables['time']
    #count=0

    #print "DZarray::" ,DZarray.shape
    #print "time.shape(before):" ,Tm.shape
    #t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    #print "time:",t_array
    #Tm[:] = t_array
    
    #nc.close()

    return 


    # get information about the copied nc file to see if its chunked 
    print "ncdump %s" % dz_filename
    os.system('ncdump -h -s %s' %  dz_filename )

    # USE nccopy to rechunk Dz
    chunked_filename = 'chunked_dz.nc'
    cmd = 'nccopy -c time/%d,row/%d,column/%d %s %s' % (tl, nz, 1, dz_filename, chunked_filename) 
    print cmd
    os.system(cmd)

    # get information about the copied nc file to see if its chunked 
    print "ncdump %s" % chunked_filename
    os.system('ncdump -h -s %s' %  chunked_filename )

    nc=netCDF4.Dataset(chunked_filename,'a')
    DZarray = nc.variables['dz_array']

    ZZ = nc.variables['row'][:]
    CC = nc.variables['column'][:]

    # TRIAL :: apply uniform filter in the time axis with the filter size of 6 (about
    # 1second). This should smoothen the dz along time.
    col_count=0
    start=0
    #print "row shape: ",CC.shape
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=CC.size).start()
    for col_count in range(CC.size):
        pbar.update(col_count)

        temp1 = DZarray[:,:,col_count]
        DZarray[:,:,col_count] = ndimage.uniform_filter(temp1,size = (12,1))

    pbar.finish()
    
    # and rechunk back to ordered by frames
    os.system('nccopy -m 1G -h 2G -e 65001 -c time/%d,row/%d,column/%d %s %s' % (1, nz, nx,
        chunked_filename, dz_filename) )

    # get information about the copied nc file to see if its chunked correctly
    print "ncdump %s" % dz_filename
    os.system('ncdump -h -s %s' %  dz_filename )

    return dz_id

def test():
    # Need two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
    video_id = 50
    filename1 = path % (video_id, 500)
    filename2 = path % (video_id, 510)

    image1 = numpy.array(Image.open(filename1))
    image2 = numpy.array(Image.open(filename2))

    # disply both images
    pylab.figure()
    ax = pylab.subplot(231)
    pylab.imshow(image1, cmap=pylab.cm.jet,interpolation='nearest')
    pylab.title('Image1')
    pylab.colorbar()

    pylab.subplot(232, sharex=ax, sharey=ax)
    pylab.imshow(image2, cmap=pylab.cm.jet,interpolation='nearest')
    pylab.title('Image2')
    pylab.colorbar()

    pylab.subplot(233, sharex=ax, sharey=ax)
    pylab.imshow(image2*1.0-image1,interpolation='nearest')
    pylab.title('Difference')
    pylab.colorbar()

    pylab.subplot(234, sharex=ax, sharey=ax)
    C = getTol(image1, mintol = 0)
    pylab.imshow(C,interpolation='nearest')
    pylab.colorbar()
    pylab.title("valid")

    pylab.subplot(235, sharex=ax, sharey=ax)
    H = 52.0
    dz = H / image1.shape[0]
    delz = compute_dz_image(image1, image2, dz)
    vmax = 0.05
    pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax)
    pylab.colorbar()

    pylab.subplot(236, sharex=ax, sharey=ax)
    delz = numpy.nan_to_num(delz) * C
    vmax = 0.05
    pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax)
    pylab.colorbar()
    pylab.show()


def fft_test_code():
    """
     separate test program to check if fft works
    """
    dz_array = compute_dz(49, 15)

    win_L = 67
    win_H = 47
    x,z,t = spectrum_test.get_arrays_XZT(dz_array,win_L,win_H,path2time)
    kx,kz,omega = spectrum_test.estimate_dominant_frequency(dz_array,x,z,t)
    spectrum_test.plot_fft(kx,kz,omega,x,z,t)


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
    parser.add_argument("--skip_frames",type = int, default = 1,help = "number of frames to jump while computing dz")
    parser.add_argument("--skip_row",type = int, default = 1,help = "frame step for rows")
    parser.add_argument("--skip_col",type = int, default = 1,help = "frame step for columns")
    parser.add_argument("--startF",type= int, default = 0, help="The frame from where to start computing dz")
    parser.add_argument("--stopF",type= int, default = 0,help= "The frame where\
            to stop computing dz. Default = number of frames in the videodata")
    parser.add_argument("--diff_frames" , type= int, default=1,\
            help="The time difference (in terms of frame numbers) between the 2 images considered")
    ## add optional arguement to override cache

    args = parser.parse_args()
    compute_dz(args.video_id,args.mintol,args.sigma,args.filter_size,\
            args.skip_frames,args.skip_row,args.skip_col,args.startF,args.stopF,args.diff_frames)

if __name__ == "__main__":
    UI()
