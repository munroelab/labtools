"""
test program for implementing line based synthetic schlieren 
using matrix based operations
"""
import spectrum_test
import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import argparse
import Image
import pylab
import numpy
import time
import os
import labdb
import netCDF4
from scipy import ndimage
import progressbar

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

def append2ncfile(dz_filename,dz_arr):
    """
    Open the nc file
    Append the array to the end of the nc file
    Close the nc file 
    """
    nc=netCDF4.Dataset(dz_filename,'a')
    DZ = nc.variables['dz_array']
    i= len(DZ)
    DZ[i,:,:]=dz_arr
    print "dz shape: ", DZ.shape,"appending"
    print "len(dz): ", i 
    nc.close()


def create_nc_file(video_id,skip_frames,mintol,sigma,filter_size,startF,stopF,diff_frames,
        dz_id=None ):

    """ 
        Given a video id it creates a dz nc file
        return dz_filename,dz_id,dt,win_h
        Need to compute dz for the first time.
        Need to create the path for the dz file and create the empty nc file
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
    print  "video_id,skip_frames,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames"
    print  video_id,skip_frames,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames
    print "dz_ID ###", dz_id
    if dz_id is None:
        sql = """INSERT INTO dz (video_id,skip_frames,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames)\
                VALUES (%d,%d,%d,%d,%f,%d,%d,%d,%d)""" %\
                (video_id,skip_frames,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames)
        print sql
        db.execute(sql)
        sql = """SELECT LAST_INSERT_ID()"""
        rows = db.execute(sql)
        dz_id = rows[0][0]
    else:
        sql = """ SELECT video_id,startF,stopF FROM dz WHERE dz_id = %s""" \
                %dz_id
        previous_row = db.execute_one(sql)
        print previous_row 
        if (previous_row[0] != video_id or previous_row[1] !=startF or\
                 previous_row[2]!=stopF):
                    print "video id/startF/stopF mismatch!"
                    return None

    dz_path = "/Volumes/HD4/dz/%d" % dz_id
    if not os.path.exists(dz_path):
        # Create the directory in which to store the nc file
        os.mkdir(dz_path)
    dz_filename = os.path.join(dz_path, "dz.nc")
    print "dz_ filename: ",dz_filename
    # Declare the nc file for the first time 
    if os.path.exists(dz_filename):
        # ensure you delete the nc file if it exists so as to create new nc file
        os.unlink(dz_filename)
    nc = netCDF4.Dataset(dz_filename,'w',format = 'NETCDF4')
    row_dim = nc.createDimension('row',964)
    col_dim = nc.createDimension('column',1292)
    t_dim = nc.createDimension('time',None)
    
    #the dimensions are also variables
    ROW = nc.createVariable('row',numpy.float32,('row'))
    print  nc.dimensions.keys(), ROW.shape,ROW.dtype
    COLUMN = nc.createVariable('column',numpy.float32,('column'))
    print nc.dimensions.keys() , COLUMN.shape, COLUMN.dtype
    TIME = nc.createVariable('time',numpy.float32,('time'))
    print nc.dimensions.keys() ,TIME.shape, TIME.dtype
    
    # declare the 3D data variable 
    DZ = nc.createVariable('dz_array',numpy.float32,('time','row','column'))
    print nc.dimensions.keys() , DZ.shape,DZ.dtype
    
    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    R =numpy.arange(0,win_h,win_h/964,dtype=float)
    C =numpy.arange(0,win_l,win_l/1292,dtype=float)
    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    t=numpy.loadtxt(path2time)
    dt = numpy.mean(numpy.diff(t[:,1]))
    print "dt = " ,dt
    dt = dt*skip_frames
    print "timestep: " ,dt

    ROW[:] = R
    COLUMN[:] = C

    db.commit()
    nc.close()
    return dz_filename,dz_id,dt,win_h



def checkifdzexists(video_id,skip_frames,mintol,sigma,filter_size,startF,stopF,diff_frames):
    """
    checks if the dz_array is already computed and return dz_id if it does and
    returns none otherwise

    """
    db = labdb.LabDB()
    # check if this dz array has already been computed?
    sql = """SELECT dz_id FROM dz WHERE video_id=%d AND skip_frames=%d AND mintol=%d AND sigma=%.1f AND filter_size=%d AND startF=%d AND\
            stopF=%d AND diff_frames=%d""" %\
            (video_id,skip_frames,mintol,sigma,filter_size,startF,stopF,diff_frames)

    rows = db.execute(sql)
    if len(rows) > 0:
        # dz array already computed
        dz_id = rows[0][0]
        print "Loading cached dz %d..." % dz_id
        # load the array from the disk
        dz_path = "/Volumes/HD4/dz/%d" % dz_id
        dz_filename = dz_path+'/'+'dz.nc'
        
        #dz_array = numpy.load(dz_filename)
        #loading the nc file
        print dz_filename
        nc=netCDF4.Dataset(dz_filename,'a')
        # some information about the nc file
        print "dimensions of nc file -> ", nc.dimensions.keys()
        print "variables of nc file -> ", nc.variables.keys()
        # loading the data 
        dz_arr = nc.variables['dz_array']
        print "shape of nc file -> ", dz_arr.shape
        return dz_id 
    else:
        return None


def compute_dz(video_id,min_tol,sigma,filter_size,skip_frames=1,startF=0,stopF=0,diff_frames=1,
            cache=True):
    """
    > Given video_id, calculate the dz array. Output is cached on disk.
    > returns the array dz
    > skip_frames is the number of frames to jump before computing dz

    Returns dz_id
    """
    db = labdb.LabDB()
    
    #check if the dz file already exists. function checkifdzexists() returns
    #dz_id if the dz file exists else returns 0
    #dz_flag = checkifdzexists(video_id,skip_frames)

    # get the number of frames if stopF is unspecified
    if (stopF == 0):
        sql = """ SELECT num_frames FROM video WHERE video_id = %d""" % video_id
        rows = db.execute(sql)
        stopF = rows[0][0]
        print "stop_frames = ", stopF 
    num_frames=stopF-startF
    print "num_frames:" ,num_frames

    dz_id =checkifdzexists(video_id,skip_frames,min_tol,sigma,filter_size,startF,stopF,diff_frames)
    if (dz_id is not None) and cache:
        return dz_id
    # Call the function that will create the nc file to append data to
    dz_filename,dz_id,dt,win_h=create_nc_file(video_id,skip_frames,min_tol,sigma,filter_size,startF,\
            stopF,diff_frames,dz_id = dz_id)

    
    # n: number of frames between 2 images we subtract 
    n=diff_frames
    # count: start from the second frame. count is the variable that tracks the
    # current frame
    count=startF+n

    # Set path to the two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
    # path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    
    dz = win_h / 964
    
    #mintol =10
   
    """ C = getTol(image1, mintol = mintol)
    delz = compute_dz_image(image1, image2, dz) 
    delz = numpy.nan_to_num(delz) * C
    #from scipy.ndimage import gaussian_filter, correlate
    #delz_filtered = gaussian_filter(delz,[10,10])
    #delz = C* delz + (1-C) * delz_filtered
    #dz_array = generate_dz_array(delz,dz_array)
    append2ncfile(dz_filename,delz)
    vmax = 0.01
    
    old ploting code
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    img = pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax,
                    animated=False, label='delz', aspect='auto')
    pylab.colorbar()
    pylab.show(block=False) """

    from scipy.ndimage import gaussian_filter
    from numpy import ma
    
    #check if path exists
    filename2 = path % (video_id, count)

    """
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=nx).start()
    for n in range(nx):
        pbar.update(n)
    pbar.finish()
    """

    # while True
    while os.path.exists(filename2) & (count <=stopF):
        
        print "render frame %d of %d" % (count, num_frames)

        filename1 = path % (video_id, count - n)
        filename2 = path % (video_id, count)

        if not os.path.exists(filename2):
            break

        image1 = numpy.array(Image.open(filename1))
        image2 = numpy.array(Image.open(filename2))

        C = getTol(image1, mintol = min_tol)
        delz = compute_dz_image(image1, image2, dz) 
        delz = numpy.nan_to_num(delz) * C
        #delz_m = ma.masked_where(C==False, delz)
        #dz_array = generate_dz_array(delz,dz_array)
        # clip large values
        bound = 1.0
        delz[delz > bound] = bound
        delz[delz < -bound] = -bound

        # fill in missing values
        filt_delz = ndimage.gaussian_filter(delz, (sigma,sigma))
        #i = abs(delz) > 1e-8
        filt_delz = C*delz+ (1-C)*filt_delz

        # smooth
        #filt_delz = ndimage.gaussian_filter(filt_delz, 7)
        
        # spatial smoothing along x and z axis
        smooth_filt_delz =ndimage.uniform_filter(filt_delz,size=(filter_size,filter_size))

        # Old ploting
        """ img.set_data(filt_delz)
        fig.canvas.draw()
        ax.set_title('n = %d' % count)"""
        print "max: ",numpy.max(smooth_filt_delz)
        print "min: ",numpy.min(smooth_filt_delz)
        append2ncfile(dz_filename,smooth_filt_delz)
        count += skip_frames
        filename2 = path % (video_id, count)


    
    # define time axis for the nc time variable
    nc = netCDF4.Dataset(dz_filename,'a')
    DZarray = nc.variables['dz_array']
    
    tl = DZarray.shape[0]
    print "no of timesteps:", tl
    Tm=nc.variables['time']
    ZZ = nc.variables['row'][:]
    CC = nc.variables['column'][:]
    count=0
    # TRIAL :: apply uniform filter in the time axis with the filter size of 6 (about
    # 1second). This should smoothen the dz along time.
    col_count=0
    start=0
    stop=20
    print "row shape: ",ZZ.shape
    for row_count in range(stop):
        print "looping %d out of %d" %(row_count,stop)
        temp1 = nc.variables['dz_array'][:,start:(start+47),:]
        temp1 = ndimage.uniform_filter(temp1,size = (6,1,1))
        print " temp1.shape:",temp1.shape
        DZarray[:,row_count,:]=temp1
        start = start+47
    
    print "DZarray::" ,DZarray.shape
    print "time.shape(before):" ,Tm.shape
    t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    print "time:",t_array
    Tm[:] = t_array
    

    
    nc.close()

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
    parser.add_argument("--startF",type= int, default = 0, help="The frame from where to start computing dz")
    parser.add_argument("--stopF",type= int, default = 0,help= "The frame where\
            to stop computing dz. Default = number of frames in the videodata")
    parser.add_argument("--diff_frames" , type= int, default=1,\
            help="The time difference (in terms of frame numbers) between the 2 images considered")
    ## add optional arguement to override cache

    args = parser.parse_args()
    dz_id=compute_dz(args.video_id,args.mintol,args.sigma,args.filter_size,\
            args.skip_frames,args.startF,args.stopF,args.diff_frames)

if __name__ == "__main__":
    UI()
