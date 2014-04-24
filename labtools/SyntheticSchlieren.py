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
import gc
import netCDF4
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from numpy import ma
import progressbar
import multiprocessing
import socket
import warnings
import tempfile
import logging
from chunk_shape_3D import chunk_shape_3D

import skimage.morphology, skimage.filter 

logger = logging.getLogger(__name__)

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

def newGetTol(image, mintol=10, mintol2=10):

    nrows, ncols = image.shape
    # work with float point arrays
    image = image * 1.0

     # compute difference between rows
    D = image[:-1,:] - image[1:,:]  # I_+1 - I_0
    E = image[:-2,:] + image[2:,:]  # I_+1 + I_-1

    z = numpy.zeros(ncols)

    # construct array of difference between current and upper
    A = numpy.vstack((D, z))

    # and between current and lower row
    B = numpy.vstack((z, D))

    # and between upper and lower row
    E = numpy.vstack((z, E, z))

    # only estimate if the sequence is monotonic and differences
    # are at least mintol
    C = (abs(A) >= mintol) & (abs(B) >= mintol) & (A*B>0) #&( abs(E/2.0 - A) < mintol2 )

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


def create_nc_file(video_id, 
                   skip_frames, 
                   skip_row, skip_col,
                   mintol,
                   sigma,
                   filter_size,
                   startF,
                   stopF,
                   diff_frames,
                   dz_id=None):
    """ 
    Given a video id create a dz nc file
    return dz_filename,dz_id,dt,dz,dx
    Need to compute dz for the first time.
    Need to create the path for the dz file and create the empty nc file
    """

    logger.debug('creating dz nc file to store the final data in..')

    # Get experiment ID
    db = labdb.LabDB()
    sql = "SELECT expt_id FROM video_experiments WHERE video_id =  %d" % video_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    logger.debug('experiment ID: %d' % expt_id)

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]*1.0
    
    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]*1.0
    
    print "length" , win_l, "\nheight", win_h
    print  "video_id,skip_frames,skip_row,skip_col,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames"
    print  video_id,skip_frames,skip_row,skip_col,expt_id,mintol,sigma,filter_size,startF,stopF,diff_frames

    print "dz_ID ###", dz_id
    if dz_id is None:
        if diff_frames is None:
            diff_frames = "NULL"
        sql = """INSERT INTO dz (video_id,skip_frames,skip_row,skip_col,expt_id,mintol,\
                sigma,filter_size,startF,stopF,diff_frames)\
                VALUES (%d,%d,%d,%d,%d,%d,%f,%d,%d,%d,%s)""" %\
                (video_id,skip_frames,skip_row,skip_col,expt_id,\
                mintol,sigma,filter_size,startF,stopF,diff_frames)
        print sql
        db.execute(sql)
        sql = """SELECT LAST_INSERT_ID()"""
        rows = db.execute(sql)
        dz_id = rows[0][0]
    else:
        # need the sql to include mintol,filtersize,sigma too REMINDER
        sql = """ SELECT video_id,skip_frames,skip_row,skip_col,\
                startF,stopF FROM dz WHERE dz_id = %s""" \
                %dz_id
        previous_row = db.execute_one(sql)
        print previous_row 
        if (previous_row[0] != video_id or previous_row[1] !=skip_frames  or\
                 previous_row[2]!=skip_row or previous_row[3]!=skip_col\
                 or  previous_row[4]!=startF or previous_row[5]!=stopF):
                    print "video id/startF/stopF mismatch!"
                    return None

    dz_path = "/data/dz/%d" % dz_id
    if not os.path.exists(dz_path):
        # Create the directory in which to store the nc file
        os.mkdir(dz_path)

    dz_filename = os.path.join(dz_path, "dz.nc")
    print "dz_ filename: ",dz_filename

    # Declare the nc file for the first time 
    if os.path.exists(dz_filename):
        # ensure you delete the nc file if it exists so as to create new nc file
        os.unlink(dz_filename)

    # find the number for rows and column of the frames and Ntime for dz field
    Nrow = 964/skip_row
    Ncol = 1292/skip_col
    Ntime = numpy.int((stopF-startF)/diff_frames)

    print "no of rows : %d and no of columns : %d and Ntime : %d" %(Nrow,Ncol,Ntime)

    # create the nc file and set the dimensions of z and x axis.
    nc = netCDF4.Dataset(dz_filename, 'w', format = 'NETCDF4')
    row_dim = nc.createDimension('row', Nrow)
    col_dim = nc.createDimension('column', Ncol)
    t_dim = nc.createDimension('time', Ntime)

    # the dimensions are also variables
    ROW = nc.createVariable('row',numpy.float32,('row'))
    COLUMN = nc.createVariable('column',numpy.float32,('column'))
    TIME = nc.createVariable('time',numpy.float32,('time'))

    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    R = numpy.arange(0, win_h, win_h/Nrow, dtype=float)
    C = numpy.arange(0, win_l, win_l/Ncol, dtype=float)

    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    t=numpy.loadtxt(path2time)
    dt = numpy.mean(numpy.diff(t[:,1]))
    
    # compute dt,dz,dx
    dt = dt * skip_frames
    dz = win_h / Nrow
    dx = win_l / Ncol

    ROW[:] = R
    COLUMN[:] = C
    # TODO: Why isn't TIME filled in here?

    # chunk the data intelligently
    valSize = numpy.float32().itemsize
    chunksizes = chunk_shape_3D( ( Ntime, Nrow, Ncol), valSize=valSize )
    logger.debug('chunksizes = {}'.format(chunksizes))

    # declare the 3D data variable 
    DZ = nc.createVariable('dz_array', numpy.float32, 
                 ('time','row','column'),
                 chunksizes = chunksizes)
    
    db.commit()
    nc.close()

    return dz_filename, dz_id, dt, dz, dx, chunksizes

def create_temp_nc_file(dz_filename):
    print "creating the temporary nc file for storing dz array.."
    """
    creating a temporary dz file so as to avoid reading and writing from the same nc file
    while filtering in time. 

    The function returns temp_dz_filename
    """

    # TODO: There is a code duplication between this create_nc function
    # the the original create_nc function

    #open the dz nc file
    dz_nc = netCDF4.Dataset(dz_filename, 'r')

    # Need the dimensions of the 3 axes to create a temporary dz file
    T = dz_nc.variables['time'][:]
    Z = dz_nc.variables['row'][:]
    X = dz_nc.variables['column'][:]
    # find the number for rows and column of the frames and Ntime for temp dz field
    Nrow = Z.size
    Ncol = X.size
    Ntime = T.size

    print "TEMP DZ FILE ### \n no of rows : %d and no of columns : %d and Ntime : %d" %(Nrow,Ncol,Ntime)

    # get a new, unique filename for the temp dz file
    temp_file = tempfile.NamedTemporaryFile(suffix='.nc',
                                            delete=False)
    temp_dz_filename = temp_file.name
    temp_file.close()

    logger.debug('temp_dz_filename: %s' % temp_dz_filename)

    # create the nc file and set the dimensions of z and x axis.
    nc = netCDF4.Dataset(temp_dz_filename,'w',format = 'NETCDF4')
    row_dim = nc.createDimension('row', Nrow)
    col_dim = nc.createDimension('column', Ncol)
    t_dim = nc.createDimension('time', Ntime)

    #the dimensions are also variables
    ROW = nc.createVariable('row', numpy.float32,('row'))
    COLUMN = nc.createVariable('column', numpy.float32,('column'))
    TIME = nc.createVariable('time', numpy.float32,('time'))

    # chunk the data intelligently
    valSize = numpy.float32().itemsize
    chunksizes = chunk_shape_3D( ( Ntime, Nrow, Ncol), valSize=valSize )
    print "chunksizes",chunksizes

    # declare the 3D data variable
    # TODO: why the name change for this variable?
    DZ = nc.createVariable('temp_dz_array', 
            numpy.float32,
            ('time','row','column'),
            chunksizes = chunksizes)

    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    ROW[:] = Z[:]
    COLUMN[:] = X[:]
    # TODO:
    # TIME[:] = ???

    nc.close()

    dz_nc.close()

    return temp_dz_filename

def checkifdzexists(video_id,skip_frames,skip_row,skip_col,mintol,sigma,filter_size,startF,stopF,diff_frames):
    """
    checks if the dz_array is already computed and return dz_id if it does and
    returns none otherwise

    """
    print "checking if dz already exists.."
    db = labdb.LabDB()
    # check if this dz array has already been computed?
    if diff_frames is None:
        diff_frames =  "NULL"
    sql = """SELECT dz_id FROM dz WHERE video_id=%d AND skip_frames=%d \
            AND skip_row = %d AND skip_col = %d AND mintol=%d AND sigma=%.1f \
            AND filter_size=%d AND startF=%d AND stopF=%d AND diff_frames=%s""" %\
            (video_id,skip_frames,skip_row,skip_col,mintol,sigma,filter_size,startF,stopF,diff_frames)

    rows = db.execute(sql)
    if (len(rows) > 0):
        # dz array already computed
        dz_id = rows[0][0]
        print "Loading cached dz %d..." % dz_id
        # load the array from the disk
        dz_path = "/data/dz/%d" % dz_id
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


def calculate(func, (args, index)):
    result = func(*args)
    msg = '%s does %s%s' % (
        multiprocessing.current_process().name,
        func.__name__, args,
        )
    return index, result, msg

def schlieren_lines(p):
    """
    p is a dictionary containing parameters need to 
    compute synthetic schlieren on a pair of images

    returns array
    """
    # Loading the INPUT :: 2 images and converting them into arrays
    IM1 = numpy.array(Image.open(p['filename1']))
    #loading the array according to user specification
    image1 = IM1[::p['skip_row'],::p['skip_col']]    
    IM2 = numpy.array(Image.open(p['filename2']))
    #loading the array according to user specification
    image2 = IM2[::p['skip_row'],::p['skip_col']]

    #step 1: getTol function returns a mask of the pixels of the image that are monotonically increasing
    C = getTol(image1,mintol = p['min_tol'])

    #step 2: call the compute_dz_image function that returns raw delz matrix
    delz = compute_dz_image(image1, image2, p['dz'])

    #step 3: convert the nan's  into 0's and multiply the array with the getTol mask to select
    # the pixels that are relevant
    delz = numpy.nan_to_num(delz) * C

    # step 4: clip the large values
    min_max = 0.03
    clip_min_max = 0.95 * min_max
    delz[delz > clip_min_max] = clip_min_max
    delz[delz < -clip_min_max] = -clip_min_max

    # Step 5 : map the original data from -0.1 to +0.1 to range from 0 to 255
    mapped_delz = numpy.uint8((delz + min_max)/ (2.0 * min_max) * 256)

    # Implementing the skimage.filter.mean so as to apply mean filter on a
    # masked array and then applying a gaussian filter to smooth the image

    # step 6: prepare a mask:: Mask value 1: use the data and 0: ignore the data here within the disk
    mask_delz = numpy.uint8(mapped_delz <>128)
    
    #Step 7 : apply the mean filter to compute values for the masked pixels
    # Apply the filter in Z not in X.
    disk_size = 15
    row_disk = numpy.ones((disk_size,1))
    filt_delz = skimage.filter.rank.mean(mapped_delz,
                #skimage.morphology.disk(disk_size),
                row_disk,
                mask = mask_delz,
                )

    # Step 8: setting the zeros in the filt_delz to 128
    filt_delz[filt_delz ==0] = 128

    # Step 9: mapping back the values from 0 to 255 to its original values of
    # -0.1 to 0.1
    filtered_delz = (filt_delz / 256.0) * (2.0 * min_max) - min_max
    
    # Step 10: Replacing the elements that were already right in the beginning
    filled_delz = (1-mask_delz) * filtered_delz + mask_delz * delz
    
    # Step 11 : applying the Gaussian filter to do a spatial smoothing of the image
    #apply the gaussian smoothing along both X and Z
    smooth_filt_delz = skimage.filter.gaussian_filter(filled_delz, 
            [p['sigma'],1])

    return p['i'], smooth_filt_delz


def compute_dz(video_id, min_tol, sigma, filter_size,skip_frames=1,skip_row=1,skip_col=1,
            startF=0,stopF=0,diff_frames=1,cache=True):
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

    dz_id =checkifdzexists(video_id,skip_frames,skip_row,skip_col,min_tol,sigma,filter_size,startF,stopF,diff_frames)
    if (dz_id is not None) and cache:
        return dz_id
    
    # Create the dz nc file to write data to
    dz_filename,dz_id,dt,dz,dx,chunkshape =create_nc_file(video_id,skip_frames,skip_row,skip_col,min_tol,\
            sigma,filter_size,startF,stopF,diff_frames,dz_id = dz_id)


    # Create a temporary dz file.
    temp_dz_filename = create_temp_nc_file(dz_filename)

    # count: start from the second frame. count is the variable that tracks the
    # current frame
    if diff_frames is None:
        count = startF
    else:
        count=startF+diff_frames

    # Set path to the two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"

    hostname = socket.gethostname()
    cpu_count = multiprocessing.cpu_count()
    if hostname == 'taylor.physics.mun.ca':
        PROCESSES = cpu_count
    else:
        PROCESSES = cpu_count / 2
    PROCESSES = 8

    # Create pool
    pool = multiprocessing.Pool(PROCESSES)

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
    Ls = 13.5 # distance from front of screen to back of tank
    Lb = 0.9 # thickness of barrier
    Ld = 14.9 # width of rear channel
    Lp = 2.4 # thickness of back and front walls
    Lw = 29.5 # width of experimental region of tank
    # index of refraction
    na = 1.00
    np = 1.49
    nb = 1.49
    nw = 1.33
    gamma = 0.0001878 # See eqn 2.8 in Sutherland1999

    # const2 = -1.0/(gamma*((0.5*L_tank*L_tank)+(L_tank*win_l*n_water)))
    # Has been recalculated for our experiment
    dN2dz = -1.0/(Lw*gamma) * 1.0/(0.5*Lw+nw/nb*Lb+Ld+nw/nb * Lp + nw/na *Ls)

    # if diff_frames is given, we are computing dN2dt
    if diff_frames is not None:
        path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
        t = numpy.loadtxt(path2time)
        dt = numpy.mean(numpy.diff(t[:,1]))
        dN2dz = dN2dz / (dt*diff_frames)

    logger.debug('dN2dz = {:f}'.format(dN2dz))

    # progress bar
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets)

    # multiprocessing the task of writing data into nc file
    lock = multiprocessing.Lock()
    def cb(r):
        with lock:
            i, dz = r
            pbar.update(i)

            nc = netCDF4.Dataset(temp_dz_filename, 'a')
            temp_dz = nc.variables['temp_dz_array']
            temp_dz[i, :, :] = dN2dz * dz
            nc.close()

    tasks = []
    #counter for the while loop
    i = 0

    # submit tasks to perform in the while loop and this is in parallel
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

        pool.apply_async(schlieren_lines, (dict(p),), callback=cb)
        i += 1

    if i == 0:
        raise Exception('No pairs of image files found')


    # submit all tasks to worker pool
    pbar.maxval = i

    logger.debug('Schlieren - frame by frame')
    pbar.start()

    # wait for all schlieren task to complete
    pool.close()
    pool.join()

    pbar.finish()

    # open the temporary nc file to put in the time axis
    nc=netCDF4.Dataset(temp_dz_filename,'a')
    temp_dz = nc.variables['temp_dz_array']
    Tm=nc.variables['time']

    # define time axis for the nc time variable
    tl = temp_dz.shape[0]
    t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    Tm[:] = t_array
    nc.close()

    # The last Step of Synthetic Schlieren... Uniform filtering in time
    # open the temp dz nc file
    nc = netCDF4.Dataset(temp_dz_filename, 'r')
    temp_dz_array = nc.variables['temp_dz_array']
    T = nc.variables['time'][:]

    # open the dz nc file for writing in the data
    dz_nc = netCDF4.Dataset(dz_filename, 'a')
    DZarray = dz_nc.variables['dz_array']
    ZZ = dz_nc.variables['row'][:]
    CC = dz_nc.variables['column'][:]
    TT = dz_nc.variables['time']
    #set the time axis for the dz array.
    TT[:] = T[:]

    t_chunk,r_chunk,c_chunk = chunkshape

    print "chunk t,z,x :", t_chunk,r_chunk,c_chunk

    print "CC size, zz size" , CC.size, ZZ.size
    col_count= CC.size-1
    row_count= ZZ.size-1

    # step 12 of Schlieren :: apply uniform filter in the time axis with the filter size of 6 (about
    # 1second). This should smoothen the dz along time.

    logger.debug('Schlieren - chunk by chunk')
    # (i, j) should index the ith and jth chunk
    logger.debug('size = {}'.format(temp_dz_array.shape))
    logger.debug('chunkshape = {}'.format(chunkshape) )
    nt, nz, nx = temp_dz_array.shape
    chunk_nt, chunk_nz, chunk_nx = chunkshape

    chunk_nz = chunk_nz * 2
    #chunk_nx = chunk_nx * 8

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, 
            maxval= (nz // chunk_nz)).start()

    for j in range(nz // chunk_nz):

        pbar.update(j)
        temp = temp_dz_array[:, 
                  j*chunk_nz:(j+1)*chunk_nz,
                  :]

        # TODO: why (6, 1, 1)?
        temp_filt = ndimage.uniform_filter(temp,size = (6,1,1))

        DZarray[:, 
                j*chunk_nz:(j+1)*chunk_nz,
                :] = temp_filt

        #for i in range(nx // chunk_nx):
#
#            pbar.update(j * (nx//chunk_nx) + i)
#
#            temp = temp_dz_array[:, 
#                                 j*chunk_nz:(j+1)*chunk_nz,
#                                 i*chunk_nx:(i+1)*chunk_nx]
#
#            # TODO: why (6, 1, 1)?
#            temp_filt = ndimage.uniform_filter(temp,size = (6,1,1))
#
#            DZarray[:, 
#                    j*chunk_nz:(j+1)*chunk_nz,
#                    i*chunk_nx:(i+1)*chunk_nx] = temp_filt

    pbar.finish()

    dz_nc.close()
    nc.close()

    # remove temp dz file
    os.unlink(temp_dz_filename)

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
    dz_id=compute_dz(args.video_id,args.mintol,args.sigma,args.filter_size,\
            args.skip_frames,args.skip_row,args.skip_col,args.startF,args.stopF,args.diff_frames)

if __name__ == "__main__":
    UI()
