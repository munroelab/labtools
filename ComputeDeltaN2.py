"""
Calculate the deltaN2 field and then displays it in a movie 
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
import netCDF4
from scipy import ndimage

def getTol(image, mintol =10):
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

def append2ncfile(deltaN2_filename,deltaN2_arr):
    """
    Open the nc file
    Append the array to the end of the nc file
    Close the nc file 
    """
    #print"deltaN2 shape", deltaN2_arr.shape
    nc=netCDF4.Dataset(deltaN2_filename,'a')
    deltaN2 = nc.variables['deltaN2_array']
    i= len(deltaN2)
    #print "len(deltaN2): ", i 
    deltaN2[i,:,:]=deltaN2_arr
    nc.close()
    print "appending"

def set_time_axis(deltaN2_filename,dt):
    # open the file in the end and set the time variable start and stop time
    # along with timestep
    nc = netCDF4.Dataset(deltaN2_filename,'a')
    Tm=nc.variables['time']
    print Tm.shape
    tl = len(Tm)
    t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    Tm[:] = t_array
    nc.close()


def compute_dz(video_id,skip_frames=100):
    """
    > Given video_id, calculate the dz array. Output is cached on disk.
    > returns the array dz
    > skip_frames is the number of frames to jump before computing dz
    """
    print "skip_frames is", skip_frames
    db = labdb.LabDB()

    # check if this dz array has already been computed?
    sql = """SELECT id FROM deltaN2 WHERE video_id = %d AND skip_frames = %d""" % (video_id,skip_frames)
    rows = db.execute(sql)
    if len(rows) > 0:
        # deltaN2 array already computed
        id = rows[0][0]
        print "Loading cached deltaN2 %d..." % id
        
        # load the array from the disk
        deltaN2_path = "/Volumes/HD3/deltaN2/%d/" % id
        deltaN2_filename = deltaN2_path+'deltaN2.nc'
        
        #loading the nc file
        print "file loading --> " ,deltaN2_filename
        nc=netCDF4.Dataset(deltaN2_filename,'a')
        
        # some information about the nc file
        print "dimensions of nc file -> ", nc.dimensions.keys()
        print "variables of nc file -> ", nc.variables.keys()
        deltaN2_arr=nc.variables['deltaN2_array']
        print "shape of nc file -> ", deltaN2_arr.shape
        nc.close()
        
        return deltaN2_filename

    
    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]
    
    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]
    
    win_l=win_l*1.0
    win_h=win_h*1.0
    print "lenght" , win_l, "\nheight", win_h

    sql = """SELECT path FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    print "path", rows[0][0]
    
    # COMPUTE DELTAN2 FOR THE FIRST TIME
    
    # Create the directory in which to store the nc file
    sql = """INSERT INTO deltaN2 (video_id, skip_frames) VALUES (%d, %d)""" % (video_id, skip_frames)
    print sql
    db.execute(sql)
    sql = """SELECT LAST_INSERT_ID()"""
    rows = db.execute(sql)
    id = rows[0][0]

    deltaN2_path = "/Volumes/HD3/deltaN2/%d" % id
    os.mkdir(deltaN2_path)

    deltaN2_filename = os.path.join(deltaN2_path, "deltaN2.nc")
    print "filename: " ,deltaN2_filename

    # Declare the nc file for the first time 
    nc = netCDF4.Dataset(deltaN2_filename,'w',format = 'NETCDF4')
    row_dim = nc.createDimension('row',964)
    col_dim = nc.createDimension('column',1292)
    t_dim = nc.createDimension('time',None)

    #the dimensions are also variables
    ROW = nc.createVariable('row',numpy.float32,('row'))
    print  nc.dimensions.keys(), ROW.shape,ROW.dtype
    COLUMN = nc.createVariable('column',numpy.float32,('column'))
    print COLUMN.shape, COLUMN.dtype
    TIME = nc.createVariable('time',numpy.float32,('time'))
    print TIME.shape, TIME.dtype
    # declare the 3D data variable 
    deltaN2 = nc.createVariable('deltaN2_array',numpy.float32,('time','row','column'))
    print nc.dimensions.keys() , deltaN2.shape, deltaN2.dtype
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
    #get the number of frames
    sql = """SELECT num_frames FROM video
             WHERE video_id = %d""" % video_id
    rows = db.execute(sql)
    num_frames = rows[0][0]
    n = 3
    count = n + 1
    print "R",ROW.shape
    print "C",COLUMN.shape

    db.commit()
    nc.close()

    # Set path to the two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    filename1 = path % (video_id, count - n)
    filename2 = path % (video_id, count)

    image1 = numpy.array(Image.open(filename1))
    image2 = numpy.array(Image.open(filename2))

    H = 52.0
    dz = H / image1.shape[0]
    
    #set mintol to some arbitrary value
    mintol = 1
    
    C = getTol(image1, mintol = mintol)
    print "C :",C
    delz = compute_dz_image(image1, image2, dz) 
    delz = numpy.nan_to_num(delz) * C
    
    # calculate deltaN2 from delz
    n_water = 1.3330
    n_air = 1.0
    L_tank = 453.0
    gamma = 0.0001878
    #deltaN2 
    a = 1.0/(gamma * ((0.5*L_tank*L_tank)+(L_tank*win_l*n_water)))
    print "a:",a
    deltaN2_arr = -1.0*delz * a
    #print deltaN2_arr[400:420,800:830]    
    append2ncfile(deltaN2_filename,deltaN2_arr)
    vmax = 0.01
    print "max: ",numpy.max(deltaN2_arr)
    print "min: ",numpy.min(deltaN2_arr)

    """ old ploting code
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    img = pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax,
                    animated=False, label='delz', aspect='auto')
    pylab.colorbar()
    pylab.show(block=False) """

    from scipy.ndimage import gaussian_filter
    from numpy import ma
    index = 0
    while True:
        print "render frame %d of %d" % (count, num_frames)
        index += 1

        filename1 = path % (video_id, count - n)
        filename2 = path % (video_id, count)

        if not os.path.exists(filename2):
            break

        image1 = numpy.array(Image.open(filename1))
        image2 = numpy.array(Image.open(filename2))

        C = getTol(image1, mintol = mintol)
        # compute dz
        delz = compute_dz_image(image1, image2, dz) 
        delz = numpy.nan_to_num(delz) * C
        # compute deltaN2
        deltaN2_arr = -1.0* delz * a
        #debug messages
        print "max: ",numpy.max(deltaN2_arr)
        print "min: ",numpy.min(deltaN2_arr)
        # append to nc file
        #print deltaN2_arr[400:420,800:830]
        append2ncfile(deltaN2_filename,deltaN2_arr)
        # clip large values
        bound = 1.0
        delz[delz > bound] = bound
        delz[delz < -bound] = -bound

        # fill in missing values
        filt_delz = ndimage.gaussian_filter(delz, (21,21))
        i = abs(delz) > 1e-8
        filt_delz[i] = delz[i]

        # smooth
        filt_delz = ndimage.gaussian_filter(filt_delz,7)

        # Old ploting
        """ img.set_data(filt_delz)
        fig.canvas.draw()
        ax.set_title('n = %d' % count)"""

        count += skip_frames

    
    # define time axis for the nc time variable
    nc = netCDF4.Dataset(deltaN2_filename,'a')
    Tm=nc.variables['time']
    print Tm.shape
    tl = len(Tm)
    t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    print "length of T axis: " ,tl, "\n array length: " ,t_array.shape
    Tm[:] = t_array
    nc.close()
    return deltaN2_filename
    #cache deltaN2_array to disk**********
    #sql = """INSERT INTO dz (video_id, skip_frames)
    #         VALUES (%d, %d)""" % (video_id, skip_frames)
    #print sql
    #db.execute(sql)
    #sql = """SELECT LAST_INSERT_ID()"""
    #rows = db.execute(sql)
    #deltaN2_id = rows[0][0]
    #deltaN2_path = "/Volumes/HD3/dz/%d" % deltaN2_id
    #os.mkdir(deltaN2_path)
    #deltaN2_filename = os.path.join(dz_path, "dz.npy")
    #numpy.save(deltaN2_filename, dz_array)
    #db.commit()
    #**********************
    #return deltaN2_array


def ncfile_movie(deltaN2_filename):
    #loading the nc file
    print "file loading --> " ,deltaN2_filename
    nc=netCDF4.Dataset(deltaN2_filename,'a')
    # some information about the nc file
    print "dimensions of nc file -> ", nc.dimensions.keys()
    print "variables of nc file -> ", nc.variables.keys()
    # loading the data into arrays
    deltaN2 = nc.variables['deltaN2_array']
    T = nc.variables['time']
    X = nc.variables['column']
    Z = nc.variables['row']
    print "shape of deltaN2 -> ", deltaN2.shape
    print "shape of T -> ", T.shape
    print "shape of X -> ", X.shape
    print "shape of Z -> ", Z.shape
    
    """fig = pylab.figure()
    ax = fig.add_subplot(111)
    img= pylab.imshow(deltaN2[0],extent=[X[0],X[-1],Z[0],Z[-1]],vmin=-0.08,vmax=0.0,interpolation='nearest',animated=False,label='deltaN2', aspect='auto')
    pylab.colorbar()
    pylab.show(block=False)
    print "length", len(T)
    for i in range()):
        img.set_data(deltaN2[i])
        ax.set_title('frame %d' % i)
        fig.canvas.draw()
        print "frame:",i
    """
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

def UI(): 
    
    """
    take arguments from the user :video id and skip frame number and call
    compute deltaN2 function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id",type = int, 
                        help = "Enter the video id of the frames on which to do Synthetic Schlieren")
    parser.add_argument("--skip_frames",type = int, default = 10,help = "number of frames to jump while computing deltaN2")
    ## add optional arguement to override cache

    args = parser.parse_args()
    deltaN2_filename = compute_dz(args.video_id,args.skip_frames)
    ncfile_movie(deltaN2_filename)

def fft_test_code():
    """
     separate test program to check if fft works
    """
    deltaN2_array = compute_dz(49, 15)

    win_L = 67
    win_H = 47
    x,z,t = spectrum_test.get_arrays_XZT(deltaN2_array,win_L,win_H,path2time)
    kx,kz,omega = spectrum_test.estimate_dominant_frequency(deltaN2_array,x,z,t)
    spectrum_test.plot_fft(kx,kz,omega,x,z,t)

if __name__ == "__main__":
    UI()
