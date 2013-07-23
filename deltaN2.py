"""
Calculate the deltaN2 field from the dz field  
"""


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
import SyntheticSchlieren

global db
db = labdb.LabDB()


def append2ncfile(deltaN2_filename,deltaN2_arr):
    """
    Open the nc file
    Append the array to the end of the nc file
    Close the nc file 
    """
    nc=netCDF4.Dataset(deltaN2_filename,'a')
    deltaN2 = nc.variables['deltaN2_array']
    i= len(deltaN2)
    deltaN2[i,:,:]=deltaN2_arr
    print "len(deltaN2): ", i 
    print "appending"
    nc.close()


def create_nc_file_test(video_id,min_tol,sigma,filter_size,skip_frames):
    # get EXPT ID
    sql = """ SELECT expt_id FROM video_experiments WHERE video_id =  %d """ % video_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print " experiment ID : ", expt_id

    # Create the directory in which to store the nc file
    sql = """INSERT INTO deltaN2 (video_id,skip_frames,expt_id,mintol,sigma,filter_size) VALUES \
            (%d,%d,%d,%d,%f,%d)""" % (video_id,skip_frames,expt_id,min_tol,sigma,filter_size)
    print sql
    db.execute(sql)
    sql = """SELECT LAST_INSERT_ID()"""
    rows = db.execute(sql)
    id = rows[0][0]
    deltaN2_path = "/Volumes/HD3/deltaN2/%d" % id
    os.mkdir(deltaN2_path)
    deltaN2_filename = os.path.join(deltaN2_path, "deltaN2.nc")
    print "filename: " ,deltaN2_filename

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]*1.0
    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]*1.0
    print "length" , win_l, "\nheight", win_h

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
    print "R",ROW.shape
    print "C",COLUMN.shape

    db.commit()
    nc.close()
    return deltaN2_filename,dt,num_frames,win_l,win_h,id


def create_nc_file(video_id,skip_frames):
    # get EXPT ID
    sql = """ SELECT expt_id FROM video_experiments WHERE video_id =  %d """ % video_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print " experiment ID : ", expt_id

    # Create the directory in which to store the nc file
    sql = """INSERT INTO deltaN2 (video_id, skip_frames,expt_id) VALUES \
            (%d,%d,%d)""" % (video_id, skip_frames,expt_id)
    print sql
    db.execute(sql)
    sql = """SELECT LAST_INSERT_ID()"""
    rows = db.execute(sql)
    id = rows[0][0]
    deltaN2_path = "/Volumes/HD3/deltaN2/%d" % id
    os.mkdir(deltaN2_path)
    deltaN2_filename = os.path.join(deltaN2_path, "deltaN2.nc")
    print "filename: " ,deltaN2_filename

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]*1.0
    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]*1.0
    print "length" , win_l, "\nheight", win_h

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
    print "R",ROW.shape
    print "C",COLUMN.shape

    db.commit()
    nc.close()
    return deltaN2_filename,dt,num_frames,win_l,win_h,id


def checkifdeltaN2exists_test(video_id,min_tol,sigma,filter_size,skip_frames):
    # check if deltaN2 is already computed
    sql = """SELECT id FROM deltaN2 WHERE video_id = %d AND skip_frames = %d AND
    mintol = %d AND sigma = %f AND filter_size=%d"""  %(video_id,skip_frames,min_tol,sigma,filter_size)
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
        return id
    else:
        # deltaN2 is not yet computed
        return 0

def checkifdeltaN2exists(video_id,skip_frames):
    # check if deltaN2 is already computed
    sql = """SELECT id FROM deltaN2 WHERE video_id = %d AND skip_frames = %d """ % (video_id,skip_frames)
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
        return id
    else:
        # deltaN2 is not yet computed
        return 0

def compute_deltaN2(video_id,min_tol,sigma,filter_size,skip_frames=1,startF=0,stopF=0):
    """ Computes deltaN2 
    """
    db = labdb.LabDB()

    # check if DELTAN2 ARRAY has already been computed?
    # if deltaN2_flag =0 means deltaN2 is not computed else it holds the value
    # of the deltaN2 id.
    
    
    deltaN2_flag = checkifdeltaN2exists_test(video_id,min_tol,sigma,filter_size,skip_frames)     

    if (deltaN2_flag != 0) :
        return deltaN2_flag # deltaN2_flag is returning the deltaN2 id of the video_id
    

    # COMPUTE DELTAN2 FOR THE FIRST TIME
    #deltaN2_filename,dt,num_frames,win_l,win_h,deltaN2_id = create_nc_file(video_id,skip_frames)
    deltaN2_filename,dt,num_frames,win_l,win_h,deltaN2_id =\
            create_nc_file_test(video_id,min_tol,sigma,filter_size,skip_frames)
    
    # check if the dz file already exists 
    dz_id = SyntheticSchlieren.compute_dz(video_id,min_tol,sigma,filter_size,skip_frames)
    print "dz_id::" ,dz_id

    # Define the constants used in the computation of deltaN2
    n_water = 1.3330
    n_air = 1.0
    L_tank = 453.0
    gamma = 0.0001878
    const = 1.0/(gamma * ((0.5*L_tank*L_tank)+(L_tank*win_l*n_water)))
    print "constant:",const

    # load the nc file of dz 
    data = netCDF4.Dataset("/Volumes/HD3/dz/%d/dz.nc" % dz_id, 'r')
    dz_array = data.variables['dz_array']
    dz_limit = dz_array.shape[0]

    from scipy.ndimage import gaussian_filter
    from numpy import ma
    index = 0
    while index < (dz_limit-1):
        print "computing deltaN2 of dz_frame %d of %d" % (index,(dz_limit-1))

        # compute deltaN2
        deltaN2_arr = -1.0* dz_array[index] * const
        #debug messages
        print "max: ",numpy.max(deltaN2_arr)
        print "min: ",numpy.min(deltaN2_arr)
        # append to nc file
        append2ncfile(deltaN2_filename,deltaN2_arr)

        index += 1

    data.close()

    # define time axis for the nc time variable
    nc = netCDF4.Dataset(deltaN2_filename,'a')
    deltaN2 = nc.variables['deltaN2_array']
    tl = deltaN2.shape[0]
    Tm=nc.variables['time']
    t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    print "length of T axis: " ,tl, "\n array length: " ,t_array.shape
    Tm[:] = t_array
    nc.close()
    return deltaN2_id
    
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
    img=pylab.imshow(deltaN2[0],extent=[X[0],X[-1],Z[0],Z[-1]],vmin=-0.08,vmax=0.0,\
            interpolation='nearest',animated=False,label='deltaN2', aspect='auto')
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
    parser.add_argument("video_id",type = int, \
            help = "Enter the video id of the frames on which to do Synthetic Schlieren")
    parser.add_argument("mintol",type = int, help="Helps estimate the monoticity of the data")
    parser.add_argument("sigma", type = float, help= "standard deviation for the Gaussian kernal")
    parser.add_argument("filter_size", type = int, help = "filter size")
    
    parser.add_argument("--skip_frames",type = int, default = 10,help = "number of frames to jump while computing deltaN2")
    parser.add_argument("--startF",type = int, default = 0,help = "Frame number to start with while computing deltaN2")
    parser.add_argument("--stopF",type = int, default = 0,help = "Frame number where to stop while computing deltaN2")
    
    ## add optional arguement to override cache

    args = parser.parse_args()
    deltaN2_id =
    compute_deltaN2(args.video_id,args.mintol,args.sigma,args.filter_size,args.skip_frames,args.startF,args.stopF)
#    ncfile_movie(deltaN2_filename)

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
