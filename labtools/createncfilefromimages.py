

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

def append2ncfile(video_filename,img_arr):
    """
    Open the nc file
    Append the array to the end of the nc file
    Close the nc file 
    """
    nc=netCDF4.Dataset(video_filename,'a')
    VIDEO = nc.variables['img_array']
    i = len(VIDEO)
    VIDEO[i,:,:]=img_arr
    print "video shape: ", VIDEO.shape,"appending"
    print "len(video): ",len(VIDEO)
    nc.close()


def create_nc_file(video_id):
    """ Need to compute dz for the first time.
        Need to create the path for the video file and create the empty nc file
    """
    
    db = labdb.LabDB()
    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]*1.0
    
    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]*1.0
    
    print "lenght" , win_l, "\nheight", win_h

    # Create the directory in which to store the nc file
    video_path = "/Volumes/HD4/videoncfiles/%d" % video_id
    os.mkdir(video_path)
    video_filename = os.path.join(video_path, "video.nc")
    
    # Declare the nc file for the first time 
    nc = netCDF4.Dataset(video_filename,'w',format = 'NETCDF4')
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
    VIDEO = nc.createVariable('img_array',numpy.float32,('time','row','column'))
    print "VIDEO array:",nc.dimensions.keys() , VIDEO.shape,VIDEO.dtype
    
    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    R =numpy.arange(0,win_h,win_h/964,dtype=float)
    C =numpy.arange(0,win_l,win_l/1292,dtype=float)
    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    t=numpy.loadtxt(path2time)
    dt = numpy.mean(numpy.diff(t[:,1]))
    print "dt = " ,dt
    ROW[:] = R
    COLUMN[:] = C

    #get the number of frames
    sql = """SELECT num_frames FROM video WHERE video_id = %d""" % video_id
    rows = db.execute(sql)
    num_frames = rows[0][0]
    
    print "R",ROW.shape
    print "C",COLUMN.shape

    db.commit()
    nc.close()
    return video_filename,dt,num_frames

def compute_videoncfile(video_id):

    db = labdb.LabDB()
    video_filename,dt,num_frames = create_nc_file(video_id)
    # current frame
    count=0

    # Set path to the images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
   
    filename1 = path % (video_id, count)
    
    # while True
    while os.path.exists(filename1) & (count <=num_frames):
        image1 = numpy.array(Image.open(filename1))*1.0
        print "render frame %d of %d" % (count, num_frames)
        append2ncfile(video_filename,image1)
        count +=1
        filename1 = path % (video_id, count)
        print "file:", filename1

    # define time axis for the nc time variable
    nc = netCDF4.Dataset(video_filename,'a')
    array = nc.variables['img_array']
    tl = array.shape[0]
    print "no of timesteps:", tl
    Tm=nc.variables['time']
    print "time.shape(before):" ,Tm.shape
    t_array = numpy.mgrid[0:tl*dt:tl*1.0j]
    print "t array:",t_array.shape
    Tm[:] = t_array
    print "time:",Tm.shape

    nc.close()
    return 




def UI(): 
    
    """
    take arguments from the user :video id and combine the images into 1 nc file 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id",type = int, 
                        help = "Enter the video id of the frames on which to do Synthetic Schlieren")
    args = parser.parse_args()
    compute_videoncfile(args.video_id)


if __name__ == "__main__":
    UI()
