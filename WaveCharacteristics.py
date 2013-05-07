import matplotlib
import argparse 
import numpy
import labdb
import os
import netCDF4
import pylab


def open_data(dz_id,L_screen):
    # open the dataset
    dz_filepath = "/Volumes/HD3/dz/%d/dz.nc" % dz_id
    nc = netCDF4.Dataset(dz_filepath,'a')
    # loading the data
    dz = nc.variables['dz_array']
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']
    # print some useful information about the dataset
    print "dimensions of the nc file :" nc.dimensions.keys()
    print "variables  of the nc file :" nc.variables.keys()
    print "dz shape : " , dz.shape
    print "t  shape : " , t.shape
    print "z shape : " , z.shape
    print "x shape : " , x.shape
    #some constants used 
    n_water = 1.3330
    n_air = 1.0
    L_tank = 453
    gamma = 0.0001878
    #calculate the deltaN2 field
    a = 1/(gamma * (0.5*L_tank**2+L_tank*L_screen*n_water))
    deltaN2 = -dz * a
    win_L = x[-1]
    win_H = z[-1]
    compute_dn2t(deltaN2,win_L,win_H)


def compute_dn2t(deltaN2,win_L,win_H):
    """calculate the derivative of deltaN2 and write it to a separate nc file
    """
    #creating an nc file
    ROW = nc.createVariable('row',numpy.float32,('row'))
    print  nc.dimensions.keys(), ROW.shape,ROW.dtype
    COLUMN = nc.createVariable('column',numpy.float32,('column'))
    print nc.dimensions.keys() , COLUMN.shape, COLUMN.dtype
    TIME = nc.createVariable('time',numpy.float32,('time'))
    print nc.dimensions.keys() ,TIME.shape, TIME.dtype
    # declare the 3D data variable 
    d2nt = nc.createVariable('d2nt',numpy.float32,('time','row','column'))
    print nc.dimensions.keys() ,d2nt.shape,d2nt.dtype
    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    R =numpy.arange(0,win_h,win_h/964,dtype=float)
    C =numpy.arange(0,win_l,win_l/1292,dtype=float)

    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    t=numpy.loadtxt(path2time)
    dt = numpy.mean(numpy.diff(t[:,1]))
    print "dt = " ,dt
    






def UI():
    """
    take the dz_id from the user and calculate the change in the squared
    buoyancy frequency (deltaN2), the time derivative of the deltaN2, U, W, and
    the energy flux
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dz_id", type=int, help="enter the dz_id")
    args = parser.parse_args()
    open_data(args.dz_id)


