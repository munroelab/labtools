# This program calculates the dn2t of the deltaN2.nc file given and then
# if not already done, computes the vertical displacement amplitude and creates an entry for it in
# the database "/Volumes/HD3/vertical_displacement_amplitude/%d/a_xi.nc 
# The deltaN2 database and the vertical_displacement_amplitude database are
# related by the deltaN2_id fields.
#


import playncfile
import matplotlib
import argparse 
import numpy
import labdb
import os
import netCDF4
import pylab
import matplotlib.pyplot as plt
import Energy_flux


def createncfile(dz_id,diff_frames,t,x,z):
    """ 
    create the nc file in which we will store a_xi array.
    create a row in the database for the nc file stored
    update the database
    """
    db = labdb.LabDB()
    #create the directory in which to store the nc file
    sql = """INSERT into dn2t (dz_id,diff_frames) VALUES (%d,%d)""" % (dz_id,diff_frames)  
    db.execute(sql)
    sql = """SELECT LAST_INSERT_ID()""" 
    rows = db.execute(sql)
    dn2t_id = rows[0][0]
    dn2t_path = "/Volumes/HD3/dn2t/%d" % dn2t_id 
    os.mkdir(dn2t_path)

    dn2t_filename = os.path.join(dn2t_path,"dn2t.nc")
    print "d(N2)/dt filename : ",dn2t_filename


    # Declare the nc file for the first time
    nc = netCDF4.Dataset(dn2t_filename,'w',format = 'NETCDF4')
    row_dim = nc.createDimension('row',964)
    col_dim = nc.createDimension('column',1292)
    lenT=t.shape[0]  #lenT is the length of the dn2t file.Its 1 element shorter in time axis than deltaN2
    print "time axis  length",lenT     # debug info
    t_dim = nc.createDimension('time',lenT)

    # Dimensions are also variable
    ROW = nc.createVariable('row',numpy.float32,('row'))
    print  nc.dimensions.keys(), ROW.shape,ROW.dtype
    COLUMN = nc.createVariable('column',numpy.float32,('column'))
    print nc.dimensions.keys() , COLUMN.shape, COLUMN.dtype
    TIME = nc.createVariable('time',numpy.float32,('time'))
    print nc.dimensions.keys() ,TIME.shape, TIME.dtype

    # declare the 3D data variable 
    dn2t = nc.createVariable('dn2t_array',numpy.float32,('time','row','column'))
    print nc.dimensions.keys() ,dn2t.shape,dn2t.dtype

    # assign the values
    TIME[:] = t
    ROW[:] = z
    COLUMN[:] = x

    nc.close()
    db.commit()
    return dn2t_filename


def append2ncfile(dn2t,var,num):
    """
    Append the array to the end of the nc file
    """
    print "appending.."
    dn2t[num] = var

    
def compute_a_xi(dz_ID):
    # access database
    db = labdb.LabDB()

    #check if the dataset already exists
    sql = """SELECT id FROM dn2t WHERE dz_id = %d """ % (dz_ID)
    rows = db.execute(sql)

    if len(rows) > 0:
    
        # A xi array already computed
        id = rows[0][0]
        print "loading dn2t %d dataset  .." % id

        #load the array from disk
        dn2t_path = "/Volumes/HD3/dn2t/%d/" % id
        dn2t_filename = dn2t_path + 'dn2t.nc'

    else:

        #  open the dataset dz.nc for calculating the time derivative for
        #  the first time
        sql = """SELECT diff_frames FROM dz WHERE dz_id = %d  """ % dz_ID 
        rows = db.execute(sql)
        diff_frames = rows[0][0]
        filepath = "/Volumes/HD3/dz/%d/dz.nc"  % dz_ID
        nc = netCDF4.Dataset(filepath,'a')
        
        # loading the dz data from the nc file
        dz = nc.variables['dz_array']
        t = nc.variables['time']
        z = nc.variables['row']
        x = nc.variables['column']
        # print information about deltaN2 dataset
        print "variables  of the nc file :", nc.variables.keys()
        print "deltaN2 shape : " , dz.shape
        print "t  shape : " , t.shape
        print "z shape : " , z.shape
        print "x shape : " , x.shape
        
        # call get_info function from Energy_flux program :: to get info!
        
        sql = """ SELECT video_id  FROM dz WHERE dz_id = %d""" % dz_ID
        rows = db.execute(sql)
        video_id = rows[0][0]
        
        # calculate dt
        dt = numpy.mean(numpy.diff(t))
        print "dt :",dt
        
    
        # get the window length and window height
        sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
        rows = db.execute(sql)
        win_l = rows[0][0]
    
        win_l=win_l*1.0
        print "lenght" , win_l
        
        # calculate dn2t from dz
        n_water = 1.3330
        n_air = 1.0
        L_tank = 453.0
        gamma = 0.0001878
        #deltaN2 
        a = -1.0/(gamma * ((0.5*L_tank*L_tank)+(L_tank*win_l*n_water)))
        print "a:",a
        
        #call the function to create the nc file in which we are going to store the dn2t array
        dn2t_filename = createncfile(dz_ID,diff_frames,t,x,z)
        
        # open the dn2t nc file for appending data
        nc=netCDF4.Dataset(dn2t_filename,'a')
        dn2t = nc.variables['dn2t_array']
         
        for num in range(dz.shape[0]):
            var1 = a*(dz[num])/dt
            print "appending frame %d" % num
            append2ncfile(dn2t,var1,num)
        print "done...!"
    return dn2t_filename 

def UI():
    """
    take the dz_id from the user and calculate the change in the squared
    buoyancy frequency (deltaN2), the time derivative of the deltaN2, U, W, and
    the energy flux
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dz_ID", type=int, help="enter the dz_id for which to compute the dn2t fields")
    args = parser.parse_args()
    a_xi_filename = compute_a_xi(args.dz_ID)
    print a_xi_filename

if __name__ == "__main__":
    UI()
