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


def createncfile(dn2t_id,t,x,z):
    """ 
    create the nc file in which we will store a_xi array.
    create a row in the database for the nc file stored
    update the database
    """
    print "inside createncfile function"
    db = labdb.LabDB()
    #create the directory in which to store the nc file
    sql = """INSERT into vertical_displacement_amplitude (dn2t_id) VALUES (%d)""" %(dn2t_id)  
    print sql
    db.execute(sql)
    sql = """SELECT LAST_INSERT_ID()""" 
    rows = db.execute(sql)
    a_xi_id = rows[0][0]
    a_xi_path = "/Volumes/HD3/vertical_displacement_amplitude/%d" % a_xi_id 
    os.mkdir(a_xi_path)

    a_xi_filename = os.path.join(a_xi_path,"a_xi.nc")
    print "A xi : ",a_xi_filename


    # Declare the nc file for the first time
    nc = netCDF4.Dataset(a_xi_filename,'w',format = 'NETCDF4')
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
    a_xi = nc.createVariable('a_xi_array',numpy.float32,('time','row','column'))
    print nc.dimensions.keys() ,a_xi.shape,a_xi.dtype

    # assign the values
    TIME[:] = t[:]
    ROW[:] = z
    COLUMN[:] = x

    nc.close()
    db.commit()
    return a_xi_filename


def append2ncfile(a_xi,var,num):
    """
    Append the array to the end of the nc file
    """
    print "appending.."
    a_xi[num] = var

    
def compute_a_xi(dn2t_id):
    # access database
    db = labdb.LabDB()

    #check if the dataset already exists
    sql = """SELECT a_xi_id FROM vertical_displacement_amplitude WHERE\
             dn2t_id = %d""" % (dn2t_id)
    rows = db.execute(sql)

    if len(rows) > 0:
    
        # A xi array already computed
        id = rows[0][0]
        print "loading a_xi %d dataset  .." % id

        #load the array from disk
        a_xi_path = "/Volumes/HD3/vertical_displacement_amplitude/%d/" % id
        a_xi_filename = a_xi_path + 'a_xi.nc'

    else:

        #  open the dataset dn2t.nc for calculating a_xi
        filepath = "/Volumes/HD3/dn2t/%d/dn2t.nc"  % dn2t_id
        nc = netCDF4.Dataset(filepath,'a')
        
        # loading the deltaN2 data from the nc file
        dn2t = nc.variables['dn2t_array']
        t = nc.variables['time']
        z = nc.variables['row']
        x = nc.variables['column']
        # print information about deltaN2 dataset
        print "variables  of the nc file :", nc.variables.keys()
        print "dn2t shape : " , dn2t.shape
        print "t  shape : " , t.shape
        print "z shape : " , z.shape
        print "x shape : " , x.shape
        
        # call get_info function from Energy_flux program :: to get info!
        
        sql = """ SELECT expt_id  FROM dz WHERE dz_id = (SELECT dz_id FROM dn2t WHERE id = %d) """ % dn2t_id
        rows = db.execute(sql)
        expt_id = rows[0][0]

        vid_id, N, omega,kz,theta = Energy_flux.get_info(expt_id)

        print "V_ID:",  vid_id,"\n N:", N, "\n OMEGA: ", omega,"\n kz:",kz,"\n theta : ", theta
        # calculate dt
        dt = numpy.mean(numpy.diff(t))
        print "dt of dn2t:",dt
        
        #call the function to create the nc file in which we are going to store the dn2t array
        a_xi_filename = createncfile(dn2t_id,t,x,z)
        
        # open the a_xi nc file for appending data
        nc=netCDF4.Dataset(a_xi_filename,'a')
        a_xi = nc.variables['a_xi_array']
         
        # Calculate kx 
        rho0 = 0.998
        kx = (omega * kz)/(N*N - omega*omega)**0.5
        const1 = -1.0 * omega* N * N * kz
        
        for num in range(dn2t.shape[0]):
            var1 = 1.0*dn2t[num]/const1
            print "appending frame %d" % num
            append2ncfile(a_xi,var1,num)
        print "done...!"
    return a_xi_filename

def test():
    nc = netCDF4.Dataset('/Volumes/HD3/dn2t/3/dn2t.nc','r')
    dn2t = nc.variables['dn2t_array']
    Eflux1 = dn2t[:,200,:]*4.125
    t = nc.variables['time']
    x = nc.variables['column']
    z = nc.variables['row']
    
    Eflux1 = numpy.ma.masked_array(Eflux1,Eflux1 < 0.05)
    Eflux1 = numpy.ma.masked_array(Eflux1,Eflux1 < -0.05)
    
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(Eflux1.T,extent=[x[0],x[-1],t[0],t[-1]],vmin=-.20,vmax=0.20, interpolation = 'nearest', aspect = 'auto')
    plt.colorbar()
    plt.xlabel('length')
    plt.ylabel('time')
    plt.title('time series energy flux (single row)')
    Eflux2= dn2t[:,:,700] * 4.125
    Eflux2 = numpy.ma.masked_array(Eflux2,Eflux2 < 2.0) 
    Eflux21= dn2t[:,:,100] * 4.125
    Eflux21 = numpy.ma.masked_array(Eflux21,Eflux21 < 2.0)
    Eflux22= dn2t[:,:,300] * 4.125
    Eflux22 = numpy.ma.masked_array(Eflux22,Eflux22 < 2.0) 
    Eflux23= dn2t[:,:,500] * 4.125
    Eflux23 = numpy.ma.masked_array(Eflux23,Eflux23 < 2.0) 
    Eflux24= dn2t[:,:,900] * 4.125
    Eflux24 = numpy.ma.masked_array(Eflux24,Eflux24 < 2.0) 
    Eflux25= dn2t[:,:,1100] * 4.125
    Eflux25 = numpy.ma.masked_array(Eflux25,Eflux25 < 2.0) 
    

    plt.subplot(1,4,2)
    plt.imshow(Eflux2,extent=[t[0],t[-1],z[0],z[-1]],vmin=-.50,vmax=0.50, interpolation = 'nearest', aspect = 'auto')
    plt.colorbar()
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title(' time series energy flux (single column)')
    Eflux3= dn2t[400,:,:] * 4.125
    plt.subplot(1,4,3)
    plt.imshow(Eflux3.T,extent=[x[0],x[-1],z[0],z[-1]],vmin=-.50,vmax=0.50, interpolation = 'nearest', aspect = 'auto')
    plt.colorbar()
    plt.xlabel('length')
    plt.ylabel('depth')
    plt.title('energy flux - single instant in time ')
    plt.subplot(1,4,4)
    plt.plot(t[:],numpy.mean(Eflux2,1),t[:],numpy.mean(Eflux21,1),t[:],numpy.mean(Eflux22,1),t[:],numpy.mean(Eflux24,1),t[:],numpy.mean(Eflux25,1))
    plt.xlabel('length')
    plt.ylabel('depth')
    plt.title('columnwise averaged  energy flux ')



    plt.show()

def UI():
    """
    take the dz_id from the user and calculate the change in the squared
    buoyancy frequency (deltaN2), the time derivative of the deltaN2, U, W, and
    the energy flux
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dn2t_id", type=int, help="enter the dN2t_id")
    args = parser.parse_args()
    a_xi_filename = compute_a_xi(args.dn2t_id)
    print a_xi_filename

if __name__ == "__main__":
    #test()
    UI()
