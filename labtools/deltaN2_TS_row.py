import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
import labdb
import os

# Open database for access
db  = labdb.LabDB()

def compute_energy_flux(deltaN2_id,row_no,max_min): 
    # Check if the database already exists
    sql = """ SELECT expt_id  FROM deltaN2 WHERE id = %d""" % deltaN2_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print "expt ID: ", expt_id

    if (len(rows) == 0):
        print "The deltaN2 for the expt_id is not yet computed.. "
        return


    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD3/deltaN2/%d/deltaN2.nc" % deltaN2_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    deltaN2_arr = nc.variables['deltaN2_array']
    print " deltaN2_array.shape ", deltaN2_arr.shape
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']

    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print "kx:",kx
    print "const:",const
    data = deltaN2_arr[:,row_no,:]
    
    data_masked = numpy.ma.masked_array(data,(data**2)**0.5>0.0004)
    data_masked.fill_value = 0.0
    print "data.T.shape", data.shape
    plt.figure()
    #plt.contourf(t,z,data.T[::-1],levels=level)
    plt.imshow(data[::-1,:],extent=[x[0],x[-1],t[0],t[-1]],vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel('windw length (cm)')
    plt.ylabel('Time (seconds)')
    plt.title('Vid ID: %d, deltaN2 timeseries of row %d (depth %.1f cm)' %\
            (video_id,row_no,z[row_no]))
    plt.colorbar()
    plt.show()


def get_info(expt_id):
    
    # Get the video_id for the corresponding expt.id
    sql = """ SELECT video_id FROM video_experiments WHERE expt_id = %d """ % expt_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    print "VIDEO ID = " ,video_id

    # Get the Buoyancy frequency
    sql = """ SELECT N_frequency FROM stratification WHERE strat_id =\
            (SELECT strat_id FROM stratification_experiments WHERE expt_id = %d )""" % expt_id
    rows = db.execute(sql)
    N_frequency = rows[0][0]
    print "Buoyancy Frequency: ", N_frequency

    # Get the frequency,kz and calculate theta
    sql = """ SELECT kz , frequency_measured FROM wavemaker WHERE wavemaker_id \
            = (SELECT wavemaker_id FROM wavemaker_experiments WHERE expt_id = %d) """% expt_id
    rows = db.execute(sql)
    kz = rows[0][0]
    f = rows[0][1]
    print "kz : ",kz, "frequency:" ,f
    omega = 2 * numpy.pi * f

    # wkt w = N * cos(theta) ,arccos returns in radians 
    theta = numpy.arccos(omega/N_frequency) 
    print "omega: ", omega, "\t theta: " ,theta*180/numpy.pi

    return video_id, N_frequency, omega, kz, theta*180.0/numpy.pi

def UI():
    """ take the experiment id from the user and calculate the energy flux from
    the vertical displacement amplitude that has already been calculated and
    plot it """
    parser = argparse.ArgumentParser()
    parser.add_argument("deltaN2_id",type=int,help="Enter the deltaN2 ID")
    parser.add_argument("row_no",type = int,help = "Enter the pixel of the row (0-963) whose timeseries you want to see")
    parser.add_argument("max_min", type =float, help = "Enter the value to be used as vmax and vmin")
    args = parser.parse_args()
    compute_energy_flux(args.deltaN2_id, args.row_no, args.max_min)

if __name__ == "__main__":
    UI()



