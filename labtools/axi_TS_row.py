import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
import labdb
import os

# Open database for access
db  = labdb.LabDB()

def compute_energy_flux(a_xi_id,row_no,max_min,plotname = "axiHorizontalTimeSeries"): 
    """
    given a_xi_id it computes the Energy flux and plots it and saves it as a pdf
    file
    """ 

    #check if the nc file exists
    print "entering compute_energy_flux function...."

    sql = """SELECT dz_id FROM vertical_displacement_amplitude WHERE a_xi_id= %d """ % a_xi_id
    print sql
    row = db.execute_one(sql)
    print "axi HTS .... row", row
    if (row[0] is None):
        print "The a_xi for the expt_id is not yet computed.. "
        return
    # get the expt_id 
    sql = """SELECT expt_id from dz where dz_id =\
            (SELECT dz_id FROM vertical_displacement_amplitude WHERE a_xi_id = %d) """ % a_xi_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print "expt ID: ", expt_id

    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD4/vertical_displacement_amplitude/%d/a_xi.nc" % a_xi_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    a_xi_arr = nc.variables['a_xi_array']
    print " a_xi array.shape ", a_xi_arr.shape
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']

    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print "kx:",kx
    print "const:",const
    data = a_xi_arr[:,row_no,:]
    
    print "data.T.shape", data.shape
    plt.figure()
    #plt.contourf(t,z,data.T[::-1],levels=level)
    plt.imshow(data[:,:],extent=[x[0],x[-1],t[-1],t[0]],vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel(' window length (cm)')
    plt.ylabel('Time (seconds)')
    plt.title('timeseries (A_xi-Video ID %d) \n At depth: %.1f cm '\
                    % (video_id, z[row_no], ) )
    plt.colorbar()
    plt.savefig(plotname)

def get_info(expt_id):
    
    # Get the video_id for the corresponding expt.id
    sql = """ SELECT video_id FROM video_experiments WHERE expt_id = %d """ % expt_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    print "VIDEO ID = " ,video_id

    # Get the Buoyancy frequency
    sql = """ SELECT N_frequency FROM stratification WHERE strat_id =\
            (SELECT strat_id FROM stratification_experiments WHERE \
            expt_id = %d ORDER BY strat_id ASC LIMIT 1)""" % expt_id
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
    parser.add_argument("a_xi_id",type=int,help="Enter the a_xi ID")
    parser.add_argument("row_no",type = int,help = "Enter the row pixel number (0-963) whose timeseries you want to see")
    parser.add_argument("max_min", type =float, help = "Enter the value to be used as vmax and vmin")

    args = parser.parse_args()
    
    compute_energy_flux(args.a_xi_id, args.row_no,args.max_min)
    plt.show()
    

if __name__ == "__main__":
    UI()



