import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
import labdb
import os

# Open database for access
db  = labdb.LabDB()


def get_MSF(id):
    #get the mintol,sigma, and filter_size of the deltaN2
    sql = """ SELECT mintol,sigma,filter_size FROM deltaN2 WHERE id= (SELECT
    deltaN2_id FROM vertical_displacement_amplitude WHERE a_xi_id=%d""" % id
    rows = db.execute(sql)
    print "rows:",rows
    mintol= rows[0][0]
    sigma = rows[0][1]
    filter_size = rows[0][2]
    print " mintol:" ,mintol, "sigma:" ,sigma, "filter_size:",filter_size
    return mintol,sigma,filter_size

def compute_energy_flux(a_xi_id,column,max_min,plotname = "axiVerticalTimeSeries"): 
    # Check if the database already exists
    sql = """SELECT expt_id FROM dz WHERE dz_id = (SELECT dz_id from \
            vertical_displacement_amplitude WHERE a_xi_id = %d)""" % a_xi_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print "expt ID: ", expt_id

    if (len(rows) == 0):
        print "The a_xi for the expt_id is not yet computed.. "
        return


    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD4/vertical_displacement_amplitude/%d/a_xi.nc" % a_xi_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    a_xi_arr = nc.variables['a_xi_array']
    print " a_xi array.shape ", a_xi_arr.shape
    #t0=10
    t = nc.variables['time'][:]
    z = nc.variables['row']
    x = nc.variables['column']

    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print "kx:",kx
    print "const:",const
    dat = a_xi_arr[:,:,column]
    data = dat.T
    print "data.T.shape", data.shape
    plt.figure(figsize=(15,10))
    #plt.contourf(t,z,data.T[::-1],levels=level)
    plt.imshow(data[:,:],extent=[t[0],t[-1],z[-1],z[0]],vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.title('Vertical Displacement Amplitude of Video ID %d' % (video_id) )
    plt.colorbar()
    plt.savefig(plotname + "_%dcol.pdf"% column )

def get_info(expt_id):
    
    # Get the video_id for the corresponding expt.id
    sql = """ SELECT video_id FROM video_experiments WHERE expt_id = %d """ % expt_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    print "VIDEO ID = " ,video_id

    # Get the Buoyancy frequency
    sql = """ SELECT N_frequency FROM stratification WHERE strat_id =\
            (SELECT strat_id FROM stratification_experiments WHERE expt_id = %d
            \
                    ORDER BY strat_id ASC LIMIT 1\
            )""" % expt_id
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
    parser.add_argument("column",type = int,help = "Enter the pixel of the column (0-1291) whose timeseries you want to see")
    parser.add_argument("max_min", type =float, help = "Enter the value to be used as vmax and vmin")

    args = parser.parse_args()
    compute_energy_flux(args.a_xi_id, args.column,args.max_min)
    plt.show()

if __name__ == "__main__":
    UI()



