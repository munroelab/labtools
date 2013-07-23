import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
import labdb
import os
import plotting_functions

# Open database for access
db  = labdb.LabDB()

def compute_energy_flux(a_xi_id):
    """ this function computes the energy flux from a_xi.nc file and displays it"""
    # Open the a_xi.nc file and load the variables
    # Check if the database already exists
    #sql = """ SELECT a_xi_id FROM vertical_displacement_amplitude WHERE\
    #        deltaN2_id = (SELECT id FROM deltaN2 WHERE expt_id = %d )""" % expt_id
    
    sql = """SELECT a_xi_id FROM vertical_displacement_amplitude WHERE a_xi_id = %d""" % a_xi_id
    rows = db.execute(sql)
    a_xi_id = rows[0][0]

    if (len(rows) == 0):
        print "The a_xi for the expt_id is not yet computed.. "
        return
    # get experiment ID
    sql = """SELECT expt_id FROM deltaN2 WHERE id = (SELECT deltaN2_id FROM\
            vertical_displacement_amplitude WHERE a_xi_id = %d )  """ % a_xi_id
    rows=db.execute(sql)
    expt_id = rows[0][0]

    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD3/vertical_displacement_amplitude/%d/a_xi.nc" % a_xi_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    #a_xi = nc.variables['a_xi_array'][:,120:900,:]
    #print a_xi.shape
    a1_xi = nc.variables['a_xi_array'][0:835,200:900,200]
    a1_xi_squared = a1_xi**2
    print "A xi shape: ",a1_xi.shape
    a2_xi = nc.variables['a_xi_array'][0:835,200:900,700]
    a2_xi_squared = a2_xi**2
    print "A xi shape: ",a2_xi.shape
    a3_xi = nc.variables['a_xi_array'][0:835,200:900,1200]
    a3_xi_squared = a3_xi**2
    #a1_xi = a_xi[:,:,800]
    #print "A xi shape: ",a1_xi.shape
    #a2_xi = a_xi[:,:,900]
    #print "A xi shape: ",a2_xi.shape
    #a3_xi = a_xi[:,:,1200]
    #print "A xi shape: ",a3_xi.shape

    t = nc.variables['time'][0:835]
    z = nc.variables['row']
    x = nc.variables['column']

    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print "kx:",kx
    print "const:",const
    
    EF1 = (numpy.mean(a1_xi_squared,1)) * const
    print EF1.shape
    EF2 = (numpy.mean(a2_xi_squared,1)) * const
    print EF2.shape
    EF3 = (numpy.mean(a3_xi_squared,1)) * const
    print EF3.shape
    #EF1 = a1_xi_squared * const
    #ef11 = numpy.mean(EF1,1)
    #print EF1.shape
    #EF2 = a2_xi_squared * const
    #ef22 = numpy.mean(EF2,1)

    #EF3 = a3_xi_squared * const
    #ef33 = numpy.mean(EF3,1)
    # call the plotting function
    title1 = "energy flux at %d cm away from the wavemaker" % (x[200]+110)
    title2 = "energy flux at %d cm away from the wavemaker" % (x[700]+110)
    title3 = "energy flux at %d cm away from the wavemaker" % (x[1200]+110)
    plt.figure(1,figsize=(17,13))
    plotting_functions.plot_3plts(EF1,EF2,EF3,t,title1,title2,title3,'time','E')
    #plt.show()
    
    
    
    #plotting_functions.plot_ts(EF1,t,1,title)
    
    #EF = (a_xi[:,:,500])**2 * const
    #print EF.shape
    #title = "energy flux at %d cm away from the left edge of the window" % x[500]
    #plotting_functions.plot_ts(numpy.mean(EF,1),t,2,title)
    #EF = (a_xi[:,:,800])**2 * const
    #print EF.shape
    #title = "energy flux at %d cm away from the left edge of the window" % x[800]
    #   plotting_functions.plot_ts(numpy.mean(EF,1),t,3,title)
    #plt.figure()
    #plt.plot(t, EF)
    #plt.figure(2)
    #plt.imshow(a_xi.T,vmax=6,vmin=-6)
    #plt.colorbar()
    #plt.show()

    return

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
    parser.add_argument("a_xi_id",type=int,help="Enter the Vertical Displacement Amplitude ID \
            of the Video you want to see the Energy Flux of..")
    args = parser.parse_args()
    compute_energy_flux(args.a_xi_id)
    plt.show()


if __name__ == "__main__":
    UI()



