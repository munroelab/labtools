import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
from . import labdb
import os

# Open database for access
db  = labdb.LabDB()

def dz_row_timeseries(dz_id,row_no,max_min,plot_name='dz_HTS.pdf'):

    # Check if the database already exists
    sql = """ SELECT expt_id  FROM dz WHERE dz_id = %d""" % dz_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print("expt ID: ", expt_id)

    if (len(rows) == 0):
        print("The dz for the expt_id is not yet computed.. ")
        return


    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/data/dz/%d/dz.nc" % dz_id
    nc = netCDF4.Dataset(path,'r')
    print(" variables: " ,list(nc.variables.keys()))
    dz_arr = nc.variables['dz_array']
    print(" dz_array.shape ", dz_arr.shape)
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']

    data = dz_arr[:,row_no,:]
    

    plt.figure()
    #plt.contourf(t,z,data.T[::-1],levels=level)
    plt.imshow(data[:,:],extent=[x[0],x[-1],t[-1],t[0]],vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel('window length (cm)')
    plt.ylabel('Time (seconds)')
    plt.title('dz timeseries of row %d (depth %.1f cm)' %(row_no,z[row_no]))
    plt.colorbar()
    plt.savefig(plot_name+'row%d.pdf' % row_no)
#    plt.show()


def get_info(expt_id):
    
    # Get the video_id for the corresponding expt.id
    sql = """ SELECT video_id FROM video_experiments WHERE expt_id = %d """ % expt_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    print("VIDEO ID = " ,video_id)

    # Get the Buoyancy frequency
    sql = """ SELECT N_frequency FROM stratification WHERE strat_id =\
            (SELECT strat_id FROM stratification_experiments WHERE expt_id = %d )""" % expt_id
    rows = db.execute(sql)
    N_frequency = rows[0][0]
    print("Buoyancy Frequency: ", N_frequency)

    # Get the frequency,kz and calculate theta
    sql = """ SELECT kz , frequency_measured FROM wavemaker WHERE wavemaker_id \
            = (SELECT wavemaker_id FROM wavemaker_experiments WHERE expt_id = %d) """% expt_id
    rows = db.execute(sql)
    kz = rows[0][0]
    f = rows[0][1]
    print("kz : ",kz, "frequency:" ,f)
    omega = 2 * numpy.pi * f

    # wkt w = N * cos(theta) ,arccos returns in radians 
    theta = numpy.arccos(omega/N_frequency) 
    print("omega: ", omega, "\t theta: " ,theta*180/numpy.pi)

    return video_id, N_frequency, omega, kz, theta*180.0/numpy.pi

def UI():
    """ take the experiment id from the user and calculate the energy flux from
    the vertical displacement amplitude that has already been calculated and
    plot it """
    parser = argparse.ArgumentParser()
    parser.add_argument("dz_id",type=int,help="Enter the dz ID")
    parser.add_argument("row_no",type = int,help = "Enter the pixel of the row (0-963) whose timeseries you want to see")
    parser.add_argument("max_min", type =float, help = "Enter the value to be used as vmax and vmin")
    args = parser.parse_args()
    compute_energy_flux(args.dz_id, args.row_no, args.max_min)

if __name__ == "__main__":
    UI()



