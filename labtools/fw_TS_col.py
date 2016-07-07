import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
from . import labdb
import os
from . import plotting_functions

# Open database for access
db  = labdb.LabDB()


def compute_energy_flux(a_xi_id,row_s,row_e,col1,rstop=0,lstop=0):
    db = labdb.LabDB()
    
    #check if the file already exists
    sql = """ SELECT fw_id FROM filtered_waves WHERE a_xi_id = %d""" %a_xi_id
    rows=db.execute(sql)
    
    if len(rows) == 0:
        print("It has not been computed")
        return

    fw_path = "/Volumes/HD4/filtered_waves/%d/waves.nc" % rows[0][0]
    print(fw_path)
    
    # Get experiment ID
    
    sql = """ SELECT  dz.expt_id FROM dz INNER JOIN \
            vertical_displacement_amplitude ON (dz.dz_id = \
            vertical_displacement_amplitude.dz_id AND \
            vertical_displacement_amplitude.a_xi_id = %d) """ %a_xi_id
    rows = db.execute(sql)
    print(rows) 
    expt_id = rows[0][0]
    print(" experiment ID : ", expt_id)

    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    print("Vid ID: ",video_id,"N: ",  N_frequency,"omega: ", omega,"kz: ", kz, "theta: ",theta)
    # Open the nc file for reading data
    nc = netCDF4.Dataset(fw_path,'r')
    raw = nc.variables['raw_array']
    ft = nc.variables['time'][:]
    fz = nc.variables['row'][row_s:row_e]
    fx = nc.variables['column']
    # print information about dz dataset
    print("variables  of the nc file :", list(nc.variables.keys()))
    print("t  shape : " , ft.shape)
    
    #select the timeseries of the rows along the column you are interested in 
    raw = nc.variables['raw_array'][row_s:row_e,:,col1]
    raw_squared = raw**2
    
    print("raw_squared shape: ",raw_squared.shape)

    print("depth from :: ", fz[0], " to ", fz[-1])
    print("time from :: ", ft[0], " to ", ft[-1])
    
    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print("kx:",kx)
    print("const:",const)
    
    EF1 = (numpy.mean(raw_squared,0)) * const
    print("EF arrays shape:: ", EF1.shape)
    
    # get dt and length of the timeseries
    dt = numpy.mean(numpy.diff(ft))
    nt = len(ft)
    print("dt :: ",dt,"nt ::", nt)
    
    #average rightward energy
    rstop = rstop*1.0/dt
    print(rstop)
    rightE = numpy.mean(EF1[0:rstop])
    print("rightE" , rightE)
    #average reflected energy
    lstop = lstop *1.0/dt
    print(lstop)
    leftE = numpy.mean(EF1[rstop:lstop])
    print("leftE", leftE)
    
    # get the moving average
    window = 2*numpy.pi/(omega*dt)
    window = numpy.int16(window)
    print(window)
    avg1 = moving_average(EF1,window)
    #print "raw energy average : ", avg1[-1],numpy.max(avg1)
    plt.plot(ft,EF1,ft,avg1)
    plt.show()


def growing_average(arr):
    sum_arr=[]
    window = arr.shape[0]
    print("input shape",arr.shape)
    sum_arr.append(arr[0:])
    i=1
    print("in loop 1")
    while (i < window):
        sum_arr.append(numpy.pad(arr[:-i], (i,0),'constant',constant_values=(0,)))
        i+=1
    sum_arr = numpy.array(sum_arr)
    total = numpy.sum(sum_arr,0)
    print("in loop 2")
    for i in range(window):
        total[i] = 1.0*total[i]/(i+1)
    print("shape",sum_arr.shape)
    print("average shape: ", total.shape)
    return total


def moving_average(arr,window):
    sum_arr=[]
    print("input shape",arr.shape)
    sum_arr.append(arr)
    i=1
    while (i < window):
        sum_arr.append(numpy.pad(arr[:-i], (i,0),'constant',constant_values=(0,)))
        i+=1

    sum_arr = numpy.array(sum_arr)
    print("shape",sum_arr.shape)
    avg= numpy.sum(sum_arr,0)/window
    print("average shape: ", avg.shape)
    return avg


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
    parser.add_argument("a_xi_id",type=int,help="Enter the Vertical Displacement Amplitude ID \
            of the Video you want to see the Energy Flux of..")
    parser.add_argument("row_s",type = int, help = " start pixel of Z ")
    parser.add_argument("row_e",type = int, help = " end pixel of Z ")
    parser.add_argument("--R_stop",type = int,default=0,help= "time without reflection ")
    parser.add_argument("--L_stop",type = int,default=0,help= "time with reflection")
    parser.add_argument("col1",type = int, help = " pixel number of the column \
            whose time series we want to see")
    args = parser.parse_args()
    compute_energy_flux(args.a_xi_id,args.row_s,args.row_e,args.col1,args.R_stop,args.L_stop)
    plt.show()


if __name__ == "__main__":
    UI()



