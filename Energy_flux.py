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


def compute_energy_flux(a_xi_id,row_s,row_e,col1,plotname="energyflux"):
    db = labdb.LabDB()
    
    #check if the file already exists
    sql = """ SELECT fw_id FROM filtered_waves WHERE a_xi_id = %d""" %a_xi_id
    rows=db.execute(sql)
    
    if len(rows) == 0:
        print "It has not been computed"
        return

    fw_path = "/Volumes/HD4/filtered_waves/%d/waves.nc" % rows[0][0]
    print fw_path
    
    # Get experiment ID
    
    sql = """ SELECT  dz.expt_id FROM dz INNER JOIN \
            vertical_displacement_amplitude ON (dz.dz_id = \
            vertical_displacement_amplitude.dz_id AND \
            vertical_displacement_amplitude.a_xi_id = %d) """ %a_xi_id
    rows = db.execute(sql)
    print rows 
    expt_id = rows[0][0]
    print " experiment ID : ", expt_id

    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    print "Vid ID: ",video_id,"N: ",  N_frequency,"omega: ", omega,"kz: ", kz, "theta: ",theta
    # Open the nc file for reading data
    nc = netCDF4.Dataset(fw_path,'r')
    raw = nc.variables['raw_array']
    left = nc.variables['left_array']
    right = nc.variables['right_array']
    ft = nc.variables['time'][:]
    fz = nc.variables['row'][row_s:row_e]
    fx = nc.variables['column']
    # print information about dz dataset
    print "variables  of the nc file :", nc.variables.keys()
    print "left_w shape : " , left.shape
    print "right_w shape : " , right.shape
    print "t  shape : " , ft.shape
    
    #select the timeseries of the rows along the column you are interested in 
    raw = nc.variables['raw_array'][row_s:row_e,:,col1]
    raw_squared = raw**2
    left = nc.variables['left_array'][row_s:row_e,:,col1]
    left_squared = left**2
    right = nc.variables['right_array'][row_s:row_e,:,col1]
    right_squared = right**2
    
    print "raw_squared shape: ",raw_squared.shape
    print "left_squared shape: ",left_squared.shape
    print "right_squared shape:  ", right_squared.shape
    print raw[300:310,30]
    print raw_squared[300:310,30]

    print "depth from :: ", fz[0], " to ", fz[-1]
    print "time from :: ", ft[0], " to ", ft[-1]
    
    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print "kx:",kx
    print "const:",const
    
    EF1 = (numpy.mean(raw_squared,0)) * const
    print "EF arrays shape:: ", EF1.shape
    EF2 = (numpy.mean(left_squared,0)) * const
    print EF2.shape
    EF3 = (numpy.mean(right_squared,0)) * const
    print EF3.shape
    
    # get dt and length of the timeseries
    dt = numpy.mean(numpy.diff(ft))
    nt = len(ft)
    print "dt :: ",dt,"nt ::", nt
    
    # perform fft along the time axis
    F1 = numpy.fft.fft(EF1)/nt*nt
    print "F1 shape::", F1.shape
    F2 = numpy.fft.fft(EF2)/nt*nt
    print "F2 shape::", F2.shape
    F3 = numpy.fft.fft(EF3)/nt*nt
    print "F3 shape::", F3.shape
    
    # normalize
    F1 = numpy.fft.fftshift(F1)
    print "after shifting fft F1 shape::", F1.shape
    F2 = numpy.fft.fftshift(F2)
    print "after shifting fft F2 shape::", F2.shape
    F3 = numpy.fft.fftshift(F3)
    print "after shifting fft F3 shape::", F3.shape

    #Frequency axis
    freq = numpy.fft.fftfreq(nt,dt)
    print "freq.shape" ,freq.shape
    freq = numpy.fft.fftshift(freq) * 2*numpy.pi
    print "after shifting freq.shape ::",freq.shape
    
    # get the moving average
    window = 2*numpy.pi/(omega*dt)
    window = numpy.int16(window)
    print window
    avg1 = moving_average(EF1,window)
    avg2 = moving_average(EF2,window)
    avg3 = moving_average(EF3,window)
    #print "raw energy average : ", avg1[-1],numpy.max(avg1)
    #print "rightward Energy average: ", avg3[-1]
    #print "leftward energy average: ", avg2[-1]
    
    #  plotting outouts
    title1 = "Energy Flux - Original data ( %dcm from the wave generator) " %(56+fx[col1])
    title2 = "Energy Flux - Rightward waves " 
    title3 = "Energy Flux - Leftward waves " 
    fig1 = plt.figure(1,figsize=(15,12))
    fig1.patch.set_facecolor('white')
    plotting_functions.sharexy_plot_6plts(EF1,EF3,EF2,avg1,avg3,avg2,ft,title1,title2,title3,'time','E')
    plt.savefig(plotname + "_ef.pdf")

    
    title1 = "FFT of energy flux at %d cm of the raw data" % (56+fx[col1])
    title3 = "FFT(energy flux) of the leftward propagating wave" 
    title2 = "FFT(energy flux) of the rightward propagating wave" 
    fig2=plt.figure(2,figsize=(15,12))
    fig2.patch.set_facecolor('white')
    plotting_functions.sharexy_plot_3plts(abs(F1),abs(F3),abs(F2),freq,title1,title2,title3,'freq','abs(fft(EF))')
    plt.savefig(plotname + "_fft.pdf")
    plt.show()

def growing_average(arr):
    sum_arr=[]
    window = arr.shape[0]
    print "input shape",arr.shape
    sum_arr.append(arr[0:])
    i=1
    print "in loop 1"
    while (i < window):
        sum_arr.append(numpy.pad(arr[:-i], (i,0),'constant',constant_values=(0,)))
        i+=1
    sum_arr = numpy.array(sum_arr)
    total = numpy.sum(sum_arr,0)
    print "in loop 2"
    for i in range(window):
        total[i] = 1.0*total[i]/(i+1)
    print "shape",sum_arr.shape
    print "average shape: ", total.shape
    return total


def moving_average(arr,window):
    sum_arr=[]
    print "input shape",arr.shape
    sum_arr.append(arr)
    i=1
    while (i < window):
        sum_arr.append(numpy.pad(arr[:-i], (i,0),'constant',constant_values=(0,)))
        i+=1

    sum_arr = numpy.array(sum_arr)
    print "shape",sum_arr.shape
    avg= numpy.sum(sum_arr,0)/window
    print "average shape: ", avg.shape
    return avg





# old compute function for the axi datasets upto 119
def old_compute_energy_flux(a_xi_id,row_s,row_e,col1,col2,col3):
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
    #sql = """SELECT expt_id FROM deltaN2 WHERE id = (SELECT deltaN2_id FROM\
    #        vertical_displacement_amplitude WHERE a_xi_id = %d )  """ % a_xi_id
    sql = """select expt_id from dz where dz_id = (select dz_id from dn2t where\
            id= (select dn2t_id from vertical_displacement_amplitude where a_xi_id=%d))""" % a_xi_id
    
    rows=db.execute(sql)
    expt_id = rows[0][0]

    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD4/vertical_displacement_amplitude/%d/a_xi.nc" % a_xi_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    #a_xi = nc.variables['a_xi_array'][:,120:900,:]
    #print a_xi.shape
    a1_xi = nc.variables['a_xi_array'][:,row_s:row_e,col1]
    a1_xi_squared = a1_xi**2
    a2_xi = nc.variables['a_xi_array'][:,row_s:row_e,col2]
    a2_xi_squared = a2_xi**2
    a3_xi = nc.variables['a_xi_array'][:,row_s:row_e,col3]
    a3_xi_squared = a3_xi**2
    
    print "A1 xi shape: ",a1_xi.shape
    print "A2 xi shape: ",a2_xi.shape
    print "A3 xi shape: ",a3_xi.shape

    t = nc.variables['time'][:]
    z = nc.variables['row'][row_s:row_e]
    print "depth from :: ", z[0], " to ", z[-1]

    x = nc.variables['column']
    

    # Calculate kx and  energy flux
    rho0 = 0.998
    kx = (omega * kz)/(N_frequency**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N_frequency**3) * numpy.cos(theta*numpy.pi/180) * (numpy.sin(theta*numpy.pi/180))**2
    print "kx:",kx
    print "const:",const
    
    EF1 = (numpy.mean(a1_xi_squared,1)) * const
    print "EF arrays shape:: ", EF1.shape
    EF2 = (numpy.mean(a2_xi_squared,1)) * const
    print EF2.shape
    EF3 = (numpy.mean(a3_xi_squared,1)) * const
    print EF3.shape
    
    # get dt and length of the timeseries
    dt = numpy.mean(numpy.diff(t))
    nt = len(t)
    print "dt :: ",dt,"nt ::", nt
    
    # perform fft along the time axis
    F1 = numpy.fft.fft(EF1)
    print "F1 shape::", F1.shape
    F2 = numpy.fft.fft(EF2)
    print "F2 shape::", F2.shape
    F3 = numpy.fft.fft(EF3)
    print "F3 shape::", F3.shape
    
    # normalize
    F1 = numpy.fft.fftshift(F1)/(nt*nt)
    print "after shifting fft F1 shape::", F1.shape
    F2 = numpy.fft.fftshift(F2)/(nt*nt)
    print "after shifting fft F2 shape::", F2.shape
    F3 = numpy.fft.fftshift(F3)/(nt*nt)
    print "after shifting fft F3 shape::", F3.shape

    #Frequency axis
    freq = numpy.fft.fftfreq(nt,dt)
    print "freq.shape" ,freq.shape
    freq = numpy.fft.fftshift(freq) * 2*numpy.pi
    print "after shifting freq.shape ::",freq.shape


    
    """EF1 = a1_xi_squared * const
    ef11 = numpy.mean(EF1,1)
    print EF1.shape
    EF2 = a2_xi_squared * const
    ef22 = numpy.mean(EF2,1)
    EF3 = a3_xi_squared * const
    ef33 = numpy.mean(EF3,1)
    """
    #  plotting outouts

    title1 = "energy flux at %f cm away from the wavemaker" % (x[col1]+50)
    title2 = "energy flux at %f cm away from the wavemaker" % (x[col2]+50)
    title3 = "energy flux at %f cm away from the wavemaker" % (x[col3]+50)
    plt.figure(1,figsize=(15,12))
    plotting_functions.plot_3plts(EF1,EF2,EF3,t,title1,title2,title3,'time','E')
    
    title1 = "FFT of energy flux at %f cm away from the wavemaker" % (x[col1]+50)
    

    title2 = "FFT of energy flux at %f cm away from the wavemaker" % (x[col2]+50)
    title3 = "FFT of energy flux at %f cm away from the wavemaker" % (x[col3]+50)
    plt.figure(2,figsize=(15,12))
    plotting_functions.plot_3plts(abs(F1),abs(F2),abs(F3),freq,title1,title2,title3,'freq','abs(fft(EF))')
    
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
    parser.add_argument("row_s",type = int, help = " start pixel of Z ")
    parser.add_argument("row_e",type = int, help = " end pixel of Z ")
    parser.add_argument("col1",type = int, help = " pixel number of the column \
            whose time series we want to see")
    args = parser.parse_args()
    compute_energy_flux(args.a_xi_id,args.row_s,args.row_e,args.col1)
    plt.show()


if __name__ == "__main__":
    UI()



