import numpy
import pylab 
import matplotlib.pyplot as plt
import netCDF4
import argparse
import labdb
import os

# Open database for access
db  = labdb.LabDB()


def get_MSFD(dz_id):
    #get the mintol,sigma, and filter_size of the deltaN2
    sql = """ SELECT mintol,sigma,filter_size,diff_frames,video_id FROM dz WHERE dz_id= %d""" % dz_id
    rows = db.execute(sql)
    print "rows:",rows
    mintol= rows[0][0]
    sigma = rows[0][1]
    filter_size = rows[0][2]
    diff_frames = rows[0][3]
    video_id = rows[0][4]
    print " mintol:" ,mintol, "sigma:" ,sigma, "filter_size:",filter_size,\
            "diff_frames : " ,diff_frames,"video_id: ",video_id
    return mintol,sigma,filter_size,diff_frames,video_id


def compute_dz_timeseries(dz_id,column,max_min,plot_name = 'dz_vts.pdf'):
    # Check if the database already exists
    sql = """ SELECT expt_id  FROM dz WHERE dz_id = %d""" % dz_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print "expt ID: ", expt_id

    if (len(rows) == 0):
        print "The dz for the expt_id is not yet computed.. "
        return


    # Call the function get_info() to get omega, kz and theta
    video_id = get_info(expt_id)
    
    # create a timeseries of a video column pixel
    

    
    #video_id, N_frequency, omega, kz, theta = get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD4/dz/%d/dz.nc" % dz_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    dz_arr = nc.variables['dz_array']
    print " dz_array.shape ", dz_arr.shape
    
    t = nc.variables['time'][:] 
    
    z = nc.variables['row']
    x = nc.variables['column']
    mintol,sigma,filter_size,diff_frames,v_id=get_MSFD(dz_id)
    dat = dz_arr[:,:,column]
    data=dat.T
    print "data.T.shape", data.shape
    plt.figure(figsize=(20,11))
    #plt.contourf(t,z,data.T[::-1],levels=level)
    plt.imshow(data[:,:],extent=[t[0],t[-1],z[-1],z[0]],vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.title('Video ID: %d, Diff_frames: %d \n dz of a column of pixels %.2f cm away\
            from the wave generator' % (video_id,diff_frames, x[column]+110.0) )
    #im1=dz_arr[203,:,:]
    #plt.figure(2)
    #plt.imshow(im1,vmax=0.005,vmin=-0.005)
    #plt.figure(3)
    #plt.imshow(im2,vmax=0.0003,vmin=-0.0003)
    #plt.figure(4)
    #plt.imshow(im3,vmax=0.0003,vmin=-0.0003)
    #plt.figure(5)
    #plt.imshow(im4,vmax=0.0003,vmin=-0.0003)

    plt.colorbar()
    plt.savefig(plot_name + 'col%d.pdf' %column)

def get_info(expt_id):
    
    # Get the video_id for the corresponding expt.id
    sql = """ SELECT video_id FROM video_experiments WHERE expt_id = %d """ % expt_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    print "VIDEO ID = " ,video_id
    
    # Get the Buoyancy frequency
    #sql = """ SELECT N_frequency FROM stratification WHERE strat_id =\
    #        (SELECT strat_id FROM stratification_experiments WHERE expt_id = %d )""" % expt_id
    #rows = db.execute(sql)
    #N_frequency = rows[0][0]
    #print "Buoyancy Frequency: ", N_frequency

    # Get the frequency,kz and calculate theta
    #sql = """ SELECT kz , frequency_measured FROM wavemaker WHERE wavemaker_id \
    #        = (SELECT wavemaker_id FROM wavemaker_experiments WHERE expt_id = %d) """% expt_id
    #rows = db.execute(sql)
    #kz = rows[0][0]
    #f = rows[0][1]
    #print "kz : ",kz, "frequency:" ,f
    #omega = 2 * numpy.pi * f

    # wkt w = N * cos(theta) ,arccos returns in radians 
    #theta = numpy.arccos(omega/N_frequency) 
    #print "omega: ", omega, "\t theta: " ,theta*180/numpy.pi

    #return video_id, N_frequency, omega, kz, theta*180.0/numpy.pi
    return video_id

def UI():
    """ take the experiment id from the user and calculate the energy flux from
    the vertical displacement amplitude that has already been calculated and
    plot it """
    parser = argparse.ArgumentParser()
    parser.add_argument("dz_id",type=int,help="Enter the dz ID")
    parser.add_argument("column",type = int,help = "Enter the pixel of the column (0-1291) whose timeseries you want to see")
    parser.add_argument("max_min",type= float, help = "Enter the values to be used as vmax and vmin")
    
    args = parser.parse_args()
    compute_dz_timeseries(args.dz_id, args.column, args.max_min)
    plt.show()

if __name__ == "__main__":
    UI()



