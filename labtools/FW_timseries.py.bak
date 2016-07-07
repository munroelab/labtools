__author__ = 'prajvala'
import axi_TS_col
import matplotlib.pyplot as plt
import netCDF4
import labdb


# Open database for access
db  = labdb.LabDB()

def compute_fw_row_timeseries(fw_id,column,max_min,plotname = "filteredWavesVerticalTimeSeries"):
    # Check if the database already exists
    sql = """SELECT expt_id FROM dz WHERE dz_id = (SELECT dz_id from \
            filtered_waves WHERE fw_id = %d)""" % fw_id
    rows = db.execute(sql)
    expt_id = rows[0][0]
    print "expt ID: ", expt_id

    if (len(rows) == 0):
        print "The a_xi for the expt_id is not yet computed.. "
        return


    # Call the function get_info() to get omega, kz and theta
    video_id, N_frequency, omega, kz, theta = axi_TS_col.get_info(expt_id)
    # Open the nc file and load the variables.
    path = "/Volumes/HD4/filtered_waves/%d/waves.nc" % fw_id
    nc = netCDF4.Dataset(path,'r')
    print " variables: " ,nc.variables.keys()
    left = nc.variables['left_array']
    right = nc.variables['right_array']

    print " L.shape ", left.shape
    #print "l['real'] shape" , left['real'].shape

    t = nc.variables['time'][:]
    z = nc.variables['row'][:]
    x = nc.variables['column'][:]


    plt.figure(figsize=(15,10))
    ax = plt.subplot(2,1,1)
    #plt.contourf(t,z,data.T[::-1],levels=level)
    plt.imshow(left[:,:,column]['real'].T,vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.title('FW of VidID %d.. %.2f cm away from the WG' % (video_id, 84+x[column]) )
    plt.colorbar()

    plt.subplot(2,1,2,sharex = ax, sharey = ax)
    plt.imshow(right[:,:,column]['real'].T,vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (cm)')
    plt.title('FW of VidID %d.. %.2f cm away from the WG' % (video_id, 84+x[column]) )
    plt.colorbar()

    plt.savefig(plotname + "_%dcol.pdf"% column )
    plt.show()

