__author__ = 'prajvala'
import os
import matplotlib.pyplot as plt
import netCDF4

def snapshot(var, id,t_num,
        max_min=None,
        plot_name= "snapshot.pdf"):

    """
    Given a 'variable name' of a given "id" make a plot
    """

    #  variable name : (ncdir, ncfile, ncvar)
    ncfiles = { 'video' : ('videoncfiles', 'video.nc', 'img_array'),
                'dz' : ('dz', 'dz.nc', 'dz_array'),
                'Axi' : ('vertical_displacement_amplitude', 'a_xi.nc',
                         'a_xi_array'),
              }
    ncdir, ncfile, ncvar = ncfiles[var]

    # arrays are stored in
    path = "/data/%s/%d/%s" % ( ncdir, id, ncfile )

    if not os.path.exists(path):
        print path, "not found"
        return

    # Load the nc file
    nc = netCDF4.Dataset(path, 'r')
    data = nc.variables[ncvar]
    print data.shape, nc.variables.keys()
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']
    print t.shape,z.shape,x.shape

    plt.figure()
    plt.imshow(data[t_num,:,:],extent=[x[0],x[-1],z[0],z[-1]],
               vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel(' window length (cm)')
    plt.ylabel('Window Height (cm)')
    plt.title('%s field at time %f seconds' %(var,t[t_num]))
    plt.colorbar()
    plt.savefig(plot_name + "_%ftime.pdf" % t[t_num])

def snapshot_complex(var, id,t_num,
        max_min=None,
        plot_name= "snapshot_LR.pdf"):

    """
    Given a 'variable name' of a given "id" make a plot
    """

    #  variable name : (ncdir, ncfile, ncvar)
    ncfiles = { 'fw' : ('filtered_waves', 'waves.nc', 'left_array', 'right_array'),
              }
    ncdir, ncfile, ncvar1, ncvar2 = ncfiles[var]

    # arrays are stored in
    path = "/data/%s/%d/%s" % ( ncdir, id, ncfile )

    if not os.path.exists(path):
        print path, "not found"
        return

    # Load the nc file
    nc = netCDF4.Dataset(path, 'r')
    var1 = nc.variables[ncvar1][t_num,:,:]
    var2 = nc.variables[ncvar2][t_num,:,:]
    #print var1.shape, nc.variables.keys()
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']
    #print t.shape,z.shape,x.shape

    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(var1['real'],extent=[x[0],x[-1],z[0],z[-1]],
               vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel(' window length (cm)')
    plt.ylabel('Window Height (cm)')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(var2['real'],extent=[x[0],x[-1],z[0],z[-1]],
               vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel(' window length (cm)')
    plt.ylabel('Window Height (cm)')
    plt.colorbar()

    plt.savefig(plot_name + "_%ftime.pdf" % t[t_num])
