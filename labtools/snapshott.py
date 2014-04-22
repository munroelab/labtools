__author__ = 'prajvala'

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
    path = "/Volumes/HD4/%s/%d/%s" % ( ncdir, id, ncfile )

    if not os.path.exists(path):
        print path, "not found"
        return

    # Load the nc file
    data = nc.Dataset(path, 'r')
    t = nc.variables['time']
    z = nc.variables['row']
    x = nc.variables['column']

    plt.figure()
    plt.imshow(data[t_num,:,:],extent=[x[0],x[-1],z[0],z[-1]],vmax=max_min,vmin=-max_min,aspect='auto',interpolation= 'nearest')
    plt.xlabel(' window length (cm)')
    plt.ylabel('Window Height (cm)')
    plt.title('%s field at time %d' %(var,t[t_num]) )
    plt.colorbar()
    plt.savefig(plotname + "_%dtime.pdf" % t[t_num])

