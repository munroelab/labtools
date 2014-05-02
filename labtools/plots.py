"""
common plotting routines
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
import logging

logger = logging.getLogger(__name__)

import labdb


def plot_slice(ncvarname, nc_id, 
        slice_type,
        n,

        maxmin=None, plotName=None, saveFig=True):
    """
    """
 
    logger.debug('Plotting %s %s %n' % (slice_type, ncvarname, n))

    #  variable name : (ncdir, ncfile, ncvar)
    ncfiles = { 'video' : ('videoncfiles', 'video.nc', 'img_array'),
                'dz' : ('dz', 'dz.nc', 'dz_array'),
                'right' : ('filtered_waves', 'waves.nc', 'right_array'),
                'left' : ('filtered_waves', 'waves.nc', 'left_array'),
                'Axi' : ('vertical_displacement_amplitude', 'a_xi.nc',
                         'a_xi_array'),
              }
    ncdir, ncfile, ncvar = ncfiles[ncvarname]

    # arrays are stored in 
    path = "/data/%s/%d/%s" % ( ncdir, nc_id, ncfile )

    if not os.path.exists(path):
        raise Exception("{} not found".format(path))

    # Load the nc file
    nc = netCDF4.Dataset(path, 'r')

    # Load the variables
    arr = nc.variables.keys()
    array = nc.variables[ncvar]

    # arr = [u'row',u'column',u'time',u'a_xi_array']

    t = nc.variables[arr[2]]
    z = nc.variables[arr[0]]
    x = nc.variables[arr[1]]

    # Need window length and height 
    win_l = x[-1]
    win_h = z[-1]
  
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes()

    if slice_type == 'frame':
        field = array[n,:,:]
        extent=[x[0],x[-1],z[0],z[-1]]
        xlabel = 'x (cm)'
        ylabel = 'z (cm)'
    elif slice_type == 'vts':
        field = array[:,:,n].T
        extent=[t[0],t[-1],z[-1],z[0]]
        xlabel = 't (s)'
        ylabel = 'z (cm)'
    elif slice_type == 'hts':
        field = array[:,n,:], 
        extent=[x[0],x[-1],t[0],t[-1]],
        xlabel = 'x (cm)'
        ylabel = 't (s)'
    else:
        raise Exception('invalid slice_type')

    complex64_t = np.dtype([('real', '<f4'), ('imag', '<f4')])
    if field.dtype == complex64_t:

        datac = np.empty(field.shape, np.complex64)
        datac.real = field['real']
        datac.imag = field['imag']

        #field = np.angle(datac)
        field = datac.real
        #field = datac.imag
        #field = np.abs(datac)

    if maxmin is None:
        # estimate a good min/max value
        maxmin = abs(field).max()

    im = plt.imshow(field, extent=extent,
                  vmax=+maxmin, vmin=-maxmin,
                  interpolation="nearest",
                  aspect='auto',)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()

    if plotName is None:
        plotName = '%s_%d.pdf' % (var,id)

    if saveFig:
        logging.debug('Saving %s' % plotName)
        plt.savefig(plotName)
 
def test_plot_real():

    # open ncfiles
    nc = netCDF4.Dataset('input.nc')
    nc_ht = netCDF4.Dataset('HT.nc')
    nc_lr = netCDF4.Dataset('LR.nc')

    # grid (should be the same for all three nc files
    x = nc_ht.variables['x'][:]
    z = nc_ht.variables['z'][:]
    t = nc_ht.variables['t'][:]

    U = nc.variables['U']
    Uht = nc_ht.variables['Uht']
    L = nc_lr.variables['L']
    R = nc_lr.variables['R']

    nt, nx, nz = Uht.shape

    fig = plt.figure()

    # HT field
    datain = Uht[0, :, :]

    # HT field real
    plt.subplot(3, 1, 1)
    im_U = plt.imshow(datain['real'].T,
                            extent=(x[0], x[-1], z[0], z[-1]),
                            aspect='auto',
                            origin='lower', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Total')

    # L field
    datain = L[0, :, :]

    # L field real
    plt.subplot(3, 1, 2)
    im_L = plt.imshow(datain['real'].T,
                          extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='lower',
                          vmin =-1, vmax=1)
    plt.colorbar()
    plt.title('Left')

    # R field
    datain = R[0, :, :]

    # R field real
    plt.subplot(3, 1, 3)
    im_R = plt.imshow(datain['real'].T,
                          extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='lower',
                          vmin =-1, vmax=1)
    plt.colorbar()
    plt.title('Right')

    def updatefig(n, *args):

        # U field
        datain = Uht[n, :, :]
        im_U.set_array(datain['real'].T)

        # L field
        datain = L[n, :, :]
        im_L.set_array(datain['real'].T)

        # R field
        datain = R[n, :, :]
        im_R.set_array(datain['real'].T)

        return (im_U,  im_L,  im_R, )

    ani = animation.FuncAnimation(fig, updatefig, nt, blit=True)
    plt.show()

    nc.close()
    nc_ht.close()
    nc_lr.close()

def test_plot_amp_phase():
    # open ncfiles
    nc_ht = netCDF4.Dataset('HT.nc')
    nc_lr = netCDF4.Dataset('LR.nc')

    # grid (should be the same for all three nc files
    x = nc_ht.variables['x'][:]
    z = nc_ht.variables['z'][:]
    t = nc_ht.variables['t'][:]

    Uht = nc_ht.variables['Uht']
    L = nc_lr.variables['L']
    R = nc_lr.variables['R']

    nt, nx, nz = Uht.shape

    fig = plt.figure()

    # HT field
    datain = Uht[0, :, :]
    # form complex array
    datac = np.empty(datain.shape, np.complex64)
    datac.real = datain['real']
    datac.imag = datain['imag']

    # HT field real
    plt.subplot(3, 2, 1)
    im_ht_real = plt.imshow(datac.real.T,
                            extent=(x[0], x[-1], z[0], z[-1]),
                            aspect='auto',
                            origin='lower', vmin=-1, vmax=1)
    plt.colorbar()
    # HT field imag
    plt.subplot(3, 2, 2)
    im_ht_imag = plt.imshow(datac.imag.T,
                            extent=(x[0], x[-1], z[0], z[-1]),
                            aspect='auto', origin='lower', vmin=-1, vmax=1,)
    plt.colorbar()

    # L field
    datain = L[0, :, :]
    datac = np.empty(datain.shape, np.complex64)
    datac.real = datain['real']
    datac.imag = datain['imag']

    # L field real
    plt.subplot(3, 2, 3)
    im_L_amp = plt.imshow(abs(datac).T,
                          extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='lower',
                          vmin =0, vmax=1)
    plt.colorbar()
    # L field imag
    plt.subplot(3, 2, 4)
    im_L_phase = plt.imshow(np.angle(datac).T,
                          extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='lower')
    plt.colorbar()

    # R field
    datain = R[0, :, :]
    datac = np.empty(datain.shape, np.complex64)
    datac.real = datain['real']
    datac.imag = datain['imag']

    # R field real
    plt.subplot(3, 2, 5)
    im_R_amp = plt.imshow(np.abs(datac).T,
                          extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='lower',
                          vmin =0, vmax=1)
    plt.colorbar()
    # R field imag
    plt.subplot(3, 2, 6)
    im_R_phase = plt.imshow(np.angle(datac).T,
                          extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='lower',
                          )
    plt.colorbar()

    def updatefig(n, *args):
        """

        @param n:
        @param args:
        @return:
        """

        # HT field
        datain = Uht[n, :, :]
        datac = np.empty(datain.shape, np.complex64)
        datac.real = datain['real']
        datac.imag = datain['imag']

        im_ht_real.set_array(datac.real.T)
        im_ht_imag.set_array(datac.imag.T)

        # L field
        datain = L[n, :, :]
        datac = np.empty(datain.shape, np.complex64)
        datac.real = datain['real']
        datac.imag = datain['imag']

        im_L_amp.set_array(np.abs(datac).T)
        im_L_phase.set_array(np.angle(datac).T)

        # R field
        datain = R[n, :, :]
        datac = np.empty(datain.shape, np.complex64)
        datac.real = datain['real']
        datac.imag = datain['imag']

        im_R_amp.set_array(np.abs(datac).T)
        im_R_phase.set_array(np.angle(datac).T)

        return (im_ht_real, im_ht_imag,
                im_L_amp, im_L_phase,
                im_R_amp, im_R_phase,)

    ani = animation.FuncAnimation(fig, updatefig, nt, blit=True)
    plt.show()

    nc_ht.close()
    nc_lr.close()


