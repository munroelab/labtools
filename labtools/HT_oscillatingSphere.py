
import matplotlib
matplotlib.use('TkAgg')
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4
import pylab
import copy
import labdb
import time
import progressbar
from matplotlib import animation
from chunk_shape_3D import chunk_shape_3D


def HT_OC(dz_id):

    filename = '/Volumes/HD4/dz/%d/dz.nc' %dz_id

    # STEP 1: FFT in t ######
    nc = netCDF4.Dataset(filename)

    dz_arr = nc.variables['dz_array']
    nt, nx, nz = dz_arr.shape

    # create ncfile to store Hilbert transform of U
    nc_ht = netCDF4.Dataset('HT.nc', 'w', format='NETCDF4')

    # since at HT of dz is complex, we need complex data types
    complex64 = np.dtype([('real', np.float32), ('imag', np.float32)])
    complex64_t = nc_ht.createCompoundType(complex64, 'complex64')

    # copy grid from U ncfile
    x_dim = nc_ht.createDimension('x', nx)
    z_dim = nc_ht.createDimension('z', nz)
    t_dim = nc_ht.createDimension('t', nt)

    x = nc_ht.createVariable('x', np.float32, ('x'))
    z = nc_ht.createVariable('z', np.float32, ('z'))
    t = nc_ht.createVariable('t', np.float32, ('t'))

    x[:] = nc.variables['x'][:]
    z[:] = nc.variables['z'][:]
    t[:] = nc.variables['t'][:]

    valSize = complex64.itemsize
    chunksizes = chunk_shape_3D( ( nt, nx, nz),
                                  valSize=valSize )
    dzht = nc_ht.createVariable('dzht', complex64_t, ('t', 'x', 'z'),
                               chunksizes=chunksizes)

    print "Temporal filtering"
    print
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=nx).start()
    # loop over all x
    for i in range(nx):
        pbar.update(i)

        datain = dz_arr[:, i, :]

        # take FFT in time
        dz_spectrum = np.fft.fft(datain, axis=0)

        # only keep positive frequencies
        #   could extend this to a band pass filter

        # explicitly set all non-positive frequencies to zero
        U_spectrum[nt/2:, :] = 0

        # take inverse FFT and multiply by 2
        datac = 2 * np.fft.ifft(U_spectrum, axis=0)

        # convert to a compound datatype
        dataout = np.empty(datain.shape, complex64)
        dataout['real'] = datac.real
        dataout['imag'] = datac.imag

        # store in netcdf4 file
        dzht[:, i, :] = dataout
    pbar.finish()

    nc_ht.close()
    nc.close()

    ## STEP 2
    ### extract out only rightward and leftward propagating portions of Uht
    nc_ht = netCDF4.Dataset('HT.nc')

    Uht = nc_ht.variables['Uht']
    nt, nx, nz = Uht.shape

    # create ncfile to store Hilbert transform of U
    nc_lr = netCDF4.Dataset('LR.nc', 'w', format='NETCDF4')

    # since at HT of U is complex, we need complex data types
    complex64 = np.dtype([('real', np.float32), ('imag', np.float32)])
    complex64_t = nc_lr.createCompoundType(complex64, 'complex64')

    # copy grid from Uht ncfile
    t_dim = nc_lr.createDimension('t', nt)
    x_dim = nc_lr.createDimension('x', nx)
    z_dim = nc_lr.createDimension('z', nz)

    t = nc_lr.createVariable('t', np.float32, ('t'))
    x = nc_lr.createVariable('x', np.float32, ('x'))
    z = nc_lr.createVariable('z', np.float32, ('z'))

    t[:] = nc_ht.variables['t'][:]
    x[:] = nc_ht.variables['x'][:]
    z[:] = nc_ht.variables['z'][:]


    valSize = complex64.itemsize
    chunksizes = chunk_shape_3D( ( nt, nx, nz),
                                  valSize=valSize )
    L = nc_lr.createVariable('L', complex64_t, ('t', 'x', 'z'),
                             chunksizes=chunksizes)
    R = nc_lr.createVariable('R', complex64_t, ('t', 'x', 'z'),
                             chunksizes=chunksizes)

    print "Spatial filtering"
    widgets = [progressbar.Percentage(), ' '
        , progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=nt).start()
    # loop over all t
    for n in range(nt):
        pbar.update(n)

        # grab frame
        datain = Uht[n, :, :]

        # form complex array
        datac = np.empty(datain.shape, np.complex64)
        datac.real = datain['real']
        datac.imag = datain['imag']

        # fft
        datac_spectrum = np.fft.fft2(datac, axes=(0,1))

        # make a copy
        datac_R_spectrum = datac_spectrum.copy()
        datac_L_spectrum = datac_spectrum # note: no copy here

        # only include right ward propagating (kx > 0)
        datac_R_spectrum[:nx/2, :] = 0

        # only include left ward propagating (kx < 0)
        datac_L_spectrum[nx/2:, :] = 0

        # inverse FFT
        datac_R = np.fft.ifft2(datac_R_spectrum, axes=(0,1))
        datac_L = np.fft.ifft2(datac_L_spectrum, axes=(0,1))

        # convert to compound datatype and save
        dataout1 = np.empty(datain.shape, complex64)
        dataout1['real'] = datac_R.real
        dataout1['imag'] = datac_R.imag
        R[n, :, :] = dataout1

        dataout = np.empty(datain.shape, complex64)
        dataout['real'] = datac_L.real
        dataout['imag'] = datac_L.imag
        L[n, :, :] = dataout

    pbar.finish()

    nc_lr.close()
    nc_ht.close()

    # need the idea of a 'temp' nc file

