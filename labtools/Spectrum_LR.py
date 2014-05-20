"""
Routines for computing Fourier transforms of f(x,z,t) fields

Computes the Hilbert transform and saves the filtered data into right.nc and
left.nc files

"""
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
import tempfile
import time
import progressbar
import multiprocessing
from matplotlib import animation
from chunk_shape_3D import chunk_shape_3D
from labtools import Energy_flux
from labtools import  plotting_functions
import logging
logger = logging.getLogger(__name__)


def create_nc_file(a_xi_id, fw_id=None):
    """ 
        Given an Axi_id, create the waves.nc file that will store the
        Hilbert transform result.

        The fw_id will be chosen as the next available id the supplied
        fw_id is None.
    
    
    Need to compute HT for the first time.
        Need to create the path for the right and left wave fields and create the empty nc file
    """

    db = labdb.LabDB()
    # Get experiment ID

    sql = """ SELECT dz.video_id, dz.expt_id FROM dz INNER JOIN \
            vertical_displacement_amplitude ON (dz.dz_id = \
            vertical_displacement_amplitude.dz_id AND \
            vertical_displacement_amplitude.a_xi_id = %d) """ % a_xi_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    expt_id = rows[0][1]
    print " experiment ID : ", expt_id, "Video ID :", video_id

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0] * 1.0

    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0] * 1.0

    # Create the directory in which to store the nc file
    if fw_id is None:
        sql = """INSERT INTO filtered_waves (a_xi_id,video_id)\
                VALUES (%d,%d)""" % (a_xi_id, video_id)
        print sql
        db.execute(sql)
        sql = """SELECT LAST_INSERT_ID()"""
        rows = db.execute(sql)
        fw_id = rows[0][0]
    else:
        sql = "SELECT a_xi_id FROM filtered_waves WHERE fw_id = %s" % fw_id
        previous_a_xi_id, = db.execute_one(sql)
        if previous_a_xi_id != a_xi_id:
            print "fw_id, a_xi_id, mismatch!"
            return None

    fw_path = "/data/filtered_waves/%d" % fw_id
    if not os.path.exists(fw_path):
        os.mkdir(fw_path)
    fw_filename = os.path.join(fw_path, "waves.nc")

    # open the axi nc file and set the nrow and Ncol for setting the limits for
    # the new created file

    axincfile = netCDF4.Dataset(
        '/data/vertical_displacement_amplitude/%d/a_xi.nc' % a_xi_id,
        'r')
    Nrow = axincfile.variables['row'].size
    Ncol = axincfile.variables['column'].size
    Ntime = axincfile.variables['time'].size

    # Declare the nc file for the first time 
    # lines copied for reference
    # create ncfile to store Hilbert transform of U
    #nc_lr = netCDF4.Dataset('LR.nc', 'w', format='NETCDF4')

    # since at HT of U is complex, we need complex data types
    #complex64 = np.dtype([('real', np.float32), ('imag', np.float32)])
    #complex64_t = nc_lr.createCompoundType(complex64, 'complex64')
#    valSize = complex64.itemsize
    #    chunksizes = chunk_shape_3D( ( nt, nx, nz),
    #                                  valSize=valSize )
    #    L = nc_lr.createVariable('L', complex64_t, ('t', 'x', 'z'),
    #                             chunksizes=chunksizes)
    #    R = nc_lr.createVariable('R', complex64_t, ('t', 'x', 'z'),
    #                             chunksizes=chunksizes)


    nc = netCDF4.Dataset(fw_filename, 'w', format='NETCDF4')
    
    # the final processed data is a compound data type consisting of the real and imaginary
    # parts
    complex64 = np.dtype([('real',np.float32), ('imag', np.float32)])
    complex64_t = nc.createCompoundType(complex64, 'complex64')
    
    # chunk the data intelligently
    valSize = complex64.itemsize
    chunksizes = chunk_shape_3D( ( Ntime, Nrow, Ncol), valSize=valSize )
    
    # Create Dimension
    row_dim = nc.createDimension('row', Nrow)
    col_dim = nc.createDimension('column', Ncol)
    t_dim = nc.createDimension('time', Ntime)

    #the dimensions are also variables
    ROW = nc.createVariable('row', np.float32, ('row'))
    COLUMN = nc.createVariable('column', np.float32, ('column'))
    TIME = nc.createVariable('time', np.float32, ('time'))

    # declare the 3D data variable 
    raw = nc.createVariable('raw_array', complex64_t, ('time', 'row',
        'column'),chunksizes = chunksizes)
    left_w = nc.createVariable('left_array', complex64_t, ('time', 'row',
        'column'),chunksizes = chunksizes)
    right_w = nc.createVariable('right_array', complex64_t, ('time', 'row',
        'column'),chunksizes = chunksizes)

    print nc.dimensions.keys()
    print "L", left_w.shape, left_w.dtype
    print "R", right_w.shape, right_w.dtype

    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    C = np.arange(0, win_l, win_l / Ncol, dtype=float)
    R = np.arange(0,win_h, win_h/Nrow,dtype=float)

    print "spectrumLR c.shape", C.shape
    print "column.shape", COLUMN.shape
    COLUMN[:] = C
    ROW[:]= R

    db.commit()
    nc.close()
    return fw_filename, fw_id

def create_nc_file_dzHT(dz_id,rowS,rowE,colS,colE, fw_id=None,startF=None,period=None):
    """
    Given an Axi_id, create the waves.nc file that will store the
    Hilbert transform result.

    The fw_id will be chosen as the next available id the supplied
        fw_id is None.


    Need to compute HT for the first time.
    Need to create the path for the right and left wave fields and create the empty nc file
    """

    db = labdb.LabDB()
    # Get experiment ID

    sql = """ SELECT video_id,expt_id FROM dz WHERE dz_id = %d """ % dz_id
    rows = db.execute(sql)
    video_id = rows[0][0]
    expt_id = rows[0][1]
    print " experiment ID : ", expt_id, "Video ID :", video_id

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0] * 1.0

    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0] * 1.0

    # Create the directory in which to store the nc file
    if fw_id is None:
        if startF is None or period is None:
            startF_ = "NULL"
            period_ = "NULL"
        else:
            startF_ = "%d" % startF
            period_ = "%d" % period

        sql = """INSERT INTO filtered_waves (dz_id,video_id,startF,period)\
                VALUES (%d,%d,%s,%s)""" % (dz_id, video_id,startF_,period_)
        print sql
        db.execute(sql)
        sql = """SELECT LAST_INSERT_ID()"""
        rows = db.execute(sql)
        fw_id = rows[0][0]
    else:
        sql = "SELECT dz_id FROM filtered_waves WHERE fw_id = %s" % fw_id
        previous_dz_id, = db.execute_one(sql)
        if previous_dz_id != dz_id:
            print "fw_id, dz_id, mismatch!"
            return None

    fw_path = "/data/filtered_waves/%d" % fw_id
    if not os.path.exists(fw_path):
        os.mkdir(fw_path)
    fw_filename = os.path.join(fw_path, "waves.nc")

    # open the axi nc file and set the nrow and Ncol for setting the limits for
    # the new created file

    dzncfile = netCDF4.Dataset(
        '/data/dz/%d/dz.nc' % dz_id,
        'r')

    Nrow_specs = dzncfile.variables['row'][rowS:rowE]
    Ncol_specs = dzncfile.variables['column'][colS:colE]
    dz_time = dzncfile.variables['time'][:]
    Nrow = Nrow_specs.size
    Ncol = Ncol_specs.size
    # if startF and period are given then you need to get to set Ntime differently
    if (startF is None or period is None):
        Ntime = dz_time.size
    else:
        sql = """ SELECT wavemaker.frequency_measured FROM wavemaker_experiments INNER JOIN wavemaker \
        ON wavemaker_experiments.wavemaker_id = wavemaker.wavemaker_id AND \
        wavemaker_experiments.expt_id = %d""" % expt_id
        rows = db.execute(sql)
        f = rows[0][0]
        Ntime = round(period*6.2/f)

    print "NROW,NCOL,NTIME::" ,Nrow,Ncol,Ntime

    nc = netCDF4.Dataset(fw_filename, 'w', format='NETCDF4')

    # the final processed data is a compound data type consisting of the real and imaginary
    # parts
    complex64 = np.dtype([('real',np.float32), ('imag', np.float32)])
    complex64_t = nc.createCompoundType(complex64, 'complex64')

    # chunk the data intelligently
    valSize = complex64.itemsize
    chunksizes = chunk_shape_3D( ( Ntime, Nrow, Ncol), valSize=valSize )


    # Create Dimension
    row_dim = nc.createDimension('row', Nrow)
    col_dim = nc.createDimension('column', Ncol)
    t_dim = nc.createDimension('time', Ntime)

    #the dimensions are also variables
    ROW = nc.createVariable('row', np.float32, ('row'))
    COLUMN = nc.createVariable('column', np.float32, ('column'))
    TIME = nc.createVariable('time', np.float32, ('time'))

    # declare the 3D data variable
    raw = nc.createVariable('raw_array', complex64_t, ('time', 'row',
        'column'),chunksizes = chunksizes)
    left_w = nc.createVariable('left_array', complex64_t, ('time', 'row',
        'column'),chunksizes = chunksizes)
    right_w = nc.createVariable('right_array', complex64_t, ('time', 'row',
        'column'),chunksizes = chunksizes)

    print nc.dimensions.keys()
    print "raw", raw.shape, raw.dtype
    print "L", left_w.shape, left_w.dtype
    print "R", right_w.shape, right_w.dtype

    # the length and height dimensions are variables containing the length and
    # height of each pixel in cm
    #C = np.arange(0, win_l,win_l/Ncol,dtype=float)
    #R = np.arange(0,win_h,win_h/Nrow,dtype=float)

    print "spectrumLR c.shape", Ncol_specs.shape
    print "column.shape", Nrow_specs.shape
    COLUMN[:] = Ncol_specs[:]
    ROW[:]= Nrow_specs[:]
    if (startF is None or period is None):
        TIME[:]= dz_time[:]
    else:
        TIME[:] = np.mgrid[0:(period*1.0/f):Ntime * 1.0j]

    db.commit()
    nc.close()
    return fw_filename, fw_id


def plotFilteredLR(fw_id,max_min,plotcolumn,
                   plotName=None,rowS=0,rowE=964
):
    """
    create timeseries of left and right waves and store the plot as a pdf
    """

    db = labdb.LabDB()

    #check if the file already exists
    fw_filename = "/data/filtered_waves/%d/waves.nc" % fw_id

    if not os.path.exists(fw_filename):
        print "waves.nc not found"
        return


    # Open the nc file for writing data
    nc = netCDF4.Dataset(fw_filename, 'r')
    left = nc.variables['left_array'][:, rowS:rowE, plotcolumn]
    right = nc.variables['right_array'][:, rowS:rowE, plotcolumn]
    z = nc.variables['row'][rowS:rowE]
    t = nc.variables['time']

    if plotName is None:
        #generate a unique path for storing plots
        path = "figures/axi_LR/fw_id%d" % fw_id
        if not os.path.exists(path):
            os.mkdir(path)
        fname = os.path.join(path, "plot.pdf")
    else:
        fname = plotName


    plt.figure()
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(left['real'].T,extent=[t[0],t[-1],z[-1],z[0]],
               aspect='auto', vmin=-max_min, vmax=max_min,
               interpolation='nearest')
    plt.ylabel('Depth (cm)')
    #plt.xlabel('Time (s)')
    plt.figtext(.13,.91,'a )')
    #plt.title('Left')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(right['real'].T, extent=[t[0],t[-1],z[-1],z[0]],
               aspect='auto', vmin=-max_min,
               vmax=max_min,interpolation='nearest')
    #plt.title('Right')
    plt.colorbar()
    plt.ylabel('Depth (cm)')
    plt.xlabel('Time (s)')
    plt.figtext(.13,.475,'b )')
    nc.close()

    plt.savefig(fname)


def just_plot(fw_path, a_xi_id, maxmin, plotcolumn):
    # Open the nc file for reading data
    nc = netCDF4.Dataset(fw_path, 'r')
    raw = nc.variables['raw_array']
    left = nc.variables['left_array']
    right = nc.variables['right_array']
    ft = nc.variables['time']
    fz = nc.variables['row']
    fx = nc.variables['column']
    # print information about dz dataset
    print "variables  of the nc file :", nc.variables.keys()
    print "left_w shape : ", left.shape
    print "right_w shape : ", right.shape
    print "t  shape : ", ft.shape

    #generate a unique path for storing plots
    import datetime as dt

    now = dt.datetime.now()
    print now
    path = "/Users/prajvala/Desktop/figures/axi_LR/axi%d_%d-%d-%d_%d-%d-%d/" % \
           (a_xi_id, now.day, now.month, now.year, now.hour, now.minute, now.second)
    os.mkdir(path)
    fname1 = os.path.join(path, "plot1.pdf")
    fname2 = os.path.join(path, "plot2.pdf")
    names = [fname1, fname2]

    print names

    #call the plotting function
    plot_axes = np.array([ft, fz, fx])

    print plot_axes.shape
    plot_plots(raw[:, :, plotcolumn], right[:, :, plotcolumn], left[:, :, plotcolumn], plotcolumn, plot_axes, maxmin,
               names)

    nc.close()
    plt.show()


def plot_plots(var1, var2, var3, col, plotaxes, maxmin, name):
    t = plotaxes[0]
    z = plotaxes[1]
    x = plotaxes[2]
    print x.shape
    print var1.shape
    print x[col]

    print "x", x[0], x[-1]
    print "z", z[0], z[-1]
    print "t", t[0], t[-1]

    fig = plt.figure(1, figsize=(17, 13))
    fig.patch.set_facecolor('white')

    a, b = 0, -1
    ax1 = plt.subplot(3, 1, 1)
    plt.imshow(var1[a:b, :], extent=[t[0], t[-1], z[b], z[a]], vmin=-maxmin, vmax=maxmin, \
               aspect='auto', interpolation='nearest')
    plt.title('Timeseries of Vertical Displacement Amplitude (cm) \n \
            Data (before applying HT ) %dcm from the wavegenerator' % (90 + x[col]))
    plt.ylabel('depth (cm)')
    plt.colorbar()

    plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    plt.imshow(var2[a:b, :], extent=[t[0], t[-1], z[b], z[a]], vmin=-maxmin, vmax=maxmin, \
               aspect='auto', interpolation='nearest')
    plt.title('(Data after applying HT) Rightward ')
    plt.ylabel('depth (cm)')
    plt.colorbar()
    pylab.subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    plt.imshow(var3[a:b, :], extent=[t[0], t[-1], z[b], z[a]], vmin=-maxmin, vmax=maxmin, \
               aspect='auto', interpolation='nearest')
    plt.title('Leftward')
    plt.ylabel('depth (cm)')
    plt.xlabel('time (s)')
    plt.colorbar()
    plt.savefig(name[1], facecolor='w', edgecolor='b', format='pdf', transparent=False)
    plt.figure(2, figsize=(15, 10))
    plt.imshow(var1[a:b, :], extent=[t[0], t[-1], z[b], z[a]], vmin=-maxmin, vmax=maxmin, \
               aspect='auto', interpolation='nearest')
    plt.title('Timeseries of Vertical Displacement Amplitude (cm) \n \
        Data (before applying HT ) %dcm from the wavegenerator' % (90 + x[col]))
    plt.ylabel('depth (cm)')
    plt.xlabel('time (s)')
    plt.colorbar()
    return

def moving_average(arr, window):
    n_rows = arr.shape[1]
    avg = []
    for nr in range(n_rows):
        print nr, "out of ", n_rows
        temp = arr[:, nr]
        sum = []
        for i in range(window):
            sum.append(np.pad(temp[i:], (0, i), 'constant', constant_values=(0,)))
        temp1 = np.sum(sum, 0) / window
        avg.append(temp1[:-(window - 1)])
    avg = np.float32(avg)
    avg = np.array(avg)
    print "average shape: ", avg.shape
    return avg

def xzt_fft(a_xi_id, row_z, col_start, col_end, max_min):
    """
    Given the three-dimensional array f(x,z,t) gridded onto x, z, t
    compute the Fourier transform F.

    Returns F, X, Z, T where F is the Fourier transform and 
    X, Z, T are the frequency axes
    """

    # get the path to the nc file
    # Open &  Load the nc file
    path = "/data/vertical_displacement_amplitude/%d" % a_xi_id
    filename = path + "/a_xi.nc"
    nc = netCDF4.Dataset(filename)

    #load the variables
    a_xi_arr = nc.variables['a_xi_array']
    t = nc.variables['time']
    x = nc.variables['column'][col_start:col_end]
    z = nc.variables['row']

    # Select region of interest and convert all the variables into float16 to
    # save memory
    a_xi_arr = a_xi_arr[:, row_z, col_start:col_end]
    #a_xi_arr = a_xi_arr - a_xi_arr.mean()
    print "mean.shape :: ", a_xi_arr.mean().shape

    a_xi_arr = np.float16(a_xi_arr)
    x = np.float16(x)
    t = np.float16(t)
    #z = np.float16(z)


    print "Vertical Displacement Amplitude array shape: ", a_xi_arr.shape
    print "T shape: ", t.shape
    print "X shape: ", x.shape
    print "Z shape: ", z.shape

    # determine lengths of x, z, t
    #nz = len(z)
    nx = len(x)
    nt = len(t)
    print "length of X, T:  ", nx, nt

    # assume data is sampled evenly
    #dz = z[1] - z[0]
    dx = np.mean(np.diff(x))
    dt = np.mean(np.diff(t))
    print "dx,dt :: ", dx, dt

    # perform FFT alone all three dimensions
    # Normalize and shift so that zero frequency is at the center
    a_xi_fft = np.fft.fft2(a_xi_arr)
    F = np.fft.fftshift(a_xi_fft)

    F_invs = np.fft.ifftshift(F)
    a_xi_rec = np.fft.ifft2(F_invs)

    print "fft of deltaN2 _array:: type and size::", a_xi_fft.dtype, a_xi_fft.size
    print "shape:", a_xi_fft.shape
    #print"F: ", F[10,200]
    #print "abs F:", abs(F[10,200])
    #print "F.real",F[10,200].real
    #print "F.imag", F[10,200].imag

    # determine frequency axes
    #kz = np.fft.fftfreq(nz, dz)
    #kz = 2*np.pi*np.fft.fftshift(kz)
    kx = np.fft.fftfreq(nx, dx)
    kx = np.fft.fftshift(kx)

    omega = np.fft.fftfreq(nt, dt)
    omega = np.fft.fftshift(omega)

    print "kx shape: ", kx.shape
    #print "kz shape: ", kz.shape
    print "omega shape: ", omega.shape
    print "omega", omega
    # create a 2D mesh grid so that omega,kx and fft have the same dimensions
    K, O = np.meshgrid(kx, omega[::-1])
    print "KX.shape", K.shape
    print "OMEGA.shape", O.shape

    #calling the filter to separate out the waves travelling right from those
    #travelling left
    F_R, F_L = filter_LR(K, O, F)
    print "shape of F_R and F_L", F_R.shape, "and ", F_L.shape

    # inverse shift and ifft the fft-ed data to get back the rightward
    # travelling and leftward travelling deltaN2
    F_Rinvs = np.fft.ifftshift(F_R)
    a_xi_R = np.fft.ifft2(F_Rinvs)
    print "a_xi_R.shape", a_xi_R.shape
    F_Linvs = np.fft.ifftshift(F_L)
    a_xi_L = np.fft.ifft2(F_Linvs)
    print "a_xi_L.shape", a_xi_L.shape

    plt.figure(8)
    pylab.subplot(2, 1, 1)
    plt.imshow(a_xi_arr, vmin=-max_min, vmax=max_min, \
               aspect='auto', interpolation='nearest')
    plt.xlabel('raw data at depth at %d cm' % z[row_z])
    plt.colorbar()
    pylab.subplot(2, 1, 2)
    plt.imshow(a_xi_rec.real, extent=[x[0], x[-1], t[-1], t[0]], vmin=-max_min, vmax=max_min, \
               aspect='auto', interpolation='nearest')
    plt.xlabel('reconstructed data (directly from raw data) for sanity check at depth %d cm' % z[row_z])
    plt.colorbar()

    plt.figure(1)
    pylab.subplot(2, 1, 1)
    plt.imshow(a_xi_arr, extent=[x[0], x[-1], t[-1], t[0]], vmin=-max_min, vmax=max_min, \
               aspect='auto', interpolation='nearest')
    plt.xlabel('raw data at depth at %d cm' % z[row_z])
    plt.colorbar()
    pylab.subplot(2, 1, 2)
    plt.imshow(((a_xi_R + a_xi_L).real), extent=[x[0], x[-1], t[-1], t[0]], vmin=-max_min, vmax=max_min, \
               aspect='auto', interpolation='nearest')
    plt.xlabel('reconstructed data (a_xi_R+a_xi_L.real)  at depth %d cm' % z[row_z])
    plt.colorbar()
    plt.figure(2)
    pylab.subplot(2, 1, 1)
    plt.imshow((a_xi_L.real), extent=[x[0], x[-1], t[-1], t[0]], vmin=-max_min, vmax=max_min, \
               aspect='auto', interpolation='nearest')
    plt.xlabel('left (a_xi_L.real) ')
    plt.colorbar()
    pylab.subplot(2, 1, 2)
    plt.imshow((a_xi_R.real), extent=[x[0], x[-1], t[-1], t[0]], vmin=-max_min, vmax=max_min, \
               aspect='auto', interpolation='nearest')
    plt.xlabel('right (a_xi_R.real)')
    plt.colorbar()

    plot_data(kx, omega, F, F_R, F_L, K, O)
    nc.close()
    return

def moving_average_2Darray(arr, N):
    """
        Computes the moving average of arr over a window length 'window'
        along the first dimension of arr
    """
    conv = np.apply_along_axis(lambda m: np.convolve(m,np.ones((N,))/N,'valid'), axis=1,arr=arr)
    PadLeft = N // 2
    PadRight = N - 1 - PadLeft
    arr_ma = np.hstack( [ np.zeros((conv.shape[0],PadLeft)), conv, np.zeros((conv.shape[0],PadRight)) ] )

    print "array size in moving average along one axis function" ,arr.shape, conv.shape,arr_ma.shape
    return arr_ma


def filtered_waves_VTS(fw_id,
                      plotcol,
                      max_min,
                      plotName = None,
                      ):

    db = labdb.LabDB()
    # get the dz_id of the raw dz data
    sql = """ SELECT dz_id FROM filtered_waves WHERE fw_id = %d """ % fw_id
    rows = db.execute(sql)
    dz_id = rows[0][0]
    print "dz_id :: " ,dz_id

    # load the dz data of the filtered waves to view the input data before hilbert transforming them
    dz_filename = "/data/filtered_waves/%d/waves.nc" % fw_id
    dz_nc = netCDF4.Dataset(dz_filename,'r')
    raw = dz_nc.variables['raw_array'][:,:,plotcol]
    t=dz_nc.variables['time']
    print "dz time:", t.shape
    # load the filtered waves data corresponding to the dz_id
    fw_filename = "/data/filtered_waves/%d/waves.nc" % fw_id
    nc = netCDF4.Dataset(fw_filename, 'r')
    left = nc.variables['left_array'][:,:,plotcol]
    right = nc.variables['right_array'][:,:,plotcol]
    t = nc.variables['time'][:]
    print "fw_time: ", t.shape
    x = nc.variables['column'][:]
    z = nc.variables['row'][:]
    print "raw array :", raw.shape, "left.array", left.shape,"right array", right.shape
    # constants needed to convert dz into energy flux
    # call get_info function from Energy_flux program :: to get info
    sql = """ SELECT expt_id  FROM dz WHERE dz_id = %d """ % dz_id
    rows = db.execute(sql)
    expt_id = rows[0][0]

    vid_id, N, omega,kz,theta = Energy_flux.get_info(expt_id)
    dN2t_to_afa = (1/N**3)/((omega/N)**2 * kz * (1.0-(omega/N)**2)**-0.5)
    print "dN2t_to_afa::", dN2t_to_afa
    rho0 = 0.998
    kx = (omega * kz)/(N**2 - omega**2)**0.5
    const = (0.5/kx) * rho0 * (N**3) * np.cos(theta*np.pi/180) * (np.sin(theta*np.pi/180))**2
    dn2t_to_ef = rho0 * (1-(omega/N)**2)**(3/2) / (kz**3 * N**3 * (omega/N)**2)
    print "dn2t_to_ef ::",dn2t_to_ef
    print "const:: ",const
    #get the window size for computing the moving average
    dt = np.mean(np.diff(t[:]))
    window = 2*np.pi/(omega*dt)
    window = np.int16(window)
    print "window:", window


    # raw_ma= moving_average_2Darray(raw['real'].T,window) * const * dN2t_to_afa
    # raw_EF = raw_ma**2
    # raw_VAEF = np.mean(raw_EF,0)
    # left_ma = moving_average_2Darray(left['real'].T,window) * const * dN2t_to_afa
    # left_EF = left_ma**2
    # left_VAEF = np.mean(left_EF,0)
    # right_ma = moving_average_2Darray(right['real'].T,window) * const * dN2t_to_afa
    # right_EF = right_ma**2
    # right_VAEF = np.mean(right_EF,0)

    raw_EF = raw['real']**2  * const * dN2t_to_afa
    left_EF =  left['real']**2 * const * dN2t_to_afa
    right_EF =  right['real']**2 * const * dN2t_to_afa
    raw_VAEF =np.mean(raw_EF,1)
    left_VAEF= np.mean(left_EF,1)
    right_VAEF =  np.mean(right_EF,1)
    print "movarr size", raw_EF.shape,left_EF.shape
    print "vertically averaged size" , raw_VAEF.shape,left_VAEF.shape,right_VAEF.shape


    # plotting all the timeseries

    plt.figure(figsize=(15,12))
    # raw field
    plt.subplot(3,1,1)
    plt.imshow(raw['real'].T,
               extent=[t[0],t[-1],z[-1],z[0]],
               vmax=max_min,vmin=-max_min,
               aspect='auto',interpolation= 'nearest',
               )
    plt.title('raw dz')
    plt.ylabel('depth (cm)')
    plt.colorbar()

    # L field
    # L field real
    plt.subplot(3, 1, 2)
    plt.imshow(left['real'].T,
               extent=(t[0], t[-1], z[-1], z[0]),
               vmin =-max_min, vmax=max_min,
               aspect='auto',interpolation= 'nearest',
               )
    plt.colorbar()
    plt.ylabel('depth (cm)')
    plt.title('Left')

    # R field
    # R field real
    plt.subplot(3, 1, 3)
    plt.imshow(right['real'].T,
               extent=(t[0], t[-1], z[-1], z[0]),
               vmin =-max_min, vmax=max_min,
               aspect='auto',interpolation= 'nearest',
               )
    plt.colorbar()
    plt.title('Right')
    plt.ylabel('depth (cm)')
    plt.xlabel('Time (s)')

    plt.savefig(plotName)

    plt.figure(figsize=(15,12))
    # raw field
    print raw_EF.max(),left_EF.max(),right_EF.max()

    plt.subplot(3,1,1)
    plt.imshow(raw_EF.T,
               extent=[t[0],t[-1],z[-1],z[0]],
               #vmax=max_min,vmin=-max_min,
               aspect='auto',interpolation= 'nearest',
               )
    plt.title('raw dz')
    plt.ylabel('depth (cm)')
    plt.colorbar()

    # L field
    # L field real
    plt.subplot(3, 1, 2)
    plt.imshow(left_EF.T,
               extent=(t[0], t[-1], z[-1], z[0]),
               #vmin =-max_min, vmax=max_min,
               aspect='auto',interpolation= 'nearest',
               )
    plt.colorbar()
    plt.ylabel('depth (cm)')
    plt.title('Left')

    # R field
    # R field real
    plt.subplot(3, 1, 3)
    plt.imshow(right_EF.T,
               extent=(t[0], t[-1], z[-1], z[0]),
               #vmin =-max_min, vmax=max_min ,
               aspect='auto',interpolation= 'nearest',
               )
    plt.colorbar()
    plt.title('Right')
    plt.ylabel('depth (cm)')
    plt.xlabel('Time (s)')

    plt.savefig(plotName+"EF.pdf")

    title1 = "energy flux at %d cm of the raw data" % x[plotcol]
    title3 = "energy flux at %d cm of the leftward propagating wave "  % x[plotcol]
    title2 = "energy flux at %d cm of the rightward propagating wave"  % x[plotcol]
    plt.figure(figsize=(15,12))
    plotting_functions.sharexy_plot_3plts(raw_VAEF,left_VAEF,right_VAEF,t[:],title1,title2,title3,'time','vertically averaged energy flux')

    plt.savefig(plotName + "_VertAvgdEF.pdf")


def plot_data(fw_id,plotName = None):

    fw_filename = "/data/filtered_waves/%d/waves.nc" % fw_id
    nc = netCDF4.Dataset(fw_filename, 'r')
    raw = nc.variables['raw_array']
    left = nc.variables['left_array']
    right = nc.variables['right_array']
    t = nc.variables['time'][:]
    x = nc.variables['column'][:]
    z = nc.variables['row'][:]

    nt, nz, nx = left.shape

    fig = plt.figure()


    # L field
    datain = left[100, :, :]

    # L field real
    plt.subplot(2, 1, 1)
    im_L = plt.imshow(datain['real'],
                          #extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='upper',
                          vmin =-.05, vmax=.05)
    plt.colorbar()
    plt.title('Left')

    # R field
    datain = right[100, :, :]

    # R field real
    plt.subplot(2, 1, 2)
    im_R = plt.imshow(datain['real'],
                          #extent=(x[0], x[-1], z[0], z[-1]),
                          aspect='auto', origin='upper',
                          vmin =-.05, vmax=.05)
    plt.colorbar()
    plt.title('Right')
    if plotName is not None:
        plt.savefig(plotName)
        return

    def updatefig(n, *args):

        # L field
        datain = left[n, :, :]
        im_L.set_array(datain['real'])

        # R field
        datain = right[n, :, :]
        im_R.set_array(datain['real'])

        return (im_L,  im_R, )

    ani = animation.FuncAnimation(fig, updatefig, nt, blit=True)
    plt.show()

    nc.close()

    return


def plot_fft(kx, kz, omega, F):
    # get the path to the nc file
    # Open &  Load the nc file
    print " kx shape", kx.shape
    print " kz shape", kz.shape
    print " omega shape", omega.shape

    path = "/data/deltaN2/%d" % deltaN2_id
    filename = path + "/deltaN2.nc"
    nc = netCDF4.Dataset(filename)
    deltaN2 = nc.variables['deltaN2_array']

    t = nc.variables['time']
    t = t[100:300]
    a = nc.variables['column']
    x = a[600:1000]
    b = nc.variables['row']
    z = b[300:800]
    print "t : ", t[0], "to ", t[-1]
    print "x : ", x[0], "to ", x[-1]
    print "z : ", z[0], "to ", z[-1]

    #plot kx_, kz_, omega_
    plt.figure(2)
    plt.subplot(1, 3, 1)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]

    #plt.imshow(dz[500,150:800,500:600].reshape(650,500),extent=[x[0],x[-1],z[0],z[-1]])
    #plt.title('length and depth window')

    plt.imshow(abs(mkx), interpolation='nearest',
               extent=[t[0], t[-1], z[0], z[-1]],
               vmin=0, vmax=np.pi / dx,
               aspect='auto')
    plt.xlabel('t')
    plt.ylabel('z')
    plt.title('kx')
    #plt.colorbar(ticks=[1,3,5,7])
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(abs(mkz), interpolation='nearest',
               extent=[t[0], t[-1], x[0], x[-1]],
               vmin=0, vmax=np.pi / dz,
               aspect='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('kz')
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(abs(momega).T, interpolation='nearest',
               extent=[x[0], x[-1], z[0], z[-1]],
               vmin=0, vmax=np.pi / dt,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('omega')
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()

    plt.savefig('/Volumes/HD2/users/prajvala/IGW_reflection/results/img1.jpeg')
    nc.close()
    plt.show()

def test():
    """
    Test dominant frequency finding routine
    """

    # create a grid for x, z, t

    xmin, xmax, nx = 0, 10, 50
    zmin, zmax, nz = 0, 100, 100
    tmin, tmax, dt = 0, 100, 0.5
    x = np.mgrid[xmin:xmax:nx * 1j]
    z = np.mgrid[zmin:zmax:nz * 1j]
    t = np.mgrid[tmin:tmax:dt]
    print "x", x.shape, "z ", z.shape, "t ", t.shape
    X, Z, T = np.mgrid[xmin:xmax:nx * 1j,
              zmin:zmax:nz * 1j,
              tmin:tmax:dt]
    print "X", X.shape, "Z ", Z.shape, "T ", T.shape
    # ensure nx, nz, nt, dx, dz, dt are all defined
    nx, nz, nt = len(x), len(z), len(t)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]

    # change here to explore different functional forms
    kx0 = 2.0
    kz0 = 2.0
    omega0 = 2.0
    f = np.cos(kx0 * X + kz0 * Z - omega0 * T)
    print "F:", f.shape
    # find the peak frequencies
    kx_, kz_, omega_ = estimate_dominant_frequency(f, x, z, t)

    # plot kx_, kz_, omega_
    # The titles should match colorbars if this is working correctly
    plt.figure(figsize=(182, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(abs(kx_).T, interpolation='nearest',
               extent=[zmin, zmax, tmin, tmax],
               vmin=0, vmax=np.pi / dx,
               aspect='auto')
    plt.xlabel('z')
    plt.ylabel('t')
    plt.title('kx_ = %.2f' % kx0)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(abs(kz_).T, interpolation='nearest',
               extent=[xmin, xmax, tmin, tmax],
               vmin=0, vmax=np.pi / dz,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('kz_ = %.2f' % kz0)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(abs(omega_).T, interpolation='nearest',
               extent=[xmin, xmax, zmin, zmax],
               vmin=0, vmax=np.pi / dt,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('omega_ = %.2f' % omega0)
    plt.colorbar()

def task_DzHilbertTransform(dz_id, cache=True,rowS=0,rowE=963,colS=0,colE=1291,startF=None,period= None):
    """
    Given an dz_id, computes Hilbert transform to filter out
    Leftward and Rightward propagating waves.

    Results stored as  'left_array' and 'right_array' in a waves.nc file
    """

    db = labdb.LabDB()
    if startF is None or period is None:
        sql = "SELECT fw_id FROM filtered_waves WHERE dz_id = %d AND startF is NULL AND period is NULL" % (dz_id)
    else:
        #check if the file already exists
        sql = "SELECT fw_id FROM filtered_waves WHERE dz_id = %d AND startF= %d AND period = %d" % (dz_id,startF,period)
    rows = db.execute(sql)
    if len(rows) > 0:

        fw_id = rows[0][0]

        print "filterLR id already exists in database"
        fw_filename = "/data/filtered_waves/%d/waves.nc" % fw_id
        print fw_filename
        #just_plot(fw_path,a_xi_id,maxmin,plotcolumn)
        #plt.show()
        if os.path.exists(fw_filename) and cache:
            print "returning cached data"
            return fw_id
        else:
            # delete waves.nc file if exists
            if os.path.exists(fw_filename):
                os.unlink(fw_filename)

            # create a new wave.nc file with the same fw_id
            fw_filename, fw_id = create_nc_file_dzHT(dz_id,rowS,rowE,colS,colE, fw_id=fw_id,startF=startF,period=period)
    else:
        # create the nc file for the first time for storing the filtered data
        fw_filename, fw_id = create_nc_file_dzHT(dz_id,rowS,rowE,colS,colE,startF=startF,period=period)

    logger.info('filterLR')

    #set the path to the data
    path = "/data/dz/%d" % dz_id
    filename = path + "/dz.nc"

    # check for existance of dz_nc
    if not os.path.exists(filename):
        raise Exception('{} not found'.format(filename))

    # open the axi dataset and get the no of rows and col
    dz_nc = netCDF4.Dataset(filename, 'r')
    # Open the filtered_waves nc file for details needed for creating the intermediate HT nc file
    nc = netCDF4.Dataset(fw_filename, 'a')

    logger.debug('dz filename {}'.format(filename))

    # variables
    #time is taken from the filtered waves nc file
    t = nc.variables['time'][:]

    z = dz_nc.variables['row'][rowS:rowE]
    x = dz_nc.variables['column'][colS:colE]
    #print "Axi, time", t

    # DEBUG MESSAGES
    print "x & z shape", x.shape, z.shape

    # determine lengths of x, z, t
    nt = len(t)
    nz = len(z)
    nx = len(x)
    print "length of T, Z, X:  ", nt, nz, nx



    # STEP 1: FFT in t ######

    # create ncfile to store Hilbert transform of U
    temp_ht_filename = 'HT.nc'

    nc_ht = netCDF4.Dataset(temp_ht_filename, 'w', format='NETCDF4')

    # since at HT of U is complex, we need complex data types
    complex64 = np.dtype([('real', np.float32), ('imag', np.float32)])
    complex64_t = nc_ht.createCompoundType(complex64, 'complex64')

    # copy grid from U ncfile
    x_dim = nc_ht.createDimension('x', nx)
    z_dim = nc_ht.createDimension('z', nz)
    t_dim = nc_ht.createDimension('t', nt)

    x = nc_ht.createVariable('x', np.float32, ('x'))
    z = nc_ht.createVariable('z', np.float32, ('z'))
    t = nc_ht.createVariable('t', np.float32, ('t'))
    x[:] = nc.variables['column'][:]
    z[:] = nc.variables['row'][:]
    t[:] = nc.variables['time'][:]

    valSize = complex64.itemsize
    chunksizes = chunk_shape_3D( ( nt, nz, nx),
                                  valSize=valSize )
    Uht = nc_ht.createVariable('Uht', complex64_t, ('t', 'z', 'x'),
                               chunksizes=chunksizes)
    chunk_nt, chunk_nz, chunk_nx = chunksizes
    # TODO: chunk shape of Uht may not be the same as Dz; float vs complex

    logger.debug('Temporal filtering')
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=(nz//chunk_nz)).start()

    # loop over all z
    for i in range(nz//chunk_nz):
        pbar.update(i)

        A = dz_nc.variables['dz_array']
        #print "###",nz//chunk_nz, (rowS+i)*chunk_nz, (rowS+i+1)*chunk_nz, rowS+ i*chunk_nz, rowS+(i+1)*chunk_nz,

        #load data
        if startF is None and period is None:
            datain = A[:,
                    rowS+i*chunk_nz:rowS+(i+1)*chunk_nz,
                    colS:colE]
        else:
            datain = A[startF:startF+nt,
                    rowS+i*chunk_nz:rowS+(i+1)*chunk_nz,
                    colS:colE]

        # take FFT in time
        U_spectrum = np.fft.fft(datain, axis=0)
        # only keep positive frequencies
        #   could extend this to a band pass filter

        # explicitly set all non-positive frequencies to zero
        U_spectrum[nt/2:, :] = 0
        U_spectrum[0, :] = 0 # setting the amplitude of f=0 to zero

        # take inverse FFT and multiply by 2
        datac = 2 * np.fft.ifft(U_spectrum, axis=0)

        # convert to a compound datatype
        dataout = np.empty(datain.shape, complex64)
        dataout['real'] = datac.real
        dataout['imag'] = datac.imag

        # store in netcdf4 file
        Uht[:, 
            i*chunk_nz:(i+1)*chunk_nz,
            :] = dataout

    pbar.finish()

    #need to close the axi array nc file
    nc_ht.close()
    #need to close the LR waves nc file as well
    nc.close()

    ## STEP 2
    ### extract out only rightward and leftward propagating portions of Uht
    #nc_ht = netCDF4.Dataset(temp_ht_filename, 'r')
    #print nc_ht.variables.keys()
    #Uht = nc_ht.variables['Uht']
    #nt, nz, nx = Uht.shape
    #print chunksizes

    logger.debug('Spatial filter - frame by frame')

    # Create pool
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool = multiprocessing.Pool(processes=8)

    # submit tasks to perform in parallel
    results = [ pool.apply_async(spatial_filter, 
                            (n, lock, temp_ht_filename, fw_filename,),)
        for n in range(nt)]
    pool.close()

    # progress bar
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=nt)

    pbar.start()

    # wait for all tasks to complete

    for i, r in enumerate(results):
        pbar.update(i)
        r.get()

    pool.join()

    pbar.finish()

    #os.unlink(temp_ht_filename)
    
    print "FW_ID ## ", fw_id
    return fw_id

def spatial_filter(n, lock, temp_ht_filename, fw_filename):
    
    # grab frame
    nc = netCDF4.Dataset(temp_ht_filename, 'r')
    Uht = nc.variables['Uht']
    datain = Uht[n, :, :]
    nc.close()

    # form complex array
    nz, nx = datain.shape
    datac = np.empty((nz, nx), np.complex64)
    datac.real = datain['real']
    datac.imag = datain['imag']

    # zero out 'infinite values'
    datac[abs(datac) > 100] = 0.777

    # fft
    datac_spectrum = np.fft.fft2(datac, axes=(0,1))

    # make a copy
    datac_R_spectrum = datac_spectrum.copy()
    datac_L_spectrum = datac_spectrum # note: no copy here

    # only include right ward propagating (kx > 0)
    datac_R_spectrum[:, nx/2:] = 0
    datac_R_spectrum[:,0]=0

    # only include left ward propagating (kx < 0)
    datac_L_spectrum[:, :nx/2] = 0
    # inverse FFT
    datac_R = np.fft.ifft2(datac_R_spectrum, axes=(0,1))

    datac_L = np.fft.ifft2(datac_L_spectrum, axes=(0,1))

    with lock:
        nc = netCDF4.Dataset(fw_filename, 'a')
        left = nc.variables['left_array']
        right = nc.variables['right_array']
        raw = nc.variables['raw_array']

        # convert to compound datatype and save
        complex64 = np.dtype([('real', np.float32), ('imag', np.float32)])
        dataout = np.empty(datain.shape, complex64)

        dataout['real'] = datac_R.real
        dataout['imag'] = datac_R.imag
        right[n, :, :] = dataout

        dataout['real'] = datac_L.real
        dataout['imag'] = datac_L.imag
        left[n, :, :] = dataout

        raw[n, :, :] = datain

        nc.close()

if __name__ == "__main__":
    pass
