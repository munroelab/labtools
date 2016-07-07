"""
Routines for computing Fourier transforms of f(x,z,t) fields

(could be generalized later so that f(x1, x2, .., xn) 
     and variable number of axes are given)
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4
def xzt_fft(ncfile, ncvar,
            timeS=0,timeE=None,
            ):
    """
    Given the three-dimensional array f(x,z,t) gridded onto x, z, t
    compute the Fourier transform F.

    Returns F, X, Z, T where F is the Fourier transform and 
    X, Z, T are the frequency axes
    """
    tstep=10
    zstep=10
    xstep=10
    # get the path to the nc file
    # Open &  Load the nc file
    nc = netCDF4.Dataset(ncfile)
    t = nc.variables['time']
    if timeE is None:
        timeE = t.shape[0]
    t = t[timeS:timeE:tstep]
    x = nc.variables['column'][::xstep]
    z = nc.variables['row'][::zstep]

    A = nc.variables[ncvar]
    print("A shape is ", A.shape)
    print("A type is " ,A.dtype)
    complex64_t = np.dtype([('real', '<f4'), ('imag', '<f4')])


    nt = (timeE - timeS -1) // tstep +1
    a_nt, a_nz , a_nx = A.shape
    nz = (a_nz-1) //zstep +1
    nx  = (a_nx-1) //xstep +1

    print(nt,nx,nz)
    dz_array = np.empty( (nt, nz, nx), dtype=np.complex64)
    for nn, n in enumerate(range(timeS, timeE, tstep)):
            frame2 = A[n, :, :]
            frame = frame2[::zstep, ::xstep]
            dz_array[nn, :, :].real = frame['real']
            dz_array[nn, :, :].imag = frame['imag']


#    plt.imshow(dz_array[1,150:800,:],extent=[x[0],x[-1],z[0],z[-1]])
    print("DZ array shape: " ,dz_array.shape)
    print("T shape: " ,t.shape)
    print("X shape: " ,x.shape)
    print("Z shape: " ,z.shape)

    # calculate the step in time , z and x
    dx = np.mean(np.diff(x))    #x[1] - x[0]
    dz = np.mean(np.diff(z))    #z[1] - z[0]
    dt = np.mean(np.diff(t))    #t[1] - t[0]
    print("dx,dz,dt :: " ,dx,dz,dt)

    # perform FFT alone all three dimensions
    F = np.fft.fftn(dz_array) 
    print("fft of dz _array:: type and size", F.dtype,F.size)
    # Normalize and shift so that zero frequency is at the center
    F = np.fft.fftshift(F)

    # determine frequency axes
    kx = np.fft.fftfreq(nx, dx)
    kx = 2*np.pi*np.fft.fftshift(kx)

    kz = np.fft.fftfreq(nz, dz)
    kz = 2*np.pi*np.fft.fftshift(kz)

    omega = np.fft.fftfreq(nt, dt)
    omega = 2*np.pi*np.fft.fftshift(omega)

    print("kx shape: ", kx.shape)
    print("kz shape: ", kz.shape)
    print("omega shape: ", omega.shape)

    max_kx, max_kz,max_omega= estimate_dominant_frequency_fft(F)
    print("max kx shape: ", max_kx.shape)
    print("max kz shape: ", max_kz.shape)
    print("max omega shape: ", max_omega.shape)
    nc.close()
    return kx,kz,omega, max_kx, max_kz,max_omega


def estimate_dominant_frequency_fft(F):
    """
    Given the Fourier spectrum F(kx, kz, omega), estimate the 
    dominant frequency along each component.

    returns the dominant frequencies as 2D arrays:
        kx_(z,t), kx_(x,t), omega_(x,z)
    """
    # compute power spectrum
    P = abs(F)
    # print "P:", P.shape
    # # identify the peak in each spectrum for each component
    # index = P.argmax(2)
    # kx_ = kx[index]
    #
    # index = P.argmax(1)
    # kz_ = kz[index]
    #
    # index = P.argmax(0)
    # omega_ = omega[index]
    kx_ = P.sum(2)
    kz_ = P.sum(1)
    omega_ = P.sum(0)


    return kx_, kz_, omega_


def plot_fft(timeS,timeE,tstep,rowS,rowE,zstep,colS,colE,xstep,mkx,mkz,momega,dz_id):
    # get the path to the nc file
    # Open &  Load the nc file
    print("max kx shape", mkx.shape)
    print("max kz shape", mkz.shape)
    print("max omega shape", momega.shape)

    dz_path = "/data/dz/%d" % dz_id
    dz_filename = dz_path + "/dz.nc"
    nc = netCDF4.Dataset(dz_filename)
    dz = nc.variables['dz_array'][20,rowS:rowE:zstep,colS:colE:xstep]
    t = nc.variables['time'][timeS:timeE:tstep]
    x = nc.variables['column'][colS:colE:xstep]
    z = nc.variables['row'][rowS:rowE:zstep]

    print(dz.shape)
    plt.figure()
    plt.imshow(dz)
    plt.colorbar()

    #plot kx_, kz_, omega_
    # determine lengths of x, z, t
    nx = len(x)
    nz = len(z)
    nt = len(t)
    dx = np.mean(np.diff(x))    # x[1]-x[0]
    dz = np.mean(np.diff(z))    # z[1]-z[0]
    dt = np.mean(np.diff(t))    # t[1]-t[0]
    kx = np.fft.fftfreq(nx, dx)
    kx = 2*np.pi*np.fft.fftshift(kx)
    kz = np.fft.fftfreq(nz, dz)
    kz = 2*np.pi*np.fft.fftshift(kz)
    omega = np.fft.fftfreq(nt, dt)
    omega = 2*np.pi*np.fft.fftshift(omega)
    # plt.figure()
    # plt.imshow(dz,extent=[x[0],x[-1],z[0],z[-1]])
    # plt.title('length and depth window')
    # plt.colorbar()

    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(abs(mkx), interpolation='nearest',
               extent=[kz[0], kz[-1],omega[-1],omega[0]],
               #vmin = 0, vmax =np.pi/dx,
               aspect='auto')
    plt.xlabel('kz')
    plt.ylabel('omega')
    plt.title('kx')
    #plt.colorbar(ticks=[1,3,5,7])
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(abs(mkz), interpolation='nearest',
               extent=[kx[0], kx[-1], omega[-1], omega[0]],
               #vmin = 0, vmax =np.pi/dz ,
               aspect='auto')
    plt.xlabel('kx')
    plt.ylabel('omega')
    plt.title('kz')
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.imshow(abs(momega), interpolation='nearest',
               extent=[kx[0], kx[-1], kz[-1], kz[0]],
               aspect='auto')
    plt.xlabel('kx')
    plt.ylabel('kz')
    plt.title('omega')
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()
    
    #plt.savefig('/Volumes/HD2/users/prajvala/IGW_reflection/results/img1.jpeg')
    nc.close()
    plt.show()




def plot_3Dfft_dominant_frequency(k_xax,k_yax,raw,left,right,xlab,ylab,title,plotname):
    # get the path to the nc file
    # Open &  Load the nc file
    print("max kx shape", raw.shape,left.shape,right.shape)

    plt.figure(figsize=(15,10))
    plt.subplot(1,3,1)
    plt.imshow(abs(raw), interpolation='nearest',
               extent=[k_xax[0], k_xax[-1],k_yax[-1],k_yax[0]],
               #vmin = 0, vmax =np.pi/dx,
               aspect='auto')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    #plt.colorbar(ticks=[1,3,5,7])
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(abs(left), interpolation='nearest',
               extent=[k_xax[0], k_xax[-1],k_yax[-1],k_yax[0]],
               #vmin = 0, vmax =np.pi/dz ,
               aspect='auto')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(abs(right), interpolation='nearest',
               extent=[k_xax[0], k_xax[-1],k_yax[-1],k_yax[0]],
               aspect='auto')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()

    plt.savefig(plotname)


def test():
    """
    Test dominant frequency finding routine
    """

    # create a grid for x, z, t
    xmin, xmax, nx = 0, 10, 50
    zmin, zmax, nz = 0, 100, 100
    tmin, tmax, dt = 0, 100, 0.5
    x = np.mgrid[xmin:xmax:nx*1j]
    z = np.mgrid[zmin:zmax:nz*1j]
    t = np.mgrid[tmin:tmax:dt]
    print("x",x.shape, "z ", z.shape,"t ",t.shape)
    X, Z, T = np.mgrid[xmin:xmax:nx*1j,
                       zmin:zmax:nz*1j,
                       tmin:tmax:dt]
    print("X",X.shape, "Z ", Z.shape,"T ",T.shape)
    # ensure nx, nz, nt, dx, dz, dt are all defined
    nx, nz, nt = len(x), len(z), len(t)
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]

    # change here to explore different functional forms
    kx0 = 2.0
    kz0 = 2.0
    omega0 = 2.0
    f = np.cos(kx0*X + kz0*Z - omega0*T)
    print("F:",f.shape)
    # find the peak frequencies
    kx_, kz_, omega_ = estimate_dominant_frequency(f, x, z, t)

    # plot kx_, kz_, omega_
    # The titles should match colorbars if this is working correctly
    plt.figure(figsize=(182,5))

    plt.subplot(1,3,1)
    plt.imshow(abs(kx_).T, interpolation='nearest',
               extent=[zmin, zmax, tmin, tmax],
               vmin = 0, vmax = np.pi/dx,
               aspect='auto')
    plt.xlabel('z')
    plt.ylabel('t')
    plt.title('kx_ = %.2f' % kx0)
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(abs(kz_).T, interpolation='nearest',
               extent=[xmin, xmax, tmin, tmax],
               vmin = 0, vmax = np.pi/dz,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('kz_ = %.2f' % kz0)
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(abs(omega_).T, interpolation='nearest',
               extent=[xmin, xmax, zmin, zmax],
              vmin = 0, vmax = np.pi/dt,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('omega_ = %.2f' % omega0)
    plt.colorbar()

def fft_test_code():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("dz_id",type=int,help="Enter the dz id of the dz.nc file extract the wavenumbers and frequency from.")
    args = parser.parse_args()
    max_kx,max_kz,max_omega= xzt_fft(args.dz_id)
    plot_fft(max_kx,max_kz,max_omega,args.dz_id)

if __name__ == "__main__":
    test()
    #fft_test_code()

