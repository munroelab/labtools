"""
Routines for computing Fourier transforms of f(x,z,t) fields

(could be generalized later so that f(x1, x2, .., xn) 
     and variable number of axes are given)
"""
import numpy as np
import matplotlib.pyplot as plt
import schlieren

def fxzt_fft(f, x, z, t):
    """
    Given the three-dimensional array f(x,z,t) gridded onto x, z, t
    compute the Fourier transform F.

    Returns F, X, Z, T where F is the Fourier transform and 
    X, Z, T are the frequency axes
    """

    # determine lengths of x, z, t
    nx = len(x)
    nz = len(z)
    nt = len(t)

    # sanity check for data consistency
    if f.shape != (nt, nx, nz):
        print "Array has shape %s but nx,nz,nt = %d,%d,%d" % (f.shape,
                nx, nz, nt)
        return None

    # assume data is sampled evenly
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]

    # perform FFT alone all three dimensions
    F = np.fft.fftn(f) 
    
    # Normalize and shift so that zero frequency is at the center
    F = np.fft.fftshift(F)/(nt*nt*nx*nx*nz*nz)

    # determine frequency axes
    kx = np.fft.fftfreq(nx, dx)
    kx = 2*np.pi*np.fft.fftshift(kx)
    kz = np.fft.fftfreq(nz, dz)
    kz = 2*np.pi*np.fft.fftshift(kz)
    omega = np.fft.fftfreq(nt, dt)
    omega = 2*np.pi*np.fft.fftshift(omega)

    return F, kx, kz, omega

def xzt_fft( x, z, t):
    """
    Given the three-dimensional array f(x,z,t) gridded onto x, z, t
    compute the Fourier transform F.

    Returns F, X, Z, T where F is the Fourier transform and
    X, Z, T are the frequency axes
    """

    # determine lengths of x, z, t
    nx = len(x)
    nz = len(z)
    nt = len(t)


    # assume data is sampled evenly
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]

    # determine frequency axes
    kx = np.fft.fftfreq(nx, dx)
    kx = 2*np.pi*np.fft.fftshift(kx)
    kz = np.fft.fftfreq(nz, dz)
    kz = 2*np.pi*np.fft.fftshift(kz)
    omega = np.fft.fftfreq(nt, dt)
    omega = 2*np.pi*np.fft.fftshift(omega)

    return kx, kz, omega

def estimate_dominant_frequency_fft(F, kx, kz, omega):
    """
    Given the Fourier spectrum F(kx, kz, omega), estimate the 
    dominant frequency along each component.

    returns the dominant frequencies as 2D arrays:
        kx_(z,t), kx_(x,t), omega_(x,z)
    """
    # compute power spectrum
    P = abs(F)
    
    # identify the peak in each spectrum for each component
    index = F.argmax(1)
    kx_ = kx[index]

    index = F.argmax(2)
    kz_ = kz[index]

    index = F.argmax(0)
    omega_ = omega[index]

    return kx_, kz_, omega_

def estimate_dominant_frequency(f, x, z, t):
    """
    Given an array f(x, z, t), estimate the dominant frequency
    along each axis

    returns the dominant frequencies as 2D arrays:
        kx_(z,t), kx_(x,t), omega_(x,z)
    """
    
    F, kx, kz, omega = xzt_fft(f, x, z, t)
    kx_, kz_, omega_ = estimate_dominant_frequency_fft(F, kx, kz, omega)

    return kx_, kz_, omega_

def get_arrays_XZT(dz_array,win_L,win_H,path2time):
    """
    a function to compute dx,dz,dt and return arrays X,Z,T
    """

    time = np.loadtxt(path2time)
    #count set for test run only
    count = 5.0
    dt = np.mean(np.diff(time[:,1])) * count
    
    nt = dz_array.shape[0]
    nx = dz_array.shape[1]
    nz = dz_array.shape[2]
    dx = win_L/nx
    dz = win_H/nz
    x = np.mgrid[0:win_L:nx*1j]
    z = np.mgrid[0:win_H:nz*1j]
    t = np.mgrid[0:nt*dt:nt*1j]
    #X,Z,T = np.mgrid[0:win_L:nx*1j,0:win_H:nz*1j,0:nt*dt:nt*1j]
    #return x,z,t,X,Z,T
    return x,z,t

def plot_fft(kx,kz,omega,x,z,t,plot_name = 'XZT_FFT.pdf'):
    #plot kx_, kz_, omega_

    dx = x[1]-x[0]
    dz = z[1]-z[0]
    dt = t[1]-t[0]
    xmin = x[0]
    xmax = x[-1]
    zmin = z[0]
    zmax = z[-1]
    tmin = t[0]
    tmax = t[-1]
    plt.figure(figsize=(12,10))
    plt.subplot(3,1,1)
    plt.imshow(abs(kx).T, interpolation='nearest',
               extent=[zmin, zmax, tmin, tmax],
               vmin = 0, vmax = np.pi/dx,
               aspect='auto')
    plt.xlabel('z')
    plt.ylabel('t')
    plt.colorbar()

    plt.subplot(3,1,2)
    plt.imshow(abs(kz).T, interpolation='nearest',
               extent=[xmin, xmax, tmin, tmax],
               vmin = 0, vmax = np.pi/dz,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()

    plt.subplot(3,1,3)
    plt.imshow(abs(omega).T, interpolation='nearest',
               extent=[xmin, xmax, zmin, zmax],
              vmin = 0, vmax = np.pi/dt,
               aspect='auto')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.colorbar()
    plt.savefig(plot_name+'.pdf')
    #plt.show()

def test():
    """
    Test dominant frequency finding routine
    """

    # create a grid for x, z, t
    xmin, xmax, nx = 0, 6, 30
    zmin, zmax, nz = 0, 100, 100
    tmin, tmax, dt = 0, 500, 0.5
    x = np.mgrid[xmin:xmax:nx*1j]
    z = np.mgrid[zmin:zmax:nz*1j]
    t = np.mgrid[tmin:tmax:dt]
    X, Z, T = np.mgrid[xmin:xmax:nx*1j,
                       zmin:zmax:nz*1j,
                       tmin:tmax:dt]
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

    # find the peak frequencies
    kx_, kz_, omega_ = estimate_dominant_frequency(f, x, z, t)

    # plot kx_, kz_, omega_
    # The titles should match colorbars if this is working correctly
    plt.figure(figsize=(12,5))

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
    """
     separate test program to check if fft works
    """
    dz_array = schlieren.compute_dz(49, 15)

    win_L = 67
    win_H = 47
    x,z,t = get_arrays_XZT(dz_array,win_L,win_H,path2time)

    kx,kz,omega = estimate_dominant_frequency(dz_array,x,z,t)
    plot_fft(kx,kz,omega,x,z,t)
    plt.show()

if __name__ == "__main__":
    test()
    #fft_test_code()
    plt.show()
