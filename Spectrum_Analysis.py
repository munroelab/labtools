"""
Routines for computing Fourier transforms of f(x,z,t) fields

(could be generalized later so that f(x1, x2, .., xn) 
     and variable number of axes are given)
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4
def xzt_fft(dz_id):
    """
    Given the three-dimensional array f(x,z,t) gridded onto x, z, t
    compute the Fourier transform F.

    Returns F, X, Z, T where F is the Fourier transform and 
    X, Z, T are the frequency axes
    """

    # get the path to the nc file
    # Open &  Load the nc file
    dz_path = "/Volumes/HD3/dz/%d" % dz_id
    dz_filename = dz_path+ "/dz.nc"
    nc = netCDF4.Dataset(dz_filename)
    dz_array = nc.variables['dz_array']
    dz_array = dz_array[100:300,300:800,600:1000] 
    dz_array = np.float16(dz_array)
    t = nc.variables['time']
    t = t[100:300]
    t = np.float16(t)
    x = nc.variables['column']
    x = x[600:1000]
    x = np.float16(x)
    z = nc.variables['row']
    z = z[300:800]
    z = np.float16(z)
#    plt.imshow(dz_array[1,150:800,:],extent=[x[0],x[-1],z[0],z[-1]])
    print "DZ array shape: " ,dz_array.shape
    print "T shape: " ,t.shape
    print "X shape: " ,x.shape
    print "Z shape: " ,z.shape

    # determine lengths of x, z, t
    nx = len(x)
    nz = len(z)
    nt = len(t)
    print "length of X,Z, T:  ", nx,nz,nt

    # assume data is sampled evenly
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]
    print "dx,dz,dt :: " ,dx,dz,dt
    # perform FFT alone all three dimensions
    F = np.fft.fftn(dz_array) 
    print "fft of dz _array:: type and size", F.dtype,F.size
    # Normalize and shift so that zero frequency is at the center
    F = np.fft.fftshift(F)/(nt*nt*nx*nx*nz*nz)

    # determine frequency axes
    kx = np.fft.fftfreq(nx, dx)
    kx = 2*np.pi*np.fft.fftshift(kx)
    kz = np.fft.fftfreq(nz, dz)
    kz = 2*np.pi*np.fft.fftshift(kz)
    omega = np.fft.fftfreq(nt, dt)
    omega = 2*np.pi*np.fft.fftshift(omega)
    print "kx shape: ", kx.shape
    print "kz shape: ", kz.shape
    print "omega shape: ", omega.shape

    max_kx, max_kz,max_omega= estimate_dominant_frequency_fft(F, kx, kz, omega)
    print "max kx shape: ", max_kx.shape
    print "max kz shape: ", max_kz.shape
    print "max omega shape: ", max_omega.shape
    nc.close()
    return max_kx, max_kz,max_omega


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
    index = P.argmax(2)
    kx_ = kx[index]

    index = P.argmax(1)
    kz_ = kz[index]

    index = P.argmax(0)
    omega_ = omega[index]

    return kx_, kz_, omega_


def plot_fft(mkx,mkz,momega,dz_id):
    # get the path to the nc file
    # Open &  Load the nc file
    print "max kx shape", mkx.shape
    print "max kz shape", mkz.shape
    print "max omega shape", momega.shape

    dz_path = "/Volumes/HD3/dz/%d" % dz_id
    dz_filename = dz_path + "/dz.nc"
    nc = netCDF4.Dataset(dz_filename)
    dz = nc.variables['dz_array']
  
    t = nc.variables['time']
    t = t[100:300]
    a = nc.variables['column']
    x = a[600:1000]
    b = nc.variables['row']
    z = b[300:800]
    print "t : " ,t[0],"to " , t[-1]
    print "x : " ,x[0],"to " , x[-1]
    print "z : " ,z[0],"to " , z[-1]

    #plot kx_, kz_, omega_
    plt.figure(2)
    plt.subplot(1,3,1)
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    dt = t[1]-t[0]
    
    #plt.imshow(dz[500,150:800,500:600].reshape(650,500),extent=[x[0],x[-1],z[0],z[-1]])
    #plt.title('length and depth window')
    
    plt.imshow(abs(mkx), interpolation='nearest',
               extent=[t[0], t[-1],z[0],z[-1]],
               vmin = 0, vmax = np.pi/dx,
               aspect='auto')
    plt.xlabel('t')
    plt.ylabel('z')
    plt.title('kx')
    #plt.colorbar(ticks=[1,3,5,7])
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(abs(mkz), interpolation='nearest',
               extent=[t[0], t[-1], x[0], x[-1]],
               vmin = 0, vmax = np.pi/dz ,
               aspect='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('kz')
    #plt.colorbar(ticks=[1,3,5,7,9])
    plt.colorbar()
    
    plt.subplot(1,3,3)
    plt.imshow(abs(momega).T, interpolation='nearest',
               extent=[x[0], x[-1], z[0], z[-1]],
              vmin = 0, vmax = np.pi/dt,
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
    #test()
    fft_test_code()

