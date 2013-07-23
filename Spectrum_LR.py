"""
Routines for computing Fourier transforms of f(x,z,t) fields

(could be generalized later so that f(x1, x2, .., xn) 
     and variable number of axes are given)
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4
import pylab

def xzt_fft(deltaN2_id):
    """
    Given the three-dimensional array f(x,z,t) gridded onto x, z, t
    compute the Fourier transform F.

    Returns F, X, Z, T where F is the Fourier transform and 
    X, Z, T are the frequency axes
    """

    # get the path to the nc file
    # Open &  Load the nc file
    path = "/Volumes/HD3/deltaN2/%d" % deltaN2_id
    filename = path+ "/deltaN2.nc"
    nc = netCDF4.Dataset(filename)
    
    #load the variables
    deltaN2_array = nc.variables['deltaN2_array']
    t = nc.variables['time']
    x = nc.variables['column']
    z = nc.variables['row']

    # Select region of interest and convert all the variables into float16 to
    # save memory
    deltaN2_array = deltaN2_array[:,500,:]
    deltaN2_array = deltaN2_array - deltaN2_array.mean()
    deltaN2_array = np.float16(deltaN2_array)
    x = np.float16(x)
    t = np.float16(t)
    #z = np.float16(z)


    print "DeltaN2 array shape: " ,deltaN2_array.shape
    print "T shape: " ,t.shape
    print "X shape: " ,x.shape
    print "Z shape: " ,z.shape

    # determine lengths of x, z, t
    #nz = len(z)
    nx = len(x)
    nt = len(t)
    print "length of X, T:  ",nx,nt

    # assume data is sampled evenly
    #dz = z[1] - z[0]
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    print "dx,dt :: " ,dx,dt
    
    # perform FFT alone all three dimensions
    # Normalize and shift so that zero frequency is at the center
    deltaN2_fft = np.fft.fft2(deltaN2_array) 
    F = np.fft.fftshift(deltaN2_fft)
    F_invs = np.fft.ifftshift(F)
    deltan2_rec = np.fft.ifft2(F_invs)

    print "fft of deltaN2 _array:: type and size::", deltaN2_fft.dtype, deltaN2_fft.size
    print "shape:", deltaN2_fft.shape
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
    
    # create a 2D mesh grid so that omega,kx and fft have the same dimensions
    K,O=np.meshgrid(kx,omega[::-1])
    print "KX.shape" ,K.shape
    print "OMEGA.shape",O.shape
    
    #calling the filter to separate out the waves travelling right from those
    #travelling left
    F_R, F_L = filter_LR(K,O,F)

    # inverse shift and ifft the fft-ed data to get back the rightward
    # travelling and leftward travelling deltaN2
    F_Rinvs = np.fft.ifftshift(F_R)
    deltaN2_R = np.fft.ifft2(F_Rinvs)
    F_Linvs = np.fft.ifftshift(F_L)
    deltaN2_L = np.fft.ifft2(F_Linvs)
    
    plt.figure(8)
    pylab.subplot(1,2,1)
    plt.imshow(deltaN2_array.T,vmin=-0.01,vmax=0.01,)
    plt.xlabel('raw data at depth at %d cm' % z[500])
    plt.colorbar()
    pylab.subplot(1,2,2)
    plt.imshow(deltan2_rec.real.T,extent=[x[0],x[-1],t[0],t[-1]],vmin=-0.01,vmax=0.01)
    plt.xlabel('reconstructed data for sanity check at depth %d cm' % z[500])
    plt.colorbar()
    
    plt.figure(1)
    pylab.subplot(1,2,1)
    plt.imshow(deltaN2_array,extent=[x[0],x[-1],t[0],t[-1]],vmin=-0.01,vmax=0.01)
    plt.xlabel('raw data at depth at %d cm' % z[500])
    plt.colorbar()
    pylab.subplot(1,2,2)
    plt.imshow((deltaN2_R+deltaN2_L).real,extent=[x[0],x[-1],t[0],t[-1]],vmin=-0.01,vmax=0.01)
    plt.xlabel('reconstructed data  at depth %d cm' % z[500])
    plt.colorbar()
    plt.figure(2)
    pylab.subplot(1,2,2)
    plt.imshow(deltaN2_R.real,extent=[x[0],x[-1],t[0],t[-1]],vmin=-0.01,vmax=0.01)
    plt.xlabel('right')
    plt.colorbar()
    pylab.subplot(1,2,1)
    plt.imshow(deltaN2_L.real,extent=[x[0],x[-1],t[0],t[-1]],vmin=-0.01,vmax=0.01)
    plt.xlabel('left')
    plt.colorbar()
    
    plot_data(kx,omega,F,F_R,F_L)
    nc.close()
    return

def filter_LR(K,O,F):
    Fright = np.copy(F)
    Fleft = np.copy(F)
    Fright[  O < 0.0 ] = 0.0
    Fleft[  O < 0.0 ] = 0.0
    Fleft,Fright = Fleft * 2.0 ,Fright*2.0
    
    Fright[ K < 0.0 ] = 0.0
    Fleft[ K > 0.0]  = 0.0
    return Fright,Fleft

def plot_data(kx,omega,F,F_R,F_L):
    #plt.figure(4)
    #plt.imshow(K,extent=[omega[0],omega[-1],kx[0],kx[-1]], interpolation = "nearest", aspect = "auto")
    #plt.xlabel('KX')
    #plt.colorbar()
    
    #plt.figure(5)
    #plt.imshow(O,extent =[omega[0],omega[-1],kx[0],kx[-1]],interpolation="nearest", aspect="auto")
    #plt.xlabel('omega')
    #plt.colorbar()
    

    plt.figure(6)
    pylab.subplot(1,2,1)
    plt.imshow(abs(F_R), extent= [omega[0],omega[-1],kx[0],kx[-1]], interpolation= "nearest", aspect = "auto")
    plt.xlabel('FFT_R')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(abs(F_L), extent= [omega[0],omega[-1],kx[0],kx[-1]], interpolation= "nearest", aspect = "auto")
    plt.xlabel('FFT_L')
    plt.colorbar()
    plt.figure(7)
    plt.subplot(2,1,1)
    plt.imshow(abs(F_L+F_R),extent=[omega[0],omega[-1],kx[0],kx[-1]],interpolation= "nearest", aspect = "auto")
    plt.xlabel('FFT reconstructed')
    plt.colorbar()
    pylab.subplot(2,1,2)
    plt.imshow(abs(F),extent=[omega[0],omega[-1],kx[0],kx[-1]],interpolation ="nearest",aspect = "auto")
    plt.xlabel('FFT of the original data')
    plt.colorbar()

    plt.show()
    return
    

def plot_fft(kx,kz,omega,F):
    # get the path to the nc file
    # Open &  Load the nc file
    print " kx shape", kx.shape
    print " kz shape", kz.shape
    print " omega shape", omega.shape

    path = "/Volumes/HD3/deltaN2/%d" % deltaN2_id
    filename = path + "/deltaN2.nc"
    nc = netCDF4.Dataset(filename)
    deltaN2 = nc.variables['deltaN2_array']
  
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

def testing_HT():
    # Implementing the Hilbert transform with a simple cosine function
    
    # Defining the function f1,f2, and f = f1+f2
    xmin,xmax,dx = 0,200,1
    tmin,tmax,dt = 0,100,0.5
    x = np.mgrid[xmin:xmax:dx]
    t = np.mgrid[tmin:tmax:dt]
    print "x & t:" , x.shape,t.shape 
    
    #X,T = np.meshgrid(x,t)
    X,T = np.mgrid[xmin:xmax:dx,tmin:tmax:dt]
    nx,nt = len(x),len(t)
    A,B = 14.0, 26.0
    #kx = 0.20
    #W = 7
    kx = 0.5
    W= 1
    #W = 50 
    
    f1 = A * np.cos(W*T - kx*X)*1.0
    f2 = B * np.cos(W*T + kx*X)*1.0
    f = f1 + f2

    #plotting the 3 functions
    plt.figure(1)
    pylab.subplot(2,1,1)
    plt.imshow(f1,extent=[x[0],x[-1],t[0],t[-1]],vmin=-24.0,vmax=24.0)
    plt.xlabel('f1')
    plt.colorbar()
    pylab.subplot(2,1,2)
    plt.imshow(f2,extent=[x[0],x[-1],t[0],t[-1]],vmin=-24.0,vmax=24.0)
    plt.xlabel('f2')
    plt.colorbar()
    plt.figure(2)
    plt.imshow(f,extent=[x[0],x[-1],t[0],t[-1]])
    plt.xlabel('function f= f1+f2')
    plt.colorbar()
    
    # implementing HT

    #  STEP1: Calculate the FFT
    fft_f = np.fft.fft2(f)
    F = np.fft.fftshift(fft_f)
    
    print "x:",X.shape,"\nT: " ,T.shape," \n F: ",f.shape,"\nFFT of f: " ,fft_f.shape
    
    #calculating the horizontal wavenumber and the omega of the function
    wavenum = np.fft.fftfreq(nx, dx)
    wavenum = 2*np.pi*np.fft.fftshift(wavenum)
    omega = np.fft.fftfreq(nt, dt)
    omega = 2*np.pi*np.fft.fftshift(omega)
    print "wavenum: ", wavenum.shape, "\n omega: ", omega.shape
    OM,KX= np.meshgrid(omega,wavenum)
    print "KX: ", KX.shape, "\n OM: ", OM.shape
    
    # Call the function that filters out negative frequencies in fourier space
    # and multiplies the result by a constant 2.0 and separates out the +ve and
    # -ve wavenumbers

    rFFT,lFFT = filter_LR(KX,OM,F)
    
    print "RFFT:", rFFT.shape, "\nLFFT :", lFFT.shape      
    """    plt.figure(5)
    pylab.subplot(2,1,1)
    plt.imshow(rFFT.real,extent=[wavenum[0],wavenum[-1],omega[0],omega[-1]],interpolation='nearest',aspect='auto')
    plt.xlabel('right fft')
    plt.colorbar()
    pylab.subplot(2,1,2)
    plt.imshow(lFFT.real,extent=[wavenum[0],wavenum[-1],omega[0],omega[-1]],interpolation='nearest',aspect='auto')
    plt.xlabel('left fft')
    plt.colorbar()
    """
    
    #inverse shift the 2 components and ifft-ing them to get the right and left
    # propagating waves
    F_Rinvs = np.fft.ifftshift(rFFT)
    deltaN2_R = np.fft.ifft2(F_Rinvs)
    F_Linvs = np.fft.ifftshift(lFFT)
    deltaN2_L = np.fft.ifft2(F_Linvs)
    print "deltaN2_R: " ,deltaN2_R.shape, "\n deltaN2_L: ", deltaN2_L.shape

    # plotting the results
    plt.figure(3)
    plt.imshow(F.real,extent=[wavenum[0],wavenum[-1],omega[0],omega[-1]],interpolation='nearest',aspect='auto')
    plt.xlabel('fft of f=f1+f1 ')
    plt.colorbar()
    plt.figure(4)
    pylab.subplot(2,1,1)
    plt.imshow(deltaN2_R.real,extent=[x[0],x[-1],t[0],t[-1]],vmin=-24.0,vmax=24.0,interpolation='nearest',aspect ='auto')
    plt.xlabel('right')
    plt.colorbar()
    pylab.subplot(2,1,2)
    plt.imshow(deltaN2_L.real,extent=[x[0],x[-1],t[0],t[-1]],vmin=-24.0,vmax =24.0,interpolation='nearest', aspect ='auto')
    plt.xlabel('left')
    plt.colorbar()

    plt.show()

    return




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
    print "x",x.shape, "z ", z.shape,"t ",t.shape
    X, Z, T = np.mgrid[xmin:xmax:nx*1j,
                       zmin:zmax:nz*1j,
                       tmin:tmax:dt]
    print "X",X.shape, "Z ", Z.shape,"T ",T.shape
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
    print "F:",f.shape
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
    parser.add_argument("deltaN2_id",type=int,help="Enter the deltaN2 id number of the file to FFT.")
    args = parser.parse_args()
    xzt_fft(args.deltaN2_id) 
    #plot_fft(max_kx,max_kz,max_omega,args.deltaN2_id)

if __name__ == "__main__":
    #test()
    #testing_HT()
    fft_test_code()

