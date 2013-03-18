"""
test program for implementing line based synthetic schlieren 
using matrx based operations
"""
import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import argparse
import Image
import pylab
import numpy
import time

def getTol(image, mintol = 10):
    """
    estimate monoticity of data
    """

    nrows, ncols = image.shape

    # work with float point arrays
 #   image = image * 1.0

    # compute diference between rows
    d = image[:-1,:] - image[1:,:]

    z = numpy.zeros(ncols)
    # construct array of difference between current and upper
    A = numpy.vstack((d, z))
    # and between current and lower row
    B = numpy.vstack((z, d))
    # only estimate if the sequence is monotonic and differences
    # are at least mintol 
    C = (abs(A) >= mintol) & (abs(B) >= mintol) & (A*B>0)
    return C

def compute_dz(im1, im2, dz = 1.0):
    """
    Estimate dz given im1 and im2 (where im1 is the reference image)
    """

    # make things all float
    im1 = im1 * 1.0
    im2 = im2 * 1.0

    # assert img.shape = img0.shape

    N, M = im1.shape
    zerorow = numpy.zeros(M)

    A = im2[:,:] - im1[:,:] 
     #   0,1,2,N-1   0,1,2,3, N-1

    B = im2[:-1,:] - im1[1:,:]
    B = numpy.vstack((B, zerorow))

    #     1,2,3,..N-1   -   0,1,2,..N-2
    C = im2[1:,:] - im1[:-1,:]
    C = numpy.vstack((zerorow, C))

    D = im1[:-1,:] - im1[1:,:]
    D = numpy.vstack((zerorow, D))

    E = im1[:-2,:] - im1[2:,:]
    E = numpy.vstack((zerorow, E, zerorow))

    F = im1[1:,:] - im1[:-1,:]
    F = numpy.vstack((F, zerorow))
    
    #ans1 = - dz * (A*B)/(D*E)
    #ans2 = - dz * (A*C)/(F*E)
    #ans = ans1 + ans2

    ans = -dz * A/E * (B/D + C/F)

    return ans

   # if j == 0:  # linear interpolation at bottom boundary 
   #     deltaZ = (zp-z0)*(img[i,0]-img0[i0,0])/(img0[i0,1]-img0[i0,0])
   # elif j == img0.nz-1: # linear interpolation at top boundary 
   #     deltaZ = (zm-z0)*(img[i,-1]-img0[i0,-1])/(img[i0,-2]-img[i0,-1])
   # else: # quadratic interpolation 
   #     deltaZ = (zm-z0)*(img[i,j]-img0[i0, j])*(img[i, j]-img0[i0, j+1]) \
   #         /((img0[i0, j-1]-img0[i0, j])*(img0[i0, j-1]-img0[i0, j+1])) \
   #         + (zp-z0)*(img[i,j]-img0[i0, j])*(img[i ,j]-img0[i0, j-1]) \
   #         /((img0[i0, j+1]-img0[i0, j])*(img0[i0, j+1]-img0[i0, j-1]))
   # #print i, j, zm - z0, zp - z0, deltaZ
   # return deltaZ

from scipy import ndimage

def movie(video_id):
    # Need two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"

    count = 10
    n = 2
    filename1 = path % (video_id, count - n)
    filename2 = path % (video_id, count)

    image1 = numpy.array(Image.open(filename1))
    image2 = numpy.array(Image.open(filename2))

    H = 52.0
    dz = H / image1.shape[0]

    mintol = 15
    C = getTol(image1, mintol = mintol)
    delz = compute_dz(image1, image2, dz) 
    delz = numpy.nan_to_num(delz) * C

    vmax = 0.01

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    img = pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax,
                    animated=False, label='delz', aspect='auto')
    pylab.colorbar()
    pylab.show(block=False)
    from scipy.ndimage import gaussian_filter
    from numpy import ma

    while True:
        print "render..."
        filename1 = path % (video_id, count - n)
        filename2 = path % (video_id, count)

        image1 = numpy.array(Image.open(filename1))
        image2 = numpy.array(Image.open(filename2))

        C = getTol(image1, mintol = mintol)
        delz = compute_dz(image1, image2, dz) 
        delz = numpy.nan_to_num(delz) * C
        #delz_m = ma.masked_where(C==False, delz)

        # clip large values
        bound = 1.0
        delz[delz > bound] = bound
        delz[delz < -bound] = -bound

        # fill in missing values
        filt_delz = ndimage.gaussian_filter(delz, (21,21))
        i = abs(delz) > 1e-8
        filt_delz[i] = delz[i]

        # smooth
        filt_delz = ndimage.gaussian_filter(filt_delz, 7)

        img.set_data(filt_delz)
        fig.canvas.draw()
        ax.set_title('n = %d' % count)

        count += 5
        time.sleep(0.1)

def test():
    # Need two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
    video_id = 50
    filename1 = path % (video_id, 500)
    filename2 = path % (video_id, 510)

    image1 = numpy.array(Image.open(filename1))
    image2 = numpy.array(Image.open(filename2))

    # disply both images
    pylab.figure()
    ax = pylab.subplot(231)
    pylab.imshow(image1, cmap=pylab.cm.jet,interpolation='nearest')
    pylab.title('Image1')
    pylab.colorbar()

    pylab.subplot(232, sharex=ax, sharey=ax)
    pylab.imshow(image2, cmap=pylab.cm.jet,interpolation='nearest')
    pylab.title('Image2')
    pylab.colorbar()

    pylab.subplot(233, sharex=ax, sharey=ax)
    pylab.imshow(image2*1.0-image1,interpolation='nearest')
    pylab.title('Difference')
    pylab.colorbar()

    pylab.subplot(234, sharex=ax, sharey=ax)
    C = getTol(image1, mintol = 0)
    pylab.imshow(C,interpolation='nearest')
    pylab.colorbar()
    pylab.title("valid")

    pylab.subplot(235, sharex=ax, sharey=ax)
    H = 52.0
    dz = H / image1.shape[0]
    delz = compute_dz(image1, image2, dz)
    vmax = 0.05
    pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax)
    pylab.colorbar()

    pylab.subplot(236, sharex=ax, sharey=ax)
    delz = numpy.nan_to_num(delz) * C
    vmax = 0.05
    pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax)
    pylab.colorbar()
    pylab.show()


def UI():
    # parse args here...

    movie(49)

if __name__ == "__main__":
    UI()
