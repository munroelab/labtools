"""
test program for implementing line based synthetic schlieren 
using matrx based operations
"""
import spectrum_test
import matplotlib
#matplotlib.use('module://mplh5canvas.backend_h5canvas')
import argparse
from PIL import Image
import pylab
import numpy
import time
import os
import labdb

def getTol(image, mintol = 10):
    """
    estimate monotonicity of data
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

def compute_dz_image(im1, im2, dz = 1.0):
    """
    Estimate dz given two images im1 and im2 (where im1 is the reference image)
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
    
    """ans1 = - dz * (A*B)/(D*E)  ans2 = - dz * (A*C)/(F*E) ans = ans1 + ans2"""

    ans = -dz * A/E * (B/D + C/F)

    return ans

"""old code  ...  if j == 0:  # linear interpolation at bottom boundary 
    deltaZ = (zp-z0)*(img[i,0]-img0[i0,0])/(img0[i0,1]-img0[i0,0])
    elif j == img0.nz-1: # linear interpolation at top boundary 
        deltaZ = (zm-z0)*(img[i,-1]-img0[i0,-1])/(img[i0,-2]-img[i0,-1])
    else: # quadratic interpolation 
        deltaZ = (zm-z0)*(img[i,j]-img0[i0, j])*(img[i, j]-img0[i0, j+1]) \
            /((img0[i0, j-1]-img0[i0, j])*(img0[i0, j-1]-img0[i0, j+1])) \
            + (zp-z0)*(img[i,j]-img0[i0, j])*(img[i ,j]-img0[i0, j-1]) \
            /((img0[i0, j+1]-img0[i0, j])*(img0[i0, j+1]-img0[i0, j-1]))
   print i, j, zm - z0, zp - z0, deltaZ
    return deltaZ"""


def generate_dz_array(delz,dz_array):
    
    """
    A function to appends delz of an image to an array that will hold the values of
    a sequence of delz
    """
    dz_array.append(delz)
    return dz_array


from scipy import ndimage

def compute_dz(video_id, skip_frames=10):
    """
    Given video_id, calculate the dz array. Output is cached on disk.

    returns the array dz

    skip_frames is the number of frames to jump before computing dz
    """

    print("skip_frames is", skip_frames)
    db = labdb.LabDB()

    # check if this dz array has already been computed?
    sql = """SELECT dz_id 
             FROM dz 
             WHERE video_id = %d AND skip_frames = %d""" % (video_id,
                     skip_frames)
    rows = db.execute(sql)
    if len(rows) > 0:
        # dz array already computed
        dz_id = rows[0][0]
        print("Loading cached dz %d..." % dz_id)
        # load the array from the disk
        dz_path = "/Volumes/HD3/dz/%d" % dz_id
        dz_filename = os.path.join(dz_path, "dz.npy")
        dz_array = numpy.load(dz_filename)
        return dz_array

    # Need to compute dz for the first time

    # Set path to the two images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"
    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    
    #declare the array to hold the value of dz for every image
    dz_array = []
    n = 5
    count = n + 1
    filename1 = path % (video_id, count - n)
    filename2 = path % (video_id, count)

    image1 = numpy.array(Image.open(filename1))
    image2 = numpy.array(Image.open(filename2))

    H = 52.0
    dz = H / image1.shape[0]

    mintol = 15
    C = getTol(image1, mintol = mintol)
    delz = compute_dz_image(image1, image2, dz) 
    delz = numpy.nan_to_num(delz) * C
    dz_array = generate_dz_array(delz,dz_array)

    vmax = 0.01

    """ old ploting code
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    img = pylab.imshow(delz, interpolation='nearest', vmin=-vmax, vmax=vmax,
                    animated=False, label='delz', aspect='auto')
    pylab.colorbar()
    pylab.show(block=False) """

    from scipy.ndimage import gaussian_filter
    from numpy import ma

    sql = """SELECT num_frames FROM video
             WHERE video_id = %d""" % video_id
    rows = db.execute(sql)
    num_frames = rows[0][0]

    index = 0
    while True:
        print("render frame %d of %d" % (count, num_frames))
        index += 1

        filename1 = path % (video_id, count - n)
        filename2 = path % (video_id, count)

        if not os.path.exists(filename2):
            break

        image1 = numpy.array(Image.open(filename1))
        image2 = numpy.array(Image.open(filename2))

        C = getTol(image1, mintol = mintol)
        delz = compute_dz_image(image1, image2, dz) 
        delz = numpy.nan_to_num(delz) * C
        #delz_m = ma.masked_where(C==False, delz)
        dz_array = generate_dz_array(delz,dz_array)

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

        # Old ploting
        """ img.set_data(filt_delz)
        fig.canvas.draw()
        ax.set_title('n = %d' % count)"""

        count += skip_frames
        time.sleep(0.1)

    dz_array = numpy.array(dz_array)
    print("final dz_array.shape" ,dz_array.shape)
    print(dz_array)

    # cache dz_array to disk
    sql = """INSERT INTO dz (video_id, skip_frames)
             VALUES (%d, %d)""" % (video_id, skip_frames)
    print(sql)
    db.execute(sql)
    sql = """SELECT LAST_INSERT_ID()"""
    rows = db.execute(sql)
    dz_id = rows[0][0]

    dz_path = "/Volumes/HD3/dz/%d" % dz_id
    os.mkdir(dz_path)

    dz_filename = os.path.join(dz_path, "dz.npy")
    numpy.save(dz_filename, dz_array)
    db.commit()

    return dz_array

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
    delz = compute_dz_image(image1, image2, dz)
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
    """
    take arguments from the user :video id and skip frame number and call
    compute dz function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id",type = int, 
                        help = "Enter the video id of the frames on which to do Synthetic Schlieren")
    parser.add_argument("--skip_frames",type = int, 
                        default = 10,
                        help = "number of frames to jump while computing dz")
    ## add optional arguement to override cache

    args = parser.parse_args()
    compute_dz(args.video_id, args.skip_frames)

def fft_test_code():
    """
     separate test program to check if fft works
    """
    dz_array = compute_dz(49, 15)

    win_L = 67
    win_H = 47
    x,z,t = spectrum_test.get_arrays_XZT(dz_array,win_L,win_H,path2time)
    kx,kz,omega = spectrum_test.estimate_dominant_frequency(dz_array,x,z,t)
    spectrum_test.plot_fft(kx,kz,omega,x,z,t)

if __name__ == "__main__":
    UI()
