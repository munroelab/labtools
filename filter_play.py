import matplotlib.pyplot as plt
import numpy as np
import netCDF4
from skimage import filter
from skimage.morphology import disk

def filter_play():
    # load a dz array
    nc = netCDF4.Dataset('/Volumes/HD4/dz/248/dz.nc')

    # pull out a single frame
    A = nc.variables['dz_array'][80,:,:]

    min_max = 0.03
    clip_min_max = 0.95 * min_max
    # clip
    A[A>clip_min_max] = clip_min_max
    A[A<-clip_min_max] = -clip_min_max

    # plot the original data
    plt.figure(figsize=(10,8))
    ax = plt.subplot(1,2,1)
    plt.imshow(A,
            vmin=-min_max,
            vmax=min_max,
            interpolation='nearest',
            aspect='equal')
    plt.colorbar()
    plt.title('Original DZ')

    print "max", A.max()
    print "min",A.min()
    print A

    # map A so that -0.1 to 0.1 is mapped to 0..255
    mappedA = np.array((A + min_max)/(2*min_max)*256, dtype=np.uint8)

    # mask for the data
    maskA = np.uint8(mappedA <> 128)

    disk_size = 4
    print disk(disk_size)
    # apply filter
    Afilt= filter.rank.median(mappedA,
                 disk(disk_size),
                 mask = maskA,
                 )

    # map 0.255 to -0.2..0.2
    filteredA = (Afilt/256.0)*2*min_max - min_max

    # replace only element that were unknown
    filledA = filteredA*(1-maskA) + A*(maskA)

    sigma = 9
    filteredA = filter.gaussian_filter(filledA, [sigma, sigma])

    plt.subplot(1,2,2, sharex=ax, sharey=ax)
    plt.imshow(filteredA,
            vmin=-min_max/3,
            vmax=min_max/3,
            interpolation='nearest',
            aspect='equal')
    plt.colorbar()
    plt.title('filtered A')


    return

    #Afilt = Afilt/255.0  

    print "max", A.max()
    print "min",A.min()
    print A
    print "max", Afilt.max()
    print "min",Afilt.min()
    print Afilt
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(A,
            vmin=A.min(),
            vmax=A.max(),
            interpolation='nearest',
            aspect='auto')
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.imshow(Afilt,
            vmin=Afilt.min(),
            vmax=Afilt.max(),
            interpolation='nearest',
            aspect='auto')
    plt.colorbar()

    #shift back
    A =  A/128.0 -1.0
    Afilt = Afilt/128.0 -1.0

    print "max", A.max()
    print "min",A.min()
    print A
    print "max", Afilt.max()
    print "min",Afilt.min()
    print Afilt
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(A,
            vmin=-min_max/10.0,
            vmax=min_max/10.0,
            interpolation='nearest',
            aspect='auto')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(Afilt,
            vmin=-min_max/10.0,
            vmax=min_max/10.0,
            interpolation='nearest',
            aspect='auto')
    plt.colorbar()


if __name__ == "__main__":
    filter_play()
    plt.show()

