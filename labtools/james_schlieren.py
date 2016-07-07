"""
schlieren module

Compute syntethic schlieren

checkMinTol() - helper function for schlieren
getDelZ() - helper function for schlieren

Defines schlieren_entry:
    command line tool for performing schlieren

"""

import os
import tables
import numpy
import pylab # matplotlib interface
from optparse import OptionParser
import tools
from . import progressbar
from igwtools import __version__

def checkMinTol(i, j, img, mintol):
    """
    Returns true if img[i,j-1], img[i,j], im[i,j+1] are a montonic
    sequence and differ by at least mintol
    """
    if ( j == 0 or j == (img.nz - 1) ):
        return False
    elif (((img[i, j-1] - img[i, j])*(img[i, j]-img[i, j+1]) < 0.0)
           or (abs(img[i, j-1] - img[i, j]) < mintol)
           or (abs(img[i, j] - img[i, j+1]) < mintol) ):
        return False
    else:
        return True

def getDelZ(i, j, i0, img0, img):
    """ 
    Perform linear interpolation to compute delta Z.

    i is the index to the x-coordinate
    j is the index to the z-coordinate

    img0[i, j] is the reference/init image
    img[i, j]  is the target/final image

    It is possible for img0/img0 to only be vectors.

    (algorithm transcribed from getDelZ.c in Schlieren2D)
    """
    zmin = img0.zmin
    dz = img0.dz
    zp = zmin + j*dz
    z0 = zmin + (j-1)*dz
    zm = zmin + (j-2)*dz
    if j == 0:  # linear interpolation at bottom boundary 
        deltaZ = (zp-z0)*(img[i,0]-img0[i0,0])/(img0[i0,1]-img0[i0,0])
    elif j == img0.nz-1: # linear interpolation at top boundary 
        deltaZ = (zm-z0)*(img[i,-1]-img0[i0,-1])/(img[i0,-2]-img[i0,-1])
    else: # quadratic interpolation 
        deltaZ = (zm-z0)*(img[i,j]-img0[i0, j])*(img[i, j]-img0[i0, j+1]) \
            /((img0[i0, j-1]-img0[i0, j])*(img0[i0, j-1]-img0[i0, j+1])) \
            + (zp-z0)*(img[i,j]-img0[i0, j])*(img[i ,j]-img0[i0, j-1]) \
            /((img0[i0, j+1]-img0[i0, j])*(img0[i0, j+1]-img0[i0, j-1]))
    #print i, j, zm - z0, zp - z0, deltaZ
    return deltaZ

class schlieren_options:
    """ """
    verbose = True
    nofill = False
    nosmooth = False
    mode = 'dz'
    imagetype = 'xz'
    mintol = 10
    sigma = 0.3
    ndt = 2
    fast = False

    # distance from back of tank to screen
    Ls = 24.0
    # width of tank walls
    Lp = 1.0
    # width of tank
    Lt = 17.5
    # distance from front of tank to camera
    Lc = 330.0


def schlieren_field(I, mode='qualitative', options=None):
    """
    given a field I of intensity data, compute dn2t(I)
    fields have x, z, and t values, assuming nt >= 2, nz >= 3
    I is replaced with the schlierened field
    
    the basic operation compares to vertical columns at the 
    same x but different times
    """
    
    if (I.nt <= 1) or (I.nz <= 2):
        print("Minimum 2 t and 3 z data points to be able to perform schlieren")
        return
    if options == None:
        options = schlieren_options()
        options.mode = mode

    options.zmin = I.zmin
    options.zmax = I.zmax
    options.xmin = I.xmin
    options.xmax = I.xmax
    options.tmin = I.tmin
    options.tmax = I.tmax
    options.dxt = I.dt
    options.dt = options.ndt * I.dt
    options.imagetype = 'tz'

    # convert to float
    I = I[:] * 1.0

    if I.nx == 1: # single vts
        nt, nz = I.shape
        data = numpy.empty((nt, nz), dtype="float")
        for i in range(nt):
            for j in range(nz):
                data[i,j] = I[i,j]
        data = tools.Field(data, 
                  xmin = options.xmin, xmax = options.xmax,
                  zmin = options.zmin, zmax = options.zmax,
                  tmin = options.tmin, tmax = options.tmax)
        I = schlieren(image=data, options=options)
    else:
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=I.nx).start()
        for n in range(I.nx):
            pbar.update(n)
            #print n, 'of', I.nx-1
            nz, nz, nt = I.shape
            data = numpy.empty((nt, nz), dtype="float")
            for i in range(nt):
                for j in range(nz):
                    data[i, j] = I[n, j, i]
            options.xmin = 0.0
            options.xmax = 0.0
            data = tools.Field(data, 
                  xmin = options.xmin, xmax = options.xmax,
                  zmin = options.zmin, zmax = options.zmax,
                  tmin = options.tmin, tmax = options.tmax)
            result = schlieren(image=data, options=options)
            I[n, :, :] = result[:, :].T
        pbar.finish()

    return I

    if mode in ['qualitative', 'dz', 'dn2']:
        # compare current frame with first frame
        if I.nx == 1:
            # the field is a vts, so images are columns
            reference_image = I[0,:]*1.
            for n in range(I.nt):
                print(n, 'of', I.nt-1)
                I[n,:] = schlieren(reference_image, I[n,:], options)
        else:
            reference_image = I[:,:,0]*1.
            for n in range(I.nt):
                print(n, 'of', I.nt-1)
                I[:,:,n] = schlieren(reference_image, I[:,:,n], options)
    elif mode in ['dzt', 'dn2t']:
        # compare current frame with the previous frame
         if I.nx == 1:
            # the field is a vts, so images are columns
            for n in range(1,I.nt):
                print(n, 'of', I.nt-1)
                I[n,:] = schlieren(I[n-1,:]*1., I[n,:], options)
         else:
            for n in range(1,I.nt):
                print(n, 'of', I.nt-1)
                I[:,:,n] = schlieren(I[:,:,n-1]*1., I[:,:,n], options)
    else:
        print('invalid schlieren mode')

def schlieren(reference_image = None, image = None, options = None):
    """
    Perform synthetic schlieren on an image.
    a) image is a timeseries: reference_image is not needed
    a) image is a frame: reference_image is needed
    """
    # set hardcoded parameters
    #acceleration due to gravity
    options.g = 980

    # density of pure water 
    options.rho0 = 0.9982

    # Index of refraction
    options.nw = 1.333
    options.na = 1.0
    options.np = 1.49
    options.dndrho = 0.2458
   
    if options.verbose:
        print('parameters used:')
        print('  mode =', options.mode)
        print('  imagetype =', options.imagetype)
        print('  mintol =', options.mintol)
        print('  sigma =', options.sigma)
        print('  zmin = %.2f' % options.zmin)
        print('  zmax = %.2f' % options.zmax)
        print('  dz = %.4f' % image.dz)
        if options.imagetype == 'tz':
            print('  tmin = %.2f' % options.tmin, end=' ') 
            print('  tmax = %.2f' % options.tmax, end=' ') 
            print('  dt = %.4f' % image.dt)
        else:
            print('  xmin = %.2f' % options.xmin)
            print('  xmax = %.2f' % options.xmax)
            print('  dx = %.4f' % image.dx)
        if (options.mode == 'dn2t') or (options.mode == 'dn2'):
            print('  nw = %.4f (index of refraction of water)' % options.nw)
            print('  na = %.4f (index of refraction of air)' % options.na)
            print('  np = %.4f (index of refraction of tank walls)' % options.np)
            print('  dn/drho = %.4f (rate of change of index of refraction with density)' % options.dndrho)
            print('  Lt = %.2f cm (width of tank)' % options.Lt)
            print('  Lp = %.2f cm (thickness of tank walls)' % options.Lp)
            print('  Ls = %.2f cm (distance from tank to image on screen)' % options.Ls)
            print('  Lc = %.2f cm (distance from tank to camera)' % options.Lc)
            print('  g = %.1f cm/s^2 (acceleration due to gravity)' % options.g)
            print('  rho0 = %.4f g/cm^3' % options.rho0)

    # allocate space for output
    output1 = tools.Field( numpy.zeros_like(image) , 
                xmin=options.xmin, xmax=options.xmax, 
                zmin=options.zmin, zmax=options.zmax,
                tmin=options.tmin, tmax=options.tmax)
    output2 = tools.Field( numpy.zeros_like(image) , 
                xmin = options.xmin, xmax = options.xmax, 
                zmin=options.zmin, zmax=options.zmax,
                tmin=options.tmin, tmax=options.tmax)

    if options.verbose:
        print('performing synthetic schlieren...')
    # loop through every pixel
    if options.verbose:
        print('  computing dz field...')

    if options.fast:
        print("fast: Idea is to use matrix opertions to compute dz")
        print("  ... not yet implemented")
        pass
    else:
        if options.imagetype == 'xz':
            if options.mode == 'qualitative':
                output1 = abs(image - reference_image)
                return output1

            for i in range(image.nx):
                for j in range(image.nz):
                    if checkMinTol(i, j, reference_image, options.mintol):
                        output1[i,j] = getDelZ(i, j, i, reference_image, image)
        else:
            if options.mode == 'dz' or options.mode == 'dn2':
                for j in range(image.nz):
                    if checkMinTol(0, j, image, options.mintol):
                        for i in range(image.nt):
                            output1[i,j] = getDelZ(i, j, 0, image, image)
            else: # dzt or dn2t
                for i in range(options.ndt, image.nt):
                    for j in range(image.nz):
                        if checkMinTol(i-options.ndt, j, 
                                       image, options.mintol):
                            output1[i,j] = getDelZ(i, j, i-options.ndt, 
                                                   image, image)

    # output1 contains the getDelZ values or 0 if checkMinTol failed

    if (options.mode == 'dzt' or options.mode =='dn2t'):
        output1 = output1 / options.dt

    from scipy.ndimage import gaussian_filter, correlate

    if options.nofill:
        print('  Skipping fill-in of unknown values with Gaussian average.')
        output2 = output1
    else:
        if options.verbose:
            print('  filling in unknown values with Gaussian average...')
        output2 = gaussian_filter(output1, 
                                  [options.sigma/options.dxt, 
                                   options.sigma/image.dz])
        output2 = tools.Field(output2, xmin=options.xmin, xmax=options.xmax, 
                                       zmin=options.zmin, zmax=options.zmax, 
                                       tmin=options.tmin, tmax=options.tmax)

        # output2 contains the Gaussian Filtered getDelZ values

        # replace values of output2 with original values in output1 if the
        # values in output1 where greater than MinReal (i.e. effectively zero)
        MinReal = 1e-8
        if options.imagetype == "xz":
            for i in range(image.nx):
                for j in range(image.nz):
                   if abs(output1[i,j]) >= MinReal:
                       output2[i,j] = output1[i,j]
        elif options.imagetype == "tz":
            for i in range(image.nt):
                for j in range(image.nz):
                   if abs(output1[i,j]) >= MinReal:
                       output2[i,j] = output1[i,j]
        else:
            raise "invalid imagetype: %s" % options.imagetype

    if options.nosmooth:
        print('  Skipping smoothing by uniform averaging filter.')
    else:
        if options.verbose:
            print('  applying uniform filter...')
        # create a circle of equal weights of radius options.sigma
        irange = int(options.sigma/options.dxt + 0.5)
        jrange = int(options.sigma/image.dz + 0.5)
        weights = numpy.empty((irange+1,jrange+1), dtype=float)
        for i in range(irange+1):
            for j in range(jrange+1):
                rad = numpy.sqrt((i*options.dxt)**2 + (j*image.dz)**2)
                if rad < options.sigma:
                    weights[i,j] = 1.
                else:
                    weights[i,j] = 0.
        weights = numpy.hstack((weights[:,:0:-1],weights))
        weights = numpy.vstack((weights[:0:-1,:],weights))

        weights = weights / weights.sum()

        output2 = correlate(output2, weights)
        output2 = tools.Field( output2, xmin=options.xmin, xmax=options.xmax,
                                        zmin=options.zmin, zmax=options.zmax, 
                                        tmin=options.tmin, tmax=options.tmax)

    # solve for perturbation to the squared buoyancy frequency
    # see eq 2.11 of Sutherland, Dalziel, Hughes, and Linden (1999)
    if (options.mode == 'dn2') or (options.mode == 'dn2t'):
        intn2 = 0.5 * options.Lt**2 + \
                options.Lt * options.nw * \
                (options.Lp/options.np + options.Ls/options.na)
        gamma = 1/options.g * (options.rho0 / options.nw) * options.dndrho
        #print "gamma = ", gamma
        dnsqrdz = - 1.0 / (gamma * intn2)
        #print "dnsqrdz = ", dnsqrdz
        output2 = output2 * dnsqrdz
        output2 = tools.Field( output2, xmin=options.xmin, xmax=options.xmax,
                                        zmin=options.zmin, zmax=options.zmax, 
                                        tmin=options.tmin, tmax=options.tmax)
        
    return output2

def schlieren_entry():
    """
    console script to apply schlieren routine to xyp files
    """
    usage = """ %prog [options] initial.xyp final.xyp
            for 'xz' images
        %prog [options] timeseries.xyp
            for 'tz' images"""
    parser = OptionParser(usage=usage, 
             version = "%prog (igwtools " + __version__ + ")")
    parser.add_option("-d", action = "store_true", dest = "display",
                      help = "displays images for debugging purposes")
    parser.add_option("-v", "--verbose", 
                      action = "store_true", dest = "verbose",
                      help = "be verbose and output settings used")
    parser.add_option("-T", "--mintol", 
                      dest="mintol", type="float", default = 10,
                      help = "Minimum intensity difference between pixels for computation [default: %default]")
    parser.add_option("--fast", action="store_true", default=False,
                      help = "use experimental 'fast' getdelz calculation")
    parser.add_option("-t", "--dt", type="float",
                      help = "Time difference between frames")
    parser.add_option("-n", "--ndt", type="int", default = 2,
                      help = "Take time difference between how many pixels [default : %default]")
    parser.add_option("-s", "--sigma", dest="sigma", 
                      type="float", default=0.3,
                      help = "Distance over which to fill in uncomputed points [default: %default]")
    parser.add_option("-M", "--mode", default="dn2t",
                      help = "Schlieren mode (dz, dzt, dn2, dn2t) [default: %default]")
    parser.add_option("-o", "--outputfile",
                      help = "Name of output file (output saved only if supplied)")
    parser.add_option("-f", "--force", action="store_true",
                      help = "Force overwriting of anyexisting view")

    parser.add_option("--nofill", action = "store_true",
                      help = "do not fill-in values that were not computed by interpolation by Gaussian average")
    parser.add_option("--nosmooth", action = "store_true",
                      help = "do not smooth with uniform average filter")

    parser.add_option("--Ls", type="float", default = 24.0,
            help = "Distance from back of tank to screen [default: %default]")
    parser.add_option("--Lt", type="float", default = 17.5,
            help = "Width of tank [default: %default]")
    parser.add_option("--Lp", type="float", default = 1.0,
            help = "Width of tank walls [default: %default]")
    parser.add_option("--Lc", type="float", default = 330.0,
            help = "Distance from fron of tank to camera [default: %default]")


    # look up defaults from environment
    default_database = os.getenv("IGWDB")
    default_experiment = os.getenv("IGWEXPT")

    #OptionGroup here
    parser.add_option("-D", "--database", default= default_database,
                       help = "Database name [default: %default]")
    parser.add_option("-e", "--experiment", default=default_experiment,
                       help = "Experiment name [default: %default]")
    parser.add_option("-L", "--load",
                      help = "load input from view")
    parser.add_option("-S", "--save",
                      help = "save results as view")

    (options, args) = parser.parse_args()

    # Infer whether input is xz or tz images based on number of arguments
    if len(args) == 1: # timeseries.xyp
        options.imagetype = "tz"
    elif len(args) == 2: # initial.xyp final.xyp
        options.imagetype = "xz"
    elif options.load != None:
        # load experiment
        if options.experiment[0] != '/':
            options.experiment = '/' + options.experiment
        h5file = tables.openFile(options.database, mode="r")
        valid_expt = h5file.__contains__(options.experiment)
        h5file.close()
        if valid_expt:
            expt = tools.Experiment(options.database, options.experiment)
        else:
            print(options.experiment, "not found in", options.database)
            return
        field = expt.load_view(options.load)

        if field is None:
            print("View", options.load, "not found!")

        field = schlieren_field(field, options=options)

        if options.save != None:
            if options.verbose:
                print("saving view...")
            expt.save_view(field, options.save, force=options.force)
        else:
            print("use --save option to store result!")

        expt.close()

        return
    else:
        parser.error("incorrect number of arguments")

    if options.mode not in ['dz', 'dzt', 'dn2', 'dn2t', 'qualitative']:
        parser.error("invalid schlieren mode: %s" % options.mode)
 
    import xplot
    # read in data from xyp files
    if options.verbose:
        print('reading in XYP files...')
    if options.imagetype == "xz":
        reference_image = xplot.readXYplot(args[0], orientation='xz')
        image = xplot.readXYplot(args[1], orientation='xz')
        options.dxt = image.dx
    else: # tz
        image = xplot.readXYplot(args[0], orientation='tz')
        options.dxt = image.dt
        print(image.shape)

    options.xmin = image.xmin
    options.xmax = image.xmax
    options.zmin = image.zmn
    options.zmax = image.zmax
    options.tmin = image.tmin
    options.tmax = image.tmax

    if ((options.mode == 'dzt') or (options.mode == 'dn2t')):
        if options.imagetype == 'x':
            if options.dt == None:
                parser.error("mode %s and 'xz' images requires --dt option" % \
                             options.mode)
        else:
            options.dt = options.ndt * image.dt

    if options.imagetype == "xz":
        output = schlieren(reference_image = reference_image, 
                           image = image, options = options)
    else:
        output = schlieren(image=image, options=options)

    # save output
    if(options.outputfile != None):
        if options.verbose:
            print ('saving results in xyp file')
        output.save(options.outputfile)

    if options.display:
        # show results
        vmax = 2.5*numpy.std(output)
        if options.verbose:
            print('plotting result')
            print('    vmin = %.3f' % -vmax, 'vmax = %.3f' % vmax)
        pylab.figure()
        output.plot(interpolation = 'bicubic', vmin=-vmax, vmax=vmax)
        pylab.colorbar()
        pylab.jet()
        pylab.title(options.mode)

        pylab.show()

if __name__ == '__main__':
    pass
