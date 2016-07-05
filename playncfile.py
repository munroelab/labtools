"""
A program that will play any nc file [3D array basically!!]
"""


import netCDF4
import numpy
import pylab
import matplotlib
import time

def ncfile_movie(filename):
    #loading the nc file
    print ("file loading --> " ,filename)
    nc=netCDF4.Dataset(filename,'a')
    # some information about the nc file
    print ("dimensions of nc file -> ", nc.dimensions.keys())
    print ("variables of nc file -> ", nc.variables.keys())

    vars = nc.variables.keys()
    # loading the data into arrays
    nDarray = nc.variables[vars[-1]]
    T = nc.variables[vars[2]]
    X = nc.variables[vars[1]]
    Z = nc.variables[vars[0]]
    print ("shape of 3Darray -> ", nDarray.shape)
    print ("shape of T -> ", T.shape)
    print ("shape of X -> ", X.shape)
    print ("shape of Z -> ", Z.shape)
     
    """ old movie code
    
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    img= pylab.imshow(nDarray[0],extent=[X[0],X[-1],Z[0],Z[-1]],interpolation='nearest',animated=False,label=vars[-1],aspect='auto')
    pylab.colorbar()
    pylab.show(block=False)
    print "length", len(T)
    for i in range(len(T)):
        img.set_data(nDarray[i])
        ax.set_title('frame %d' % i)
        fig.canvas.draw()
        time.sleep(1)
        print "frame:",i
    """

def UI():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="enter the path to the nc file that you wanna play")
    args = parser.parse_args()
    ncfilemovie(args.filepath)

if __name__ == "__main__":
    UI()

