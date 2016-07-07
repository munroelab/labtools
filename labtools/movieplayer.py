""" 
Routine to create movies from datasets

- open nc file
- generate animation
- save as video file

Movies are sequences in time
"""

import os
import argparse
import logging

logger = logging.getLogger(__name__)


import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import animation
import os

from . import labdb
from . import progressbar

def movie(var, id, 
        max_min=None,
        start_frame=0, stop_frame=None,
        saveFig=False,
        movieName = None):
    """
    Given a 'variable name' of a given "id" make a movie
    """

    skip_frames = 4

    logger.debug('Making a movie %s %d' % (var, id))

    #  variable name : (ncdir, ncfile, ncvar)
    ncfiles = { 'video' : ('videoncfiles', 'video.nc', 'img_array'),
                'dz' : ('dz', 'dz.nc', 'dz_array'),
                'Axi' : ('vertical_displacement_amplitude', 'a_xi.nc',
                         'a_xi_array'),
              }
    ncdir, ncfile, ncvar = ncfiles[var]

    # arrays are stored in 
    path = "/data/%s/%d/%s" % ( ncdir, id, ncfile )

    if not os.path.exists(path):
        print(path, "not found")
        return

    logger.debug('Dataset is %s' %  path)

    # Load the nc file
    data = nc.Dataset(path, 'r')

    if stop_frame is None:
        # look at number of times in dataset
        stop_frame = data.variables['time'].size 

    #print "variables: ", data.variables.keys()

    # Load the variables
    arr = list(data.variables.keys())
    array = data.variables[ncvar]
    t = data.variables[arr[2]]
    z = data.variables[arr[0]]
    x = data.variables[arr[1]]

    # Need window length and height 
    win_l = x[-1]
    win_h = z[-1]
    #print "window length:" , win_l
    #print "window height:" , win_h
  
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize = (10,8))
    ax = plt.axes(xlim=(0,win_l), ylim =(win_h,0))
    plt.xlabel('window length (cm)')
    plt.ylabel('window height (cm)')
    if max_min is None:
        max_min = abs(array[start_frame,:,:]).max()

    im=plt.imshow(array[start_frame,:,:], 
                  extent=[x[0],x[-1],z[0],z[-1]],
                  vmax=+max_min, vmin=-max_min,
                  interpolation="nearest",
                  aspect=win_h/win_l,
                  origin='lower')
    plt.colorbar()
    frame_num = stop_frame - start_frame 

    title = plt.title('Frame number :: %d, time :: %.2f s, Field: %s'% (
            start_frame, t[start_frame], var))

    def animate(i, pbar):
        pbar.update(i)

        title.set_text('Frame number :: %d, time :: %.2f s, Field: %s'% (
            skip_frames*i+start_frame, t[skip_frames*i+start_frame], var))

        im.set_array(array[skip_frames*i+start_frame,:,:])

        return im, title


    # debug
    #print "Render movie %s, %d  (%d..%d, %dx%d)" % (var, id, 
            #start_frame, stop_frame, len(x), len(z))

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,
            maxval=frame_num//skip_frames).start()

    anim = animation.FuncAnimation(fig,
            animate, frames=frame_num//skip_frames, fargs=(pbar,),
            repeat=False, blit=True)

    if saveFig:
        if movieName is None:
            movieName = '%s_%d.mp4' % (var,id)

        speedup = 4
        anim.save(movieName,
             #   fps = speedup * 6.0 / skip_frames,
                  dpi=72,
          fps = 4,
          codec='libx264',
        #  extra_args=['-q', '1'],
          extra_args=['-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p'],

         # extra_args=['-b:v', '720k'],
          )



    pbar.finish()



def UI():
    parser = argparse.ArgumentParser()
    parser.add_argument("var", help = "Enter the variable to play. Example: 'dz' ")
    parser.add_argument("id",type = int, help = "Enter the id number of the var in the database")
    parser.add_argument("max_min" ,type = float, help = "Enter a value that will be used as vmax and vmin")
    parser.add_argument("--start_frame",default=0, type = int,
                    help = "Enter the frame number from which to start the\
                            animation(default=0)")
    parser.add_argument("--stop_frame",default=0, type = int, help = "Enter\
            the frame number where to stop the animation.DEFAULT=last frame")    
    parser.add_argument("--saveFig",type = int, default = 0,help ="To save an animation, saveFig = 1 ")
    # add optional arguement to override cache
    args = parser.parse_args()
    movie(args.var,args.id,args.max_min,args.start_frame,args.stop_frame,args.saveFig)

if __name__ == "__main__":
    UI()
                     

