""" 
To open an nc file and generate an animation and be able to save it as a
video file
"""

import pylab
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from  matplotlib import animation 
import labdb
import argparse
def checkvar(var,id):
    """
    Given a variable name with a given id,
    return pathToNCFile, array_name, video_id
    """

    # get EXPT ID
    print var,type(var),id
    db = labdb.LabDB()

    if var == 'deltaN2':
        path = "/Volumes/HD4/deltaN2/%d/deltaN2.nc" % (id)
        array_name ='deltaN2_array'
        sql = """ SELECT video_id  FROM deltaN2 WHERE id =  %d """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'dz':
        path = "/Volumes/HD4/dz/%d/dz.nc" % (id)
        array_name ='dz_array'
        sql = """ SELECT video_id  FROM dz WHERE dz_id =  %d """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'dn2t':
        path = "/Volumes/HD4/dn2t/%d/dn2t.nc" % (id)
        array_name ='dn2t_array'
        sql = """ SELECT video_id FROM dz where dz_id = (SELECT dz_id \
                FROM dn2t WHERE id =%d) """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'axi':
        path = "/Volumes/HD4/vertical_displacement_amplitude/%d/a_xi.nc" %(id)
        print path
        array_name ='a_xi_array'
        print array_name
        if (id <= 119):
            sql = """ SELECT video_id FROM dz WHERE dz_id = (SELECT dz_id from \
                 dn2t WHERE id = (SELECT dn2t_id from \
                 vertical_displacement_amplitude WHERE a_xi_id = %d))"""% id
        else:
            sql = """ SELECT video_id FROM dz WHERE dz_id = (SELECT dz_id from \
                 vertical_displacement_amplitude WHERE a_xi_id = %d)"""% id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'video':
        path = "/Volumes/HD4/videoncfiles/%d/video.nc" %(id)
        print path
        array_name = 'img_array'
        video_id = id

    print path,array_name,video_id
    return path,array_name,video_id


def movie(var, id, max_min,
        start_frame=0, stop_frame=None,
        saveFig=False,
        movieName = None):
    #get the full path to the nc file that you want to play as a movie

    path, array_name, video_id = checkvar(var,id)

    # Load the nc file
    data = nc.Dataset(path, 'r')

    if stop_frame is None:
        # look at number of times in dataset
        stop_frame = data.variables['time'].size 

    # debug
    print "var: ",var, "id:" , id, "max_min:" ,max_min,\
            "start_frame:",start_frame, "stop_frame:" , stop_frame

    print "variables: ",data.variables.keys()
    # Load the variables
    arr = data.variables.keys()
    array = data.variables[arr[-1]]
    t = data.variables[arr[2]]
    z = data.variables[arr[0]]
    x = data.variables[arr[1]]
    print "x:",x.shape, "z:",z.shape,"t:",t.shape
    n= start_frame
    i=0

    def a(i):
        return array[i+n,:,:]
    
    
    def animate(i):
        print "frame num", i+n
        plt.title(' Frame number :: %d , Time ::  %.2f s , Field: %s , Video ID:%d'\
                    % (i+n, t[i+n],array_name,video_id))
        im.set_array(a(i))
        return im 
    # Need window length and height 
    win_l = x[-1]
    win_h = z[-1]
    print "window length:" , win_l
    print "window height:" , win_h
    print "%s shape" % arr[-1],array.shape
  
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize = (10,8))
    ax = plt.axes(xlim=(0,win_l), ylim =(win_h,0))
    plt.xlabel('window length (cm)')
    plt.ylabel('window height (cm)')
    im=plt.imshow(a(i),extent=[x[0],x[-1],z[0],z[-1]],\
            vmax=+max_min,vmin=-max_min,interpolation="nearest",aspect=win_h/win_l,origin='lower')
    plt.colorbar()
    frame_num = stop_frame - start_frame 
    print "frames to render: ", frame_num

    anim =animation.FuncAnimation(fig,animate,frames=frame_num,repeat=False,blit=False)
    #plt.show()

    if saveFig:
        if movieName is None:
            movieName = '%s_VID%d_animation.mp4' % (array_name,video_id)
        anim.save(movieName, dpi=100,fps=7,extra_args=['-vcodec','libx264'])



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
                     

