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
    # get EXPT ID
    print var,type(var),id
    db = labdb.LabDB()

    if var == 'deltaN2':
        path = "/Volumes/HD3/deltaN2/%d/deltaN2.nc" % (id)
        array_name ='deltaN2_array'
        sql = """ SELECT video_id  FROM deltaN2 WHERE id =  %d """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'dz':
        path = "/Volumes/HD3/dz/%d/dz.nc" % (id)
        array_name ='dz_array'
        sql = """ SELECT video_id  FROM dz WHERE dz_id =  %d """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'dn2t':
        path = "/Volumes/HD3/dn2t/%d/dn2t.nc" % (id)
        array_name ='dn2t_array'
        sql = """ SELECT video_id FROM deltaN2 where id = (SELECT deltaN2_id \
                FROM dn2t WHERE id =%d) """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    elif var == 'a_xi_array':
        print ",,,"
        path = "/Volumes/HD3/vertical_displacement_amplitude/%d/a_xi.nc" %(id)
        print path
        array_name ='a_xi_array'
        print array_name
        sql = """ SELECT video_id FROM deltaN2 where id = (SELECT deltaN2_id \
                FROM vertical_displacement_amplitude WHERE a_xi_id =%d) """ % id
        rows = db.execute(sql)
        video_id = rows[0][0]
    print path,array_name,video_id
    return path,array_name,video_id


def movie(var,id,max_min,start_frame,stop_frame,saveFig=0):
    #get the full path to the nc file that you want to play as a movie
    path,array_name,video_id = checkvar(var,id)
   
    # debug
    print "var: ",var, "id:" , id, "max_min:" ,max_min,\
            "start_frame:",start_frame, "stop_frame:" , stop_frame
    # Load the nc file
    data = nc.Dataset(path, 'r')
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
        plt.title(' Frame number :: %d , Time ::  %.2f s , Field: %s , Video ID:%d'\
                    % (i+n, t[i+n],array_name,video_id))
        im.set_array(a(i+n))
        return im 
    # Need window length and height 
    win_l = x[-1]
    win_h = z[-1]
    print "window length:" , win_l
    print "window height:" , win_h
    print "%s shape" % arr[-1],array.shape
  
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0,win_l), ylim =(win_h,0))
    plt.xlabel('window length (cm)')
    plt.ylabel('window height (cm)')
    im=plt.imshow(a(i),extent=[x[0],x[-1],z[0],z[-1]],\
            vmax=+max_min,vmin=-max_min,interpolation="nearest",aspect=win_h/win_l,origin='lower')
    plt.colorbar()
    frame_num = stop_frame - start_frame

    anim =animation.FuncAnimation(fig,animate,frames=frame_num,repeat=False,blit=False)
    plt.show()
    if saveFig==1:
        anim.save('%s_VID%d_animation.mp4' % (array_name,video_id),fps=7,extra_args=['-vcodec','libx264'])

    return



def UI():
    parser = argparse.ArgumentParser()
    parser.add_argument("var", help = "Enter the variable to play. Example: 'dz' ")
    parser.add_argument("id",type = int, help = "Enter the id number of the var in the database")
    parser.add_argument("max_min" ,type = float, help = "Enter a value that will be used as vmax and vmin")
    parser.add_argument("start_frame", type = int ,\
            help = "Enter the frame number from which to start the animation")
    parser.add_argument("stop_frame", type = int, help = "Enter the frame number where to stop the animation")
    parser.add_argument("--saveFig",type = int, default = 0,help ="To save an animation, saveFig = 1 ")
    # add optional arguement to override cache
    args = parser.parse_args()
    movie(args.var,args.id,args.max_min,args.start_frame,args.stop_frame,args.saveFig)

if __name__ == "__main__":
    UI()
                     

