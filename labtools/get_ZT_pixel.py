import argparse
import netCDF4 as nc
from . import labdb
from . import movieplayer as mp
from matplotlib import pyplot as plt
import numpy


def plotimg(var,id):
    
    path,array_name,video_id= mp.checkvar(var,id)
    print("path:", path)
    print("array_name", array_name, "and video_id ", video_id)
    """    db = labdb.labDB()
    sql = SELECT win_l and win_h FROM video WHERE video_id = %d % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]
    win_h = rows[0][1]
    print "win_l :: %d, win_h :: %d " % (win_l, win_h)
    """
    # Load the nc file
    data = nc.Dataset(path, 'r')
    print("variables: ",list(data.variables.keys()))
    # Load the variables
    arr = list(data.variables.keys())
    print("ARR",arr)
    data_arr = data.variables[arr[-1]]
    t = data.variables[arr[2]]
    z = data.variables[arr[0]]
    x = data.variables[arr[1]]
    print("array shape: ", data_arr.shape, " x: ",x.shape, " z: ",z.shape," t: ",t.shape)
    print(z[100], z[200])
    plt.figure(figsize  =(11,8))
    plt.subplot(2,1,1)
    plt.imshow(data_arr[0], extent=[x[0],x[-1],z[-1],z[0]], \
            aspect = 'auto', interpolation = 'nearest')
    plt.xlabel('window length')
    plt.ylabel('window height')
    plt.subplot(2,1,2)
    plt.imshow(data_arr[0],aspect = 'auto', interpolation = 'nearest')
    plt.xlabel('pixel')
    plt.ylabel('pixel')

    plt.show()


def UI():
    parser = argparse.ArgumentParser()
    parser.add_argument("var", help = "Enter the variable choices:: dz dn2t axi ")
    parser.add_argument("id",type = int, help = "Enter the id number of the var in the database")
    args = parser.parse_args()
    plotimg(args.var,args.id)

if __name__ == "__main__":
    UI()
                     


