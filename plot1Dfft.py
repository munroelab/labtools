import copy 
import movieplayer
import numpy
import argparse
import labdb
import netCDF4 as nc
import matplotlib.pyplot as plt

def get_dt(var,id):
    db = labdb.LabDB()
    if var == 'dz':
        sql = """ SELECT diff_frames FROM dz WHERE dz_id =  %d """ % id
        rows = db.execute(sql)
        diff_frames = rows[0][0]
    elif var == 'dn2t':
        sql = """ SELECT diff_frames FROM dz where dz_id =\
                (SELECT dz_id FROM dn2t where id = %d) """ % id
        rows = db.execute(sql)
        diff_frames = rows[0][0]
    elif var == 'a_xi_array':
        sql = """ SELECT diff_frames FROM dz where dz_id =\
               (SELECT dz_id FROM dn2t where id =\
               (SELECT dn2t_id FROM vertical_displacement_amplitude \ 
               WHERE a_xi_id = %d))""" % id
        rows = db.execute(sql)
        diff_frames = rows[0][0]
    print diff_frames
    return diff_frames


def compute_fft(var,id,rowstart,colstart,rowend=0,colend=0):
    path , array_name, video_id = movieplayer.checkvar(var,id)
    
    # Load the nc file
    data = nc.Dataset(path, 'r')
    print "variables: ",data.variables.keys()
    # Load the variables
    arr = data.variables.keys()
    print "ARR:: ", arr
    t = data.variables[arr[2]][:]
    print t[1:10]
    if (rowend ==0 ):
        z = data.variables[arr[0]][rowstart]
    else:
        z = data.variables[arr[0]][rowstart:rowend]

    if (colstart ==0):
        x = data.variables[arr[1]][colstart]
    else:
        x = data.variables[arr[1]][colstart:colend]

    if (rowend==0 and colend==0):
        data_array = data.variables[arr[-1]][:,rowstart,colstart]
        print "data mean shape :: " ,data_array.shape
    elif (rowend !=0 and colend!=0):
        data_array = data.variables[arr[-1]][:,rowstart:rowend,colstart:colend]
        print "data mean shape :: " ,data_array.shape
        data_array = numpy.mean(data_array,axis = (1,2))
        print "data mean shape :: " ,data_array.shape
    elif (rowend !=0 and colend ==0):
        data_array = data.variables[arr[-1]][:,rowstart:rowend,colstart]
        print "data mean shape :: " ,data_array.shape
        data_array = numpy.mean(data_array,axis =1)
        print "data mean shape :: " ,data_array.shape
    elif (rowend ==0 and colend !=0):
        data_array = data.variables[arr[-1]][:,rowstart,colstart:colend]
        print "data mean shape :: " ,data_array.shape
        data_array = numpy.mean(data_array,axis =2)
        print "data mean shape :: " ,data_array.shape
    
        
    print"data array shape: ",data_array.shape, "x:",x.shape, "z:",z.shape,"t:",t.shape
    print data_array[800:850]
    
    #determine the length of T
    nt = len(t)
    print "length of T: ", nt

    # get dt
    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id
    var1 = numpy.loadtxt(path2time)
    dt = numpy.mean(numpy.diff(var1[:,1]))
    print " dt ::" ,dt

    #perform fft along the time axis
    F = numpy.fft.fft(data_array)
    print "fft shape::", F.shape
    #normalize
    F = numpy.fft.fftshift(F)/(nt*nt)
    print "after shifting fft shape::", F.shape

    #Frequency axis
    freq = numpy.fft.fftfreq(nt,dt)
    print "freq.shape" ,freq.shape
    freq = numpy.fft.fftshift(freq) * 2* numpy.pi 
    print "after shifting freq.shape ::" ,freq.shape

    # filtering out the higher frequencies
    filtered_F = copy.copy(F)
    filtered_F[abs(freq) > 0.7]=0.0
    filtered_F[abs(freq) < 0.6] =0.0

    # inverse FFT
    filt_data = numpy.fft.ifftshift(filtered_F)*(nt*nt)
    filt_data =  numpy.fft.ifft(filt_data)


    # plotting the output
    
    plt.figure(figsize = (15,10))
    plt.subplots_adjust(hspace = 0.5)

    plt.subplot(4,1,1)
    plt.title("original signal")
    plt.plot(t,data_array)
    
    plt.subplot(4,1,2);
    plt.title("fft of the original data")
    plt.plot(freq,abs(F))
    
    plt.subplot(4,1,3);
    plt.title("filtered fft data")
    plt.plot(freq,abs(filtered_F))

    plt.subplot(4,1,4);
    plt.title("filtered data")
    plt.plot(t,filt_data)
    plt.show()



def UI(): 
    
    """
    take arguments from the user :video id and skip frame number and call
    compute dz function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("var", help=" Choices: dz dn2t axi ")
    parser.add_argument("id",type = int, help = "Enter the id of the variable \
                                whose fft timeseries you want to see")
    
    parser.add_argument("rowstart", type = int, help= "start depth pixel")
    parser.add_argument("colstart", type = int, help = "start length \
            pixel")
    parser.add_argument("--rowend",type = int, default = 0,help =\
            "End depth pixel")
    parser.add_argument("--colend",type= int, default = 0, help=\
            "end length pixel")
    ## add optional arguement to override cache

    args = parser.parse_args()
    compute_fft(args.var,args.id,args.rowstart,args.colstart,args.rowend,args.colend)


if __name__ == "__main__":
    UI()
