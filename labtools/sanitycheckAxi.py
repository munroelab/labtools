from PIL import Image
import numpy
import matplotlib.pyplot as plt
import netCDF4 as nc



path2axincfile = "/Volumes/HD4/vertical_displacement_amplitude/267/a_xi.nc"
path2dzncfile = "/Volumes/HD4/dz/319/dz.nc"
path2image = "/Users/prajvala/Documents/Project_labtools/labtools/plots/vts845.png"
# open the vts
vts = Image.open(path2image)
vts = numpy.array(vts)

#open the axi nc file
data = nc.Dataset(path2axincfile, 'r')
print(list(data.variables.keys()))
a_xi = data.variables['a_xi_array']
t = data.variables['time'][:]
z = data.variables['row'][:]
x = data.variables['column'][:]

#open the dz nc file
dz = nc.Dataset(path2dzncfile,'r')
dz_array = dz.variables['dz_array']

print(a_xi.shape, dz_array.shape)
# row and column numbers
column1 = 570 # 10 cm to the left of the ball
column2 = 716 # where the ball is floating
column3 = 1000 # 10 cm to the right of the ball
row_ball = 830 # a few cm below the ball without the horizontal bands

im = Image.open('/Volumes/HD3/video_data/765/frame00000.png')
plt.figure()
plt.imshow(im,extent=[x[0],x[-1],z[-1],z[0]])
plt.axvline(x[column1])
plt.axvline(x[column2])
plt.axvline(x[column3])
plt.axhline(z[row_ball],color='white',linewidth = 2)

for i in range(100,1000,200):
    plt.figure()
    plt.imshow(a_xi[:,:,i].T,vmin = -2,vmax = 2,extent = [t[0],t[-1],z[-1],z[0]],aspect='auto')
    plt.axhline(z[row_ball],color='white',linewidth=2)
    plt.title(i)
    plt.colorbar()

#plt.figure()
#plt.imshow(vts,extent=[t[0],t[-1],z[-1],z[0]],aspect='auto')
#plt.plot(t[:],a_xi[:,row_ball,column1]+z[row_ball],'r')
#plt.axhline(z[row_ball],color='white',linewidth=2)

#plt.figure()
#plt.imshow(a_xi[:,:,column1].T,vmin = -2,vmax = 2,extent = [t[0],t[-1],z[-1],z[0]],aspect='auto')
#plt.axhline(z[row_ball],color='white',linewidth=2)
#plt.colorbar()

plt.show()
