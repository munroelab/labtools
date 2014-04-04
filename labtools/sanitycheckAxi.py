import Image
import numpy
import matplotlib.pyplot as plt
import netCDF4 as nc

axiID = 267
column1 = 570 # 10 cm to the left of the ball
column1 = 716 # where the ball is floating
column3 = 870 # 10 cm to the right of the ball
row_ball = 688


path2ncfile = "/Volumes/HD4/vertical_displacement_amplitude/267/a_xi.nc"
path2image = "/users/prajvala/Documents/Project_labtools/labtools/plots/vts845.png"
# open the vts
vts = Image.open(path2ncfile)
vts = numpy.array(vts)

#open the nc file
data = nc.Dataset(path2ncfile, 'r')
print data.variables.keys()
a_xi = data.variables['a_xi_array']
t = data.variables['time'][:]
z = data.variables['row'][:]
x = data.variables['column']

plt.figure()
plt.imshow(vts,extent=[t[0],t[-1],z[0],z[-1]],aspect='auto')
plt.plot(t[:],axi[:,row_ball,column1]+16,'r')
plot(t[:],axi[:,row_ball,column3]+16,'o')
plt.show()
