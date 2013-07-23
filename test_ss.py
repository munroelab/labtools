import deltaN2
import SyntheticSchlieren
import os
import netCDF4 as nc
from matplotlib import pyplot as plt
from  matplotlib import animation

# generate the dz and deltaN2 fields 

dz_id = SyntheticSchlieren.compute_dz(166,15,9.6,9.6,1,0,0,4)
print dz_id
dz_id = SyntheticSchlieren.compute_dz(166,15,9.6,9.6,1,0,0,8)
print dz_id
dz_id = SyntheticSchlieren.compute_dz(166,60,9.6,9.6,1,0,0,12)
print dz_id

"""
deltaN2_id = deltaN2.compute_deltaN2(166,30,15,25,1)
print deltaN2_id
deltaN2_id = deltaN2.compute_deltaN2(166,30,20,25,1)
print deltaN2_id
deltaN2_id = deltaN2.compute_deltaN2(166,30,25,25,1)
print deltaN2_id
deltaN2_id = deltaN2.compute_deltaN2(166,30,30,25,1)
print deltaN2_id
deltaN2_id = deltaN2.compute_deltaN2(166,30,35,25,1)
print deltaN2_id

import WaveCharacteristics
ID = 51
for i in range(0,5):
    a_xi_filename = WaveCharacteristics.compute_a_xi(ID)
    ID=ID+1
    print a_xi_filename
"""
import Compute_dn2t
import WaveCharacteristics
deltaN2_id = deltaN2.compute_deltaN2(166,15,7,7,1)
print deltaN2_id
a_xi_filename = WaveCharacteristics.compute_a_xi(deltaN2_id)
print a_xi_filename


"""

# generate plots
import deltaN2_TS_col
import axi_TS_col
import Energy_flux
import pylab

dn2_id = 37
axi_id= 32
dn2_maxmin=pylab.array([0.00005,0.00005,0.00005,0.00005,0.00005,\
        0.00005,0.00001,0.00001,0.00005,0.00005,0.00005,0.00005,0.00005])
axi_maxmin=pylab.array([0.01,0.01,0.01,0.01,0.01,0.005,0.001,0.005,\
        0.005,0.005,0.003,0.003,0.003])

for j in range (0,13):
    print "dn2_id:",dn2_id
    print "axi_id:",axi_id
    mintol,sigma,filter_size = deltaN2_TS_col.get_MSF(dn2_id)
    path = "/Volumes/Macintosh HD/Users/prajvala/Desktop/figures/M%d_S%.1f_F%d/" % (mintol,sigma,filter_size)
    os.mkdir(path)

    row=100
    for i in range (1,4):
        fname = os.path.join(path,"M%d_S%.1f_F%d_deltaN2ID%d_pix%d.pdf"\
                %(mintol,sigma,filter_size,dn2_id,row))
        deltaN2_TS_col.compute_energy_flux(dn2_id,row,dn2_maxmin[j])
        plt.savefig(fname,facecolor = 'w',edgecolor= 'b',format='pdf',transparent=False)
        plt.clf()
        row=row+500

    row=100

    for i in range(1,4):
        fname= os.path.join(path,"M%d_S%.1f_F%d_axiID%d_pix%d.pdf"\
                %(mintol,sigma,filter_size,axi_id,row))
        axi_TS_col.compute_energy_flux(axi_id,row,axi_maxmin[j])
        plt.savefig(fname,facecolor = 'w',edgecolor= 'b',format='pdf',transparent=False)
        plt.clf()
        row=row+500
    fname = os.path.join(path,"M%d_S%.1f_F%d_EF.pdf"\
            %(mintol,sigma,filter_size))
    Energy_flux.compute_energy_flux(axi_id)
    plt.savefig(fname,facecolor = 'w',edgecolor='b',format='pdf',transparent=False)
    plt.clf()
    
    if dn2_id==44:
        dn2_id=dn2_id+2
    else:
        dn2_id=dn2_id+1
    axi_id=axi_id+1

plt.show()


print dz_id
i=0
# Import the netCDF file

data = nc.Dataset('/Volumes/HD3/vertical_displacement_amplitude/31/a_xi.nc','r')

#data = nc.Dataset('/Volumes/HD3/deltaN2/%d/deltaN2.nc' % dz_id,'r')

#data = nc.Dataset('/Volumes/HD3/deltaN2/20/deltaN2.nc','r')
# Print  the variables
print "variables"  , data.variables.keys()
# Load the variables of the netCDF file into python variables
array = data.variables['a_xi_array']

#array = data.variables['deltaN2_array']
#print "dz.shape: ", array.shape
#print "deltaN2.shape: ", array.shape
n=00
print array.shape
t = data.variables['time']
z = data.variables['row']
x = data.variables['column']

def a(i):
    return array[n+i,:,:]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0,53), ylim =(40,0))
plt.xlabel('window length (cm)')
plt.ylabel('window height (cm)')

#for deltaN2
im=plt.imshow(a(i),extent=[x[0],x[-1],z[0],z[-1]],vmax=0.05,vmin=-0.05,interpolation="nearest",aspect=0.75,origin='lower')

# for dz
#im=plt.imshow(a(i),extent=[x[0],x[-1],z[0],z[-1]],vmax=0.003,vmin=-0.003,interpolation="nearest",aspect=0.75,origin='lower')
plt.colorbar()


def animate(i):
    plt.title(' Frame number :: %d \n Time ::  %.2f s\n Field: dz  Video ID: 166 ' % (n+i, t[n+i]))
    im.set_array(a(i))
    return im 

anim =animation.FuncAnimation(fig,animate,frames=1000,repeat=False,blit=False)
#anim.save('dz_VID166_animation.mp4',fps=7,extra_args=['-vcodec','libx264'])
plt.show()
"""





