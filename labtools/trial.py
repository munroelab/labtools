import numpy as np
import matplotlib.pyplot as plt
import argparse
import netCDF4


nc = netCDF4.Dataset('/volumes/HD3/deltaN2/10/deltaN2.nc')

deltaN2_array = nc.variables['deltaN2_array']
deltaN2_array = np.float16(deltaN2_array)
t = nc.variables['time']
#t = np.float16(t)
x = nc.variables['column']
#x = np.float16(x)
z = nc.variables['row']
#z = np.float16(z)

print(list(nc.dimensions.keys()))
print(list(nc.variables.keys()))
