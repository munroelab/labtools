"""
This module subclasess a netCDF4 Dataset and provides support
for database integration and file system management

Datasets contain one or more variables of the form

Q(x, z, t)

where x, z, and t are uniformly gridded axes

For each variable, we need

variable name
data type
long name

variables are chunked to optimize I/O temporally and spatially

datasets are open for writing - create
datasets are opened for appending?
datasets are opened for reading
    # dataset must already exist

"""
import numpy as np
import netCDF4
from chunk_shape_3D import chunk_shape_3D

class Dataset(netCDF4.Dataset):

    def __init__(self, nc_id=None):

        # nc_id is a identifier for this dataset

        # if nc_id is None, it means we are creating a new dataset

        # if nc_id is not None, then we want to open a dataset with specific id
        #   what if that id does not exist?


        #nc = netCDF4.Dataset.__init__(self, filename, 'w')

        if nc_id is None:
            # select a new nc_id
            filename = 'temp.nc'
        else:
            filename = "sample_%d.nc" % nc_id
        super(Dataset, self).__init__(filename, 'w', format='NETCDF4')
        #super(Dataset, self).__init__(filename, 'r')

    def defineGrid(self, x, z, t):
        # x, z, t are 1D sequences that define the size of the array

        nx = len(x)
        nz = len(z)
        nt = len(t)

        self.createDimension('x', nx)
        self.createDimension('z', nz)
        self.createDimension('t', nt)

        X = self.createVariable('x', np.float32, ('x'))
        Z = self.createVariable('z', np.float32, ('z'))
        T = self.createVariable('t', np.float32, ('t'))

        X[:] = np.array(x)
        Z[:] = np.array(z)
        T[:] = np.array(t)

    def addVariable(self, varName, dtype):
        # create a variable of a given name and
        # a numpy datatype

        nx = len(self.dimensions['x'])
        nz = len(self.dimensions['z'])
        nt = len(self.dimensions['t'])

        # ensure data type isa dtype
        dtype = np.dtype(dtype)
        valSize = dtype.itemsize
        chunksizes = chunk_shape_3D( (nx, nz, nt),
                                  valSize=valSize)
        if dtype.isbuiltin==0:
            # datatype has fields
            if dtype.name in self.cmptypes:
                dtype_t = self.cmptypes[dtype.name]
            else:
                dtype_t = self.createCompoundType(dtype, dtype.name)
        else:
            dtype_t = dtype

        return self.createVariable(varName, dtype_t, ('x', 'z', 't'),
                          chunksizes=chunksizes)

