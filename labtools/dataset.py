"""
This module subclasses a netCDF4 Dataset and provides support
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
datasets are opened for reading
    # dataset must already exist

nc_id's are stored in a database table

"""
import numpy as np
import netCDF4
from chunk_shape_3D import chunk_shape_3D
import labdb

class Dataset(netCDF4.Dataset):

    def __init__(self, nc_id=None, mode='r'):

        # nc_id is a identifier for this dataset

        # if nc_id is None, it means we are creating a new dataset

        # if nc_id is not None, then we want to open a dataset with specific id
        #   what if that id does not exist?

        # mode only matters if nc_id is not None, otherwise the
        # netcdf file is opened for writing

        # access database
        db = labdb.LabDB()

        if nc_id is None:
            # creating a new netcdf file

            sql = "INSERT INTO datasets"
            db.execute(sql)

            sql = "SELECT LAST_INSERT_ID()"
            row = db.execute_one(sql)
            nc_id = row[0]

            db.commit()

            filename = "%d.nc" % nc_id

            super(Dataset, self).__init__(filename, 'w', format='NETCDF4')

            # set global attributes
            self.nc_id = nc_id

        else:
            #check if this dataset already exists
            sql = "SELECT nc_id FROM datasets WHERE nc_id = %d" % (nc_id)
            rows = db.execute(sql)

            if len(rows) == 0:
                # no such nc_id
                # raise exception?
                return None

            filename = "%d.nc" % nc_id

            super(Dataset, self).__init__(filename, mode, format='NETCDF4')

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

