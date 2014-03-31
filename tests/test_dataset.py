from unittest import TestCase
from labtools.dataset import Dataset
import numpy as np

__author__ = 'jmunroe'

class TestDataset(TestCase):
    def test_empty_dataset(self):
        d = Dataset(1)

        d.close()


    def test_create_complex_variable(self):
        d = Dataset(2)

        H = 10.0
        L = 50.0
        T = 120

        # define grids
        nx = 64
        nz = 64
        nt = 256

        x = np.mgrid[0:L:nx * 1j]
        z = np.mgrid[0:H:nz * 1j]
        t = np.mgrid[0:T:nt * 1j]
        d.defineGrid(x, z, t)

        X, Z = np.meshgrid(x, z, indexing='ij')

        complex64 = np.dtype([('real', np.float32), ('imag', np.float32)])
        Pc = d.addVariable('Pc', complex64)
        Qc = d.addVariable('Qc', complex64)

        d.close()

    def test_create_dataset(self):
        d = Dataset(3)

        H = 10.0
        L = 50.0
        T = 120
        omega = 0.72
        kx = 8*np.pi / L
        kz = 4*np.pi / H
        A = 1.0
        B = 0.5

        # define grids
        nx = 64
        nz = 64
        nt = 256

        x = np.mgrid[0:L:nx * 1j]
        z = np.mgrid[0:H:nz * 1j]
        t = np.mgrid[0:T:nt * 1j]
        d.defineGrid(x, z, t)

        X, Z = np.meshgrid(x, z, indexing='ij')

        U = d.addVariable('U', np.float32)

        for n in range(nt):
            U[:, :, n] = A * np.cos(kx * X + kz * Z - omega * t[n]) \
                       + A * np.cos(kx * X - kz * Z - omega * t[n]) \
                       + B * np.cos(kx * X + kz * Z + omega * t[n]) \
                       + B * np.cos(kx * X - kz * Z + omega * t[n])

        d.close()

