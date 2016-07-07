

from . import SyntheticSchlieren
from . import Compute_dn2t
from . import WaveCharacteristics
import os
import netCDF4 as nc
from matplotlib import pyplot as plt
from  matplotlib import animation

cmd = "python WaveCharacteristics.py 167"
os.system(cmd)
cmd = "python WaveCharacteristics.py 168"
os.system(cmd)
cmd = "python WaveCharacteristics.py 169"
os.system(cmd)

