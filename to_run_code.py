#import Spectrum_LR
#import FW_timseries

import matplotlib
#matplotlib.use('TkAgg')

#fw_id = Spectrum_LR.task_DzHilbertTransform(337,False)
#print fw_id
#Spectrum_LR.plot_data(130)

#FW_timseries.compute_fw_row_timeseries(111,100,0.15)

import Spectrum_Analysis

kx,kz,omega, max_kx, max_kz,max_omega = Spectrum_Analysis.xzt_fft(369,rowS=250,rowE=800)
Spectrum_Analysis.plot_fft(kx,kz,omega,max_kx, max_kz,max_omega,369)