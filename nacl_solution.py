"""
Provides lookups for Cs and Cw in NaCl concentration table
"""
from numpy import loadtxt
from scipy.interpolate import splrep, splev

# Load CRC table on NaCl solutions
data = loadtxt('NaCl.txt')

A_table = data[:,0]/100    # % mass fraction
rho_table = data[:,1]  # g / mL
Cw_table = rho_table*(1-A_table)  # g / mL
Cs_table = rho_table*A_table  # g / mL

def rho_to_Cs(rho_input):
    """ Using table, interpolated Cs given rho """
    # B-spline interpolation (FITPACK)
    tck = splrep(rho_table, Cs_table, s = 0) # see splrep to details
    Cs_output = splev(rho_input, tck)
    return Cs_output

def rho_to_Cw(rho_input):
    return Cw_lookup(rho_input)

def Cs_to_rho(Cs_input):
    """ Using table, iterpolate rho given A """
    # B-spline interpolation (FITPACK)
    tck = splrep(Cs_table, rho_table, s = 0) # see splrep to details
    rho_output = splev(Cs_input, tck)
    return rho_output

def Cs_lookup(rho_input):
    """ Using table, interpolated Cs given rho """
    # B-spline interpolation (FITPACK)
    tck = splrep(rho_table, Cs_table, s = 0) # see splrep to details
    Cs_output = splev(rho_input, tck)
    return Cs_output

def Cw_lookup(rho_input):
    """ Using table, interpolated Cw given rho """
    # B-spline interpolation (FITPACK)
    tck = splrep(rho_table, Cw_table, s = 0) # see splrep to details
    Cw_output = splev(rho_input, tck)
    return Cw_output

def A_lookup(rho_input):
    """ Using table, interpolate A given rho """
    # B-spline interpolation (FITPACK)
    tck = splrep(rho_table, A_table, s = 0) # see splrep to details
    A_output = splev(rho_input, tck)
    return A_output

def rho_lookup(A_input):
    """ Using table, iterpolate rho given A """
    # B-spline interpolation (FITPACK)
    tck = splrep(A_table, rho_table, s = 0) # see splrep to details
    rho_output = splev(A_input, tck)
    return rho_output
