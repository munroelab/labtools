#!/usr/bin/env python
import pylab
import numpy as np
import datetime
import glob
import os
import dateutil
import argparse
from .labdb import LabDB

g = 980
rho0 = 0.9982

def plot_stratification_file(filename,zoffset=5):
    """
    Load and plot stratification data stored in filename.

    Calibration data must be available.
    """
    # broken -- issue with datetime64 in numpy 1.7???
    #converters = {0 : dateutil.parser.parse}
    #data = np.genfromtxt(filename, 
    #        dtype=[np.datetime64, np.uint16, np.float, np.float],
    #        skiprows=1, delimiter=',', unpack=True)

    z = []
    Vc = []
    f = open(filename)
    ll = f.readlines()
    for line in ll[1:]:
        d_, steps_, z_, Vc_ = line.split(',')
        d_ = np.datetime64(d_)
        steps_ = np.uint16(steps_)
        z_ = np.float(z_)
        Vc_ = np.float(Vc_)
        z.append(z_)
        Vc.append(Vc_)
    f.close()

    #for row in data:
    #    z.append(row[2])
    #    Vc.append(row[3])

    z = np.array(z)
    Vc = np.array(Vc)

    # sort so z values are always increasing
    index = np.argsort(z)
    z = z[index]
    Vc = Vc[index]

    # we need to convert from Voltage measurement to
    # density measurements (if possible)

    # look up the strat_id for this stratification profile
    db = LabDB()
    rows = db.execute("""SELECT strat_id, calib_id FROM stratification WHERE
                  path = '%s'""" % filename)
    if len(rows) > 0:
        strat_id, calib_id= rows[0]
    else:
        strat_id, calib_id = None, None
    if calib_id is None:
        print("calibration data unavailable")

        print(" plotting rho = Vc")
        rho = 1.0 * Vc
        return

    else:

        # grab calibration data
        rows = db.execute("""SELECT density, voltage
                             FROM stratification_calib
                             WHERE calib_id = %d""" % calib_id)
        data = np.array(rows)
        rho_calib = data[:,0]
        volt_calib = data[:,1]

        # do a quadratic fit
        calib = np.polyfit(volt_calib, rho_calib, 2)
        rho = np.polyval(calib, Vc)

    pylab.plot(rho, z)

    # fit a straight line and estimate N
    # z is an array going from zmin to zmax (e.g. -50 to 3)
    zmin = z[0]
    zmax = z[-1]
    if zmax > 0:
        zmax = 0.0
    
    # only selected data at least 3cm below the surface
    m = (z < (zmax-zoffset)) & (z > (zmin)) # 7 is the mixed layer depth
    fit = np.polyfit(z[m], rho[m], 1)
    fit_line = np.polyval(fit, z[m])

    pylab.plot(fit_line, z[m], "k--")
    print("N = %.3f" % np.sqrt(-g/rho0*fit[0]))

def plot_all():
    """
    Make a plot for every stratification file in the database
    """
    db = LabDB()
    plt.figure()
    rows = db.execute("""SELECT path FROM stratification
                       WHERE path IS NOT NULL""")

    for row in rows:
        filename, = row
        plot_stratification_file(filename)

    pylab.xlabel('Density (g/cm3)')
    pylab.ylabel('z (cm)')
    pylab.title(filename)

def plot_stratification(strat_id=None,zoffset=5,plot_name='stratplot.pdf'):
    """ Make a plot of the given strat_id.
    If strat_id is None, plot the "newest" stratification
    """
    db = LabDB()

    if strat_id is None:
        rows = db.execute("SELECT path FROM stratification")
        filename,  = rows[-1]
    else:
        rows = db.execute("""SELECT path FROM stratification
                WHERE strat_id = %d""" % strat_id)
        filename,  = rows[0]

    plot_stratification_file(filename,zoffset)

    pylab.xlabel('Density (g/cm3)')
    pylab.ylabel('z (cm)')
    pylab.title("Density profile of strat_id %d"%strat_id )
    pylab.savefig(plot_name)

def load_calib_data(filename, calib_id):
    """
    Load of the calibration data from a calibration file corresponding to a
    paricular calib_id
    
    """
    db = LabDB()

    print(filename)
    f = open(filename, 'r')
    # skip header line
    f.readline()
    for line in f:
        print(line)
        sample, t, rho, V = line.strip().split(',')
        t = float(t)
        rho = float(rho)
        V = float(V)

        sql = """INSERT INTO stratification_calib
                 (sample, temperature, density, voltage, calib_id)
                 VALUES
                 ('%s', %f, %f, %f, %d)
                 """ % (sample, t, rho, V, calib_id)
        print(sql)
        db.execute(sql)
        db.commit()

def load_all_calib_data():
    """
    Look for all available calibration data and load it into the 
    database.  calib_id based on directory path
    """

    return

    #DANGEROUS below here!!!
    calib_directories = glob.glob('/Volumes/HD3/strat_calib_data/*')

    for d in calib_directories:
        calib_id = int(d.split('/')[-1])
        filename = os.path.join(d, 'calib.csv')
        if os.path.exists(filename):
            print("Importing %d/calib.csv" % calib_id)
            load_calib_data(filename, calib_id)

def main():
    
    parser = argparse.ArgumentParser(description="Plot a stratification profile")
    parser.add_argument("strat_id", type=int, help = "strat_id to plot",
            default = None, nargs = '*')

    args = parser.parse_args()

    if len(args.strat_id) == 0:
        plot_stratification()
    else:
        for strat_id in args.strat_id:
            plot_stratification(strat_id)

    pylab.show()

if __name__ == "__main__":
    #load_calib_data("/Volumes/HD3/strat_calib_data/3/calib.csv", 3)
    main()
