from . import labdb
import pylab
import numpy
from numpy import linspace,exp
from scipy.interpolate import LSQUnivariateSpline
import sys

db = labdb.LabDB()
video_id = sys.argv[1]

#sql = """SELECT dx, dz
#         FROM grids NATURAL JOIN video
#         WHERE video_id =94"""
#rows = db.execute(sql)
#print rows
#dx, dz = rows[0]
#print "dx=", dx, "dz=", dz
dz = dx = 0.0604

sql = """SELECT x, z, t
         FROM tracks
         WHERE object_id = 194 AND video_id = %s""" % video_id
rows = db.execute(sql)
data = numpy.array(rows)
i, j, t = data.T

x =   dx * i
z = - dz * j

dt = 1.0
l = numpy.arange(t[0]+1, t[-1], dt)
#s = LSQUnivariateSpline(t, x, l, k = 5)
#ts= linspace(0,450,1000)
#Xs= s(ts)
#ts2= ts[1:996]
#d = numpy.array([s.derivatives(ts[i])[1] for i in range(1,996)])
#print d
pylab.subplot(311)
pylab.plot(t, x) 
#pylab.plot(ts,Xs,'y-')
#pylab.plot(ts2,d,'r-')
pylab.ylabel("x/cm")
pylab.xlabel("t")
pylab.subplot(312)
pylab.ylabel("z/cm")
pylab.plot(t, z,) 
#pylab.plot(ts,Zs,'y-')
#pylab.plot(ts2,d,'b-')
pylab.xlabel("t")
pylab.subplot(313)
pylab.plot(x, z) 
pylab.xlabel("x/cm")
pylab.ylabel("z/cm")
#pylab.savefig( '../Plots/94.png')
pylab.show()


