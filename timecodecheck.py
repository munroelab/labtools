import labdb
import pylab
import numpy
import scipy
import os

# get a list of all videos with 'time.txt'
db = labdb.LabDB()
rows = db.execute('SELECT video_id FROM video')
fps = []
for video_id, in rows:
    filename = '/Volumes/HD3/video_data/%d/time.txt' % video_id
    if os.path.exists('/Volumes/HD3/video_data/%d/time.txt' % video_id):
        # open time.txt
        t = numpy.loadtxt(filename)
        frame = t[:,0]
        time = t[:,1] 
        
        fit = scipy.polyfit(time, frame, 1)
        print fit
        fps.append(fit[0])


        print video_id

# make a histogram
pylab.hist(fps)
pylab.show()

# plot time codes and fps for each

