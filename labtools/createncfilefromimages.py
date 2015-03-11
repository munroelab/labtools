"""
Given a video_id, create an nc file of raw images
"""

from PIL import Image
import numpy as np
import os

import labdb
from dataset import Dataset
import progressbar

def video2dataset(video_id,
                  imin = 0, imax = 0,
                  jmin = 0, jmax = 0,
                  tmin = 0, tmax = 1000,
                  ):
    """
    Given a video_id, we create a Dataset contained a limited field of view

    Grid and time axis are also defined
    """

    db = labdb.LabDB()

    # for a first go, just include everything

    # get the window length and window height
    sql = """SELECT length FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_l = rows[0][0]*1.0

    sql = """SELECT height FROM video WHERE video_id = %d  """ % video_id
    rows = db.execute(sql)
    win_h = rows[0][0]*1.0

    print "length", win_l, "\nheight", win_h

    nx = 1292
    nz = 964
    x = np.arange(0, win_l, win_l/nx, dtype=np.float32)
    z = np.arange(0, win_h, win_h/nz, dtype=np.float32)

    path2time = "/Volumes/HD3/video_data/%d/time.txt" % video_id

    t = np.loadtxt(path2time)
    dt = np.mean(np.diff(t[:,1]))
    print "dt = " ,dt

    #get the number of frames
    sql = """SELECT num_frames FROM video WHERE video_id = %d""" % video_id
    rows = db.execute(sql)
    num_frames = rows[0][0]

    # uniform grid
    t = np.arange(0, num_frames*dt, dt, dtype=np.float32)

    # select a window
    # xmin, xmax -> imin, imax
    # zmin, zmax -> jmin, jmax
    # tmin, tmax -> nmin, nmax

    # determine indices, grid defined only on range that is included

    # create nc file to hold output
    d = Dataset('video', 'w')
    d.defineGrid(x, z, t)
    # video are images of intensity
    I = d.addVariable('I', np.uint8)

    # Set path to the images
    path = "/Volumes/HD3/video_data/%d/frame%05d.png"

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=num_frames).start()
    for n in range(num_frames):
        pbar.update(n)

        t_star = t[n]
        if t_star < tmin or t_star > tmax:
            continue

        filename = path % (video_id, n)
        if not os.path.exists(filename):
            break

        im = Image.open(filename)

        # TODO: need to determine these from xmin, zmin, ...
        box = (50, 50, 200, 300)
        region = np.array(im.crop(box))

        I[:, :, n] = region

    pbar.finish()

    return d.nc_id
