"""
This workflow processes all experiments

- produces plots of EnergyFlux vs Time for each experiment

"""

from ruffus import *
import sys
import os
import pickle
import datetime
import time

from labtools import SyntheticSchlieren
from labtools import WaveCharacteristics
from labtools import Spectrum_LR
from labtools import Energy_flux
from labtools import movieplayer
from labtools import axi_TS_row
from labtools import axi_TS_col
from labtools import createncfilefromimages
from labtools import labdb
from labtools import predictions_wave

workingdir = "workflow/"
moviedir = "movies/"
plotdir = "plots/"
tabledir = "tables/"
cacheValue = False 

@follows(mkdir(workingdir))
@follows(mkdir(moviedir))
@follows(mkdir(plotdir))
@follows(mkdir(tabledir))
@split(None, workingdir + '*.expt_id')
def forEachExperiment(infiles, outfiles):
    
    # select experiments
    db = labdb.LabDB()

    # only experiments where length and height in the videos table have been
    # defined should be processed
    sql = """SELECT ve.expt_id 
             FROM video as v INNER JOIN video_experiments AS ve ON v.video_id = ve.video_id
             WHERE height IS NOT NULL and length IS NOT NULL
               AND ve.expt_id >= 777
             LIMIT 5 
             """
    rows = db.execute(sql)

    for expt_id, in rows:
        expt_id_filename = workingdir + '%d.expt_id' % expt_id

        # does the .expt_id already exist?
        if expt_id_filename in outfiles:
            # all good, leave it alone
            outfiles.remove(expt_id_filename)
        else:
            f = open(expt_id_filename, 'wb')
            pickle.dump(expt_id, f)

    # clean up files from previous runs
    for f in outfiles:
        os.unlink(f)

@transform(forEachExperiment, suffix('.expt_id'), '.video_id')
def determineVideoId(infile, outfile):

    expt_id = pickle.load(open(infile))

    # select video_id given expt_id
    db = labdb.LabDB()
    sql = """SELECT v.video_id 
             FROM video as v INNER JOIN video_experiments AS ve ON v.video_id = ve.video_id
             WHERE ve.expt_id = %s""" % expt_id
    video_id, = db.execute_one(sql)

    pickle.dump(video_id, open(outfile, 'w'))

@transform(determineVideoId, suffix('.video_id'), '.videonc')
def createVideoNcFile(infile, outfile):
    video_id = pickle.load(open(infile))
    
    createncfilefromimages.compute_videoncfile(video_id)

    pickle.dump(video_id, open(outfile, 'w'))

@transform(createVideoNcFile, suffix('.videonc'), '.movieVideo')
def movieVideo(infile, outfile):
    video_id = pickle.load(open(infile))

    movieName = os.path.basename(outfile) # e.g 123.movieDz
    movieName = os.path.join(moviedir, movieName + '.mp4')

    # make the movie
    movieplayer.movie('video',  # var
                      video_id, # id of nc file
                      256,  # min_max value
                      saveFig=True,
                      movieName= movieName
                     )
    pickle.dump(movieName, open(outfile, 'w'))

   
@transform(forEachExperiment, suffix('.expt_id'), '.schlierenParameters')
def determineSchlierenParameters(infile, outfile):
    expt_id = pickle.load(open(infile))
   
    # sigma depends on background image line thickness
    p = {}
    p['sigma'] = 8 # constant for now
    p['filterSize'] = 400 # 190 pixel is about 10 cm in space.

    pickle.dump(p, open(outfile, 'w'))

@transform([determineVideoId, determineSchlierenParameters],
        suffix('.video_id'),
        add_inputs(r'\1.schlierenParameters'),
        '.dz_id')
def computeDz(infiles, outfile):
    videoIdFile = infiles[0]
    schlierenParametersFile = infiles[1]

    video_id = pickle.load(open(videoIdFile))
    p = pickle.load(open(schlierenParametersFile))

    dz_id = SyntheticSchlieren.compute_dz( 
            video_id,
            10, # minTol
            p['sigma'],
            p['filterSize'],
            #skip_row = 2, # number of rows to jump ... z
            skip_col = 2 , # number of columns to jump .. x
            startF = 0,        # startFrame
            stopF = 1500,         # stopFrame
                    # skipFrame
                    # diffFrame
            cache = cacheValue
            )

    pickle.dump(dz_id, open(outfile, 'w'))

@transform(computeDz, suffix('.dz_id'), '.Axi_id')
def computeAxi(infile, outfile):
    dz_id = pickle.load(open(infile))
    
    Axi_id = WaveCharacteristics.compute_a_xi(
            dz_id,
            cache=cacheValue,
            )

    pickle.dump(Axi_id, open(outfile, 'w'))

@transform(computeDz, suffix('.dz_id'), '.movieDz')
def movieDz(infile, outfile):
    dz_id = pickle.load(open(infile))

    movieName = os.path.basename(outfile) # e.g 123.movieDz
    # get time stamp
    t = time.ctime().rsplit()[3]
    movieName = os.path.join(moviedir, movieName+ t + '.mp4')
    
    # make the movie
    movieplayer.movie('dz',  # var
                      dz_id, # id of nc file
                      0.01,  # min_max value
                      saveFig=True,
                      movieName= movieName
                     )
    pickle.dump(movieName, open(outfile, 'w'))


@transform(computeAxi, suffix('.Axi_id'), '.fw_id')
def filterAxiLR(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    fw_id = Spectrum_LR.task_hilbert_func(
            Axi_id,
            0.02, #maxMin
            100, # plotColumn
            cache=cacheValue,
            )

    pickle.dump(fw_id, open(outfile, 'w'))

@transform(computeAxi, suffix('.Axi_id'), '.movieAxi')
def movieAxi(infile, outfile):
    Axi_id = pickle.load(open(infile))

    movieName = os.path.basename(outfile) # e.g 123.movieAxi
    movieName = os.path.join(moviedir, movieName + '.mp4')

    # make the movie
    movieplayer.movie('Axi',  # var
                      Axi_id, # id of nc file
                      0.02,  # min_max value
                      saveFig=True,
                      movieName= movieName
                     )
    pickle.dump(movieName, open(outfile, 'w'))


@transform(forEachExperiment, suffix('.expt_id'), '.exptParameters')
def getParameters(infile, outfile):
    expt_id = pickle.load(open(infile))

    params = {}

    # look up parameters for the given expt_id
    video_id, N, omega, kz, theta = Energy_flux.get_info(expt_id)
    theta,kz,wavelengthZ,kx,wavelengthX,c_gx,c_px=predictions_wave.predictionsWave(N,omega)
    params = { 'expt_id' : expt_id, 
               'N' : N,
               'omega (/s)' : omega,
               'theta (degree)' : theta,
               'kz (/cm)' : kz,
               'wavelength_z (cm)' : wavelengthZ,
               'kx (/cm)': kx,
               'wavelength_x (cm)' : wavelengthX,
               'group velocity (cm/s)' : c_gx,
               'phase velocity (cm/s)' : c_px,
               }
    
    pickle.dump(params, open(outfile, 'w'))

@merge(getParameters, tabledir + 'tableExperimentParameters.txt')
def tableExperimentParameters(infiles, outfile):

    if os.path.exists(outfile):
        os.unlink(outfile)

    f = open(outfile, 'w')

    for infile in infiles:
        params = pickle.load(open(infile))
        
        for key, value in params.iteritems():
            f.write(key + ":\t" + str(value) + "\n")
        f.write("\n")



@transform([computeAxi, filterAxiLR], suffix('.Axi_id'), '.plotEnergyFlux')
def plotEnergyFlux(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    Energy_flux.compute_energy_flux(
            Axi_id,
            300,  # rowStart
            400,  # rowEnd
            500,      # column
            plotname = plotName,
            )

    pickle.dump(plotName, open(outfile, 'w'))

@transform(computeAxi, suffix('.Axi_id'), '.plotAxiHorizontalTimeSeries')
def plotAxiHorizontalTimeSeries(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    axi_TS_row.compute_energy_flux(
            Axi_id,
            400,  # row number 
            0.02,      # maxmin
            plotname = plotName,
            )

    pickle.dump(plotName, open(outfile, 'w'))

@transform(computeAxi, suffix('.Axi_id'), '.plotAxiVerticalTimeSeries')
def plotAxiVerticalTimeSeries(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    axi_TS_col.compute_energy_flux(
            Axi_id,
            300,  # column number 
            0.02,      # maxmin
            plotname = plotName,
            )

    pickle.dump(plotName, open(outfile, 'w'))


@transform([filterAxiLR], suffix('.fw_id'), '.plotFilteredLR')
def plotFilteredLR(infile, outfile):
    fw_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    Spectrum_LR.plotFilteredLR(fw_id,
            plotName = plotName,
            )

    pickle.dump('outfile', open(outfile, 'w'))

if __name__ == "__main__":

    start_time = datetime.datetime.now()

    print "="*40
    print "  WORKFLOW START"

    print "="*40

    finalTasks = [
    #        movieVideo, 
            movieDz, 
            movieAxi,
            plotEnergyFlux, 
            plotFilteredLR,
            tableExperimentParameters,
            plotAxiHorizontalTimeSeries,
            plotAxiVerticalTimeSeries,
            ]

    forcedTasks = [
            forEachExperiment,
    #        computeDz,
    #        computeAxi,
            filterAxiLR,
            plotAxiHorizontalTimeSeries,
            plotAxiVerticalTimeSeries,
            ]

    pipeline_printout_graph( open('workflow.pdf', 'w'), 
        'pdf', 
        finalTasks,
        forcedtorun_tasks = forcedTasks,
        
        no_key_legend=True)

    pipeline_run(finalTasks,
            forcedTasks,
            verbose=2, 
        #    multiprocess=4, 
            one_second_per_job=True)

    stop_time = datetime.datetime.now()
    elapsed_time = stop_time - start_time

    print "="*40
    print "  WORKFLOW STOP"
    print
    print "    start_time:", start_time
    print "    stop_time:", stop_time
    print "    elapsed_time:", elapsed_time
    print "="*40
