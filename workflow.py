"""
This workflow processes all experiments

Each step in the workflow processes data from a particular experiment.
    Experiments are identified by experiment_id

"""

from ruffus import *
import sys
import os
import pickle
import datetime
import time
import collections
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
from labtools import plots

workingdir = "workflow/"
moviedir = "movies/"
plotdir = "plots/"
tabledir = "tables/"
cacheValue = True

@follows(mkdir(workingdir))
@follows(mkdir(moviedir))
@follows(mkdir(plotdir))
@follows(mkdir(tabledir))
@split(None, workingdir + '*.expt_id')
def forEachExperiment(infiles, outfiles):
    
    logger.info('Determining set of experiments')

    # select experiments
    db = labdb.LabDB()

    # only experiments where length and height in the videos table have been
    # defined should be processed
    sql = """SELECT ve.expt_id 
             FROM video as v INNER JOIN video_experiments AS ve ON v.video_id = ve.video_id
             WHERE height IS NOT NULL and length IS NOT NULL
               AND ve.expt_id IN (833)
             LIMIT 3
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

    # assumption: there is only one video per experiment
    # In general, one experiment would lead to multiple videos
    #   might need to change this step in a split

    pickle.dump(video_id, open(outfile, 'w'))

@transform(determineVideoId, suffix('.video_id'), '.videonc')
def createVideoNcFile(infile, outfile):

    # Videos need a associate time and space grid

    # also, we only want to consider a limited window in time and space

    # multiple such windows would require this step to be converted
    # into a split

    video_id = pickle.load(open(infile))

    # should this return something?
    # an nc_id containing the the videonc?
    # currently, the video_id is used as the id

    #createncfilefromimages.compute_videoncfile(video_id)

    pickle.dump(video_id, open(outfile, 'w'))

@transform(createVideoNcFile, suffix('.videonc'), '.movieVideo')
def movieVideo(infile, outfile):
    # given a videonc file, make a movie

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
    p['sigma'] = 11 # constant for now
    p['filterSize'] = 40 # 190 pixel is about 10 cm in space. # it is not used

    pickle.dump(p, open(outfile, 'w'))

@transform([determineVideoId, determineSchlierenParameters],
        suffix('.video_id'),
        add_inputs(r'\1.schlierenParameters'),
        '.dz_id')
def computeDz(infiles, outfile):

    logger.info('Performing Synthetic Schlieren')

    videoIdFile = infiles[0]
    schlierenParametersFile = infiles[1]

    video_id = pickle.load(open(videoIdFile))
    p = pickle.load(open(schlierenParametersFile))

    dz_id = SyntheticSchlieren.compute_dz( 
            video_id,
            7, # minTol
            p['sigma'],
            p['filterSize'],
            #skip_row = 2, # number of rows to jump ... z
            skip_col = 1 , # number of columns to jump .. x
            startF = 0,        # startFrame
            stopF = 10,         # stopFrame ..
            #set stopF=0 if you want it to consider all the frames
                    # skipFrame
            #diff_frames=None, # diffFrame set diff_frame to None if you want to compute deltaN2
            cache = cacheValue #set to False to recompute the nc file
            )

    pickle.dump(dz_id, open(outfile, 'w'))

@transform(computeDz, 
           formatter('.dz_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotDz.pdf')
def plotDz(infile, outfile):
    print "infile:", infile
    print "outfile:", outfile

    dz_id = pickle.load(open(infile))

    plots.plot_slice('dz',  # var
                dz_id, # id of nc file
                'vts', 300,
                maxmin = 0.05,  # min_max value
                plotName=outfile,
               )

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
                      0.5,  # min_max value
                      saveFig=True,
                      movieName= movieName
                     )
    pickle.dump(movieName, open(outfile, 'w'))

@transform(computeDz, suffix('.dz_id'), '.fw_id')
def filterLR(infile, outfile):
    dz_id = pickle.load(open(infile))

    fw_id = Spectrum_LR.task_DzHilbertTransform(dz_id, cache=cacheValue)

    pickle.dump(fw_id, open(outfile, 'w'))

@transform(filterLR, 
           formatter('.fw_id'), 
           [('{subpath[0][1]}/plots/{basename[0]}.LR.right.pdf','right'),
            ('{subpath[0][1]}/plots/{basename[0]}.LR.left.pdf', 'left'),]
           )
def plotLR(infile, outfile):
    nc_id = pickle.load(open(infile))

    for i in range(2):
        plots.plot_slice(outfile[i][1],  # var
                    nc_id, # id of nc file
                    'vts', 300,
                    maxmin = 0.05,  # min_max value
                    plotName=outfile[i][0],
                   )

@transform(filterLR, suffix('.fw_id'), '.Axi_id')
def computeAxi(infile, outfile):
    dz_id = pickle.load(open(infile))
    
    Axi_id = WaveCharacteristics.compute_a_xi(
            dz_id,
            cache=cacheValue,
            )

    pickle.dump(Axi_id, open(outfile, 'w'))


@transform(computeAxi, suffix('.Axi_id'), '.movieAxi')
def movieAxi(infile, outfile):
    Axi_id = pickle.load(open(infile))

    movieName = os.path.basename(outfile) # e.g 123.movieAxi
    movieName = os.path.join(moviedir, movieName + '.mp4')

    # make the movie
    movieplayer.movie('Axi',  # var
                      Axi_id, # id of nc file
                      2,  # min_max value
                      saveFig=True,
                      movieName= movieName
                     )
    pickle.dump(movieName, open(outfile, 'w'))


@transform(forEachExperiment, suffix('.expt_id'), '.exptParameters')
def getParameters(infile, outfile):
    expt_id = pickle.load(open(infile))

    # look up parameters for the given expt_id
    video_id, N, omega, kz, theta = Energy_flux.get_info(expt_id)
    theta,kz,wavelengthZ,kx,wavelengthX,c_gx,c_px=predictions_wave.predictionsWave(N,omega)
    wgAmplitude = Energy_flux.getwgamplitude(expt_id)
    params = collections.OrderedDict([ ('expt_id' , expt_id), 
                ('WG amplitude' , wgAmplitude),
               ('N' , N),
               ('omega (/s)' , omega),
               ('theta (degree)' , theta),
               ('kz (/cm)' , kz),
               ('wavelength_z (cm)' , wavelengthZ),
               ('kx (/cm)', kx),
               ('wavelength_x (cm)' , wavelengthX),
               ('group velocity (cm/s)' , c_gx),
               ('phase velocity (cm/s)' , c_px),
               ])
    
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


#@transform([computeAxi, filterAxiLR], suffix('.Axi_id'), '.plotEnergyFlux')
@transform([computeAxi], suffix('.Axi_id'), '.plotEnergyFlux')
def plotEnergyFlux(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    Energy_flux.compute_energy_flux_raw(
            Axi_id,
            400,  # rowStart
            500,  # rowEnd
            300,      # column
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
            700,  # row number
            1,      # maxmin
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
            600,  # column number
            4,      # maxmin
            plotname = plotName,
            )

    pickle.dump(plotName, open(outfile, 'w'))


@transform([filterLR], suffix('.fw_id'), '.plotFilteredLR')
def plotFilteredLR(infile, outfile):
    fw_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    #Spectrum_LR.plot_data(fw_id, #plotName = plotName,)

    #pickle.dump('outfile', open(outfile, 'w'))

if __name__ == "__main__":

    start_time = datetime.datetime.now()

    print "="*40
    print "  WORKFLOW START"

    print "="*40

    finalTasks = [
            plotLR,
            plotDz,
            #movieVideo,
    #         movieDz,
             #movieAxi,
    #         plotEnergyFlux, 
    #         plotFilteredLR,
    #         tableExperimentParameters,
    #        plotAxiHorizontalTimeSeries,
    #        plotAxiVerticalTimeSeries,
            ]

    forcedTasks = [
   #         plotDz,
            #plotLR,
            filterLR,
    #        forEachExperiment,
    #        determineSchlierenParameters,
    #         computeDz,
             #computeAxi,
    #        filterAxiLR,
    #        plotAxiHorizontalTimeSeries,
            #plotAxiVerticalTimeSeries,
    #        plotEnergyFlux, 
            #getParameters,
            #tableExperimentParameters,
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
