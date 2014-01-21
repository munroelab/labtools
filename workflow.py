"""
This workflow processes all experiments

- produces plots of EnergyFlux vs Time for each experiment

"""

from ruffus import *
import sys
import os
import labdb
import pickle

import SyntheticSchlieren
import WaveCharacteristics
import Spectrum_LR
import Energy_flux
import movieplayer

workingdir = "workflow/"
moviedir = "movies/"
plotdir = "plots/"

@follows(mkdir(workingdir))
@follows(mkdir(moviedir))
@follows(mkdir(plotdir))
@split(None, workingdir + '*.expt_id')
def forEachExperiment(infiles, outfiles):
    
    #   clean up files from previous runs
    for f in outfiles:
        os.unlink(f)

    # select experiments
    db = labdb.LabDB()

    # only experiments where length and height in the videos table have been
    # defined should be processed
    sql = """SELECT ve.expt_id 
             FROM video as v INNER JOIN video_experiments AS ve ON v.video_id = ve.video_id
             WHERE height IS NOT NULL and length IS NOT NULL
               AND ve.expt_id >= 748
             LIMIT 2
             """
    rows = db.execute(sql)

    for expt_id, in rows:
        f = open(workingdir + '%d.expt_id' % expt_id, 'wb')
        pickle.dump(expt_id, f)

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
    
@transform(forEachExperiment, suffix('.expt_id'), '.schlierenParameters')
def determineSchlierenParameters(infile, outfile):
    expt_id = pickle.load(open(infile))
   
    # sigma depends on background image line thickness
    p = {}
    p['sigma'] = 8 # constant for now
    p['filterSize'] = 30

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
           startF = 100,        # startFrame
           stopF = 100+1300,         # stopFrame
                    # skipFrame
                    # diffFrame
            )

    pickle.dump(dz_id, open(outfile, 'w'))

@transform(computeDz, suffix('.dz_id'), '.Axi_id')
def computeAxi(infile, outfile):
    dz_id = pickle.load(open(infile))
    
    Axi_id = WaveCharacteristics.compute_a_xi(
            dz_id,
            cache=False,
            )

    pickle.dump(Axi_id, open(outfile, 'w'))

@transform(computeDz, suffix('.dz_id'), '.movieDz')
def movieDz(infile, outfile):
    dz_id = pickle.load(open(infile))

    movieName = os.path.basename(outfile) # e.g 123.movieDz
    movieName = os.path.join(moviedir, movieName + '.mp4')

    # make the movie
    movieplayer.movie('dz',  # var
                      dz_id, # id of nc file
                      0.001,  # min_max value
                      saveFig=True,
                      movieName= movieName
                     )
    pickle.dump(movieName, open(outfile, 'w'))

@transform(computeAxi, suffix('.Axi_id'), '.fw_id')
def filterAxiLR(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    fw_id = Spectrum_LR.task_hilbert_func(
            Axi_id,
            0.1, #maxMin
            100, # plotColumn
            cache=False,
            )

    pickle.dump(fw_id, open(outfile, 'w'))


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

    pickle.dump('outfile', open(outfile, 'w'))

@transform([filterAxiLR], suffix('.fw_id'), '.plotFilteredLR')
def plotFilteredLR(infile, outfile):
    fw_id = pickle.load(open(infile))
    
    plotName = os.path.basename(outfile) + '.pdf'
    plotName = os.path.join(plotdir, plotName)

    Spectrum_LR.plotFilteredLR(fw_id,
            plotName = plotName,
            )

    pickle.dump('outfile', open(outfile, 'w'))

finalTasks = [
        movieDz, 
        plotEnergyFlux, 
        plotFilteredLR,
        ]

pipeline_printout_graph( open('workflow.pdf', 'w'), 
    'pdf', 
    finalTasks,
    forcedtorun_tasks = [forEachExperiment],
    no_key_legend=True)

pipeline_printout(sys.stdout,
        finalTasks, 
        [forEachExperiment],
        )

pipeline_run(finalTasks, 
       [forEachExperiment], 
        verbose=2, 
    #    multiprocess=4, 
        one_second_per_job=False)

