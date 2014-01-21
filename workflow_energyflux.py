"""
This workflow produces plots of EnergyFlux vs Time for each experiment`

"""

from ruffus import *
import sys
import labdb
import pickle

import SyntheticSchlieren
import WaveCharacteristics
import Spectrum_LR
import Energy_flux

workingdir = "test_workflow/"

@follows(mkdir(workingdir))
@split(None, workingdir + '*.expt_id')
def forEachExperiment(infiles, outfiles):
    # select experiments
    #db = labdb.LabDB()

    #rows = db.execute('SELECT expt_id FROM experiments WHERE expt_id IN [821]')
    expt_id_list = [821,822,823,824]

    for expt_id in expt_id_list:
        f = open(workingdir + '%d.expt_id' % expt_id, 'wb')
        pickle.dump(expt_id, f)

@transform(forEachExperiment, suffix('.expt_id'), '.video_id')
def determineVideoId(infile, outfile):

    expt_id = pickle.load(open(infile))

    # select video_id given expt_id
    db = labdb.LabDB()
    video_id, = db.execute_one('SELECT video_id FROM video WHERE expt_id = %s' % expt_id)

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
           stopF = 100+1200,         # stopFrame
                    # skipFrame
                    # diffFrame
            )

    pickle.dump(dz_id, open(outfile, 'w'))

@transform(computeDz, suffix('.dz_id'), '.Axi_id')
def computeAxi(infile, outfile):
    dz_id = pickle.load(open(infile))
    
    Axi_id = WaveCharacteristics.compute_a_xi(dz_id)

    pickle.dump(Axi_id, open(outfile, 'w'))

@transform(computeAxi, suffix('.Axi_id'), '.fw_id')
def filterAxiLR(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    fw_id = Spectrum_LR.task_hilbert_func(
            Axi_id,
            0.1, #maxMin
            100, # plotColumn
            cache=True,
            )

    pickle.dump(fw_id, open(outfile, 'w'))


@transform([computeAxi, filterAxiLR], suffix('.Axi_id'), '.plotEnergyFlux')
def plotEnergyFlux(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    Energy_flux.compute_energy_flux(
            Axi_id,
            300,  # rowStart
            400,  # rowEnd
            500,      # column
            plotname = outfile,
            )

    pickle.dump('outfile', open(outfile, 'w'))

@transform([filterAxiLR], suffix('.fw_id'), '.plotFilteredLR')
def plotFilteredLR(infile, outfile):
    fw_id = pickle.load(open(infile))
    
    Spectrum_LR.plotFilteredLR(fw_id)

    pickle.dump('outfile', open(outfile, 'w'))

finalTask = [plotFilteredLR, plotEnergyFlux]
pipeline_printout_graph( open('workflow.pdf', 'w'), 
    'pdf', 
    finalTask,
    forcedtorun_tasks = [forEachExperiment],
    no_key_legend=True)

pipeline_printout(sys.stdout,
        finalTask, 
        [forEachExperiment],
        )

pipeline_run(finalTask, 
       [forEachExperiment], 
        verbose=2, 
    #    multiprocess=4, 
        one_second_per_job=False)

