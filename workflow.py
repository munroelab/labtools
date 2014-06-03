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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from labtools import SyntheticSchlieren
from labtools import WaveCharacteristics
from labtools import Spectrum_LR
from labtools import Energy_flux
from labtools import Spectrum_Analysis
from labtools import movieplayer
from labtools import axi_TS_row
from labtools import axi_TS_col
from labtools import createncfilefromimages
from labtools import labdb
from labtools import predictions_wave
from labtools import plots
from labtools import stratification_plot
from labtools import plotting_functions

# EDIT list of expt_id's to process:
# 764,760,759,758,757,766,763,761,762,779
expt_ids = [579,583,584,586,587,589,750,751,752,753,754,755,756,757,758,759,760,761,762,764,768,778,779,817,818,819,821,822,823,824,825]

#Done expt_ids = [754,764,760]
## expt_ids = [579,583,584,586,587,589]

params = {579 : { 'rowS': 100, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          583 : { 'rowS': 100, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          584 : { 'rowS': 100, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 900, },
          586 : { 'rowS': 100, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          587 : { 'rowS': 100, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          589 : { 'rowS': 100, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },

          750 : { 'rowS': 100, 'rowE' : 830, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          751 : { 'rowS': 100, 'rowE' : 830, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          752 : { 'rowS': 100, 'rowE' : 830, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 900, },
          753 : { 'rowS': 260, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          754 : { 'rowS': 260, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          755 : { 'rowS': 260, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 800, },
          756 : { 'rowS': 260, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 900, },
          757 : { 'rowS': 250, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          758 : { 'rowS': 300, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          759 : { 'rowS': 300, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          760 : { 'rowS': 330, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          761 : { 'rowS': 300, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          762 : { 'rowS': 300, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },

          763 : { 'rowS': 250, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          764 : { 'rowS': 250, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          766 : { 'rowS': 280, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          768 : { 'rowS': 280, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          778 : { 'rowS': 230, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          779 : { 'rowS': 230, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          780 : { 'rowS': 230, 'rowE' : 860, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },

          817 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          818 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          819 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          821 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          822 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          823 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          824 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },
          825 : { 'rowS': 250, 'rowE' : 900, 'colS' : 60, 'colE': 1260, 'startF_NR' : 100,'startF_WR' : 700, },



        }


workingdir = "workflow/"
moviedir = "movies/"
plotdir = "plots/"
tabledir = "tables/"
cacheValue = True

@follows(mkdir(workingdir))
@follows(mkdir(moviedir))
@follows(mkdir(plotdir))
@follows(mkdir(tabledir))

@originate( [ workingdir + "%d.expt_id" % i for i in expt_ids] )
def startExperiment(expt_id_filename):
    logging.info("Start Experiment!")
    expt_id = int(os.path.splitext(os.path.basename(expt_id_filename))[0])
    f = open(expt_id_filename, 'wb')
    pickle.dump(expt_id, f)


@transform(startExperiment, suffix('.expt_id'), '.video_id')
def determineVideoId(infile, outfile):
    logging.info("Determine Video id...")
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

@transform(startExperiment, suffix('.expt_id'), '.stratParams')
def determineStratParams(infile, outfile):
    logging.info("Determine stratification parameters & create dictionary 'strat'.")
    expt_id = pickle.load(open(infile))

    # select strat_id and zoffset given expt_id
    db = labdb.LabDB()
    sql = """ SELECT stratification.zoffset, stratification.strat_id \
    FROM stratification INNER JOIN stratification_experiments \
    WHERE stratification_experiments.expt_id = %d AND \
    stratification.strat_id = stratification_experiments.strat_id """ % expt_id
    rows = db.execute(sql)
    z_offset = rows[0][0]
    strat_id = rows[0][1]
    print "z_OFF : ",  z_offset
    print "strat_id : " , strat_id

    # assumption: there is only one stratification per experiment
    strat = collections.OrderedDict([ ('strat_id' , strat_id),
                ('z_offset' , z_offset),
                ])

    pickle.dump(strat, open(outfile, 'w'))

@transform(determineVideoId, suffix('.video_id'), '.videonc')
def createVideoNcFile(infile, outfile):
    logging.info("Creating an nc file of the raw images.")
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
    logging.info(" Making a movie of the raw images")
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

   
@transform(startExperiment, suffix('.expt_id'), '.schlierenParameters')
def determineSchlierenParameters(infile, outfile):
    logging.info("Inside determineSchlierenParameters ... determine sigma and filterSize before computing schlieren")
    expt_id = pickle.load(open(infile))
   
    # sigma depends on background image line thickness
    p = {}

    if expt_id in (753,579,583,584,586,587,589,):
        p['sigma'] = 5
    elif expt_id in (817,818,819,):
        p['sigma'] = 6
    elif expt_id in (754,755,756,757,758,759,760,761,762,778,779):
        p['sigma'] = 10
    elif expt_id in (763,764,766,768):
        p['sigma'] = 11
    elif expt_id in (751,752,821,822,823,824,825,):
        p['sigma'] = 8
    else:
        p['sigma'] = 10

    p['filterSize'] = 40 # 190 pixel is about 10 cm in space. # it is not used

    pickle.dump(p, open(outfile, 'w'))

@jobs_limit(1)
@transform([determineVideoId, determineSchlierenParameters],
        suffix('.video_id'),
        add_inputs(r'\1.schlierenParameters'),
        '.dz_id')
def computeDz(infiles, outfile):

    logger.info('Performing Synthetic Schlieren..')

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
            #skip_col = 2 , # number of columns to jump .. x
            #skip_frames= 2, # number of columns to jump .. t
            #startF = 0,        # startFrame
            #stopF = 1113,         # stopFrame ..
            #set stopF=0 if you want it to consider all the frames
                    # skipFrame
            #diff_frames=None, # diffFrame set diff_frame to None if you want to compute deltaN2
            cache = cacheValue #cacheValue #set to False to recompute the nc file
            )

    pickle.dump(dz_id, open(outfile, 'w'))

@transform(computeDz, 
           formatter('.dz_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotDz.pdf')
def plotDz(infile, outfile):
    logging.info("Inside plotDz function. Plotting vertical time series along a column in dz.nc file")
    print "infile:", infile
    print "outfile:", outfile

    dz_id = pickle.load(open(infile))

    plots.plot_slice('dz',  # var
                dz_id, # id of nc file
                'vts', 300,
                maxmin = 0.02,  # min_max value
                plotName=outfile,
               )


@transform(computeDz, 
           formatter('.dz_id'),
           '{subpath[0][1]}/movies/{basename[0]}.dN2t.mp4')
def movieDz(infile, outfile):
    logging.info("Making a movie of dz.nc file.")
    dz_id = pickle.load(open(infile))

    # make the movie
    movieplayer.movie('dz',  # var
                      dz_id, # id of nc file
                      0.02,  # min_max value
                      saveFig=True,
                      movieName= outfile
                     )

@subdivide(computeDz, formatter(),
           "{path[0]}/{basename[0]}.*.fw_id",
           "{path[0]}/{basename[0]}",
           "{basename[0]}"
           )
def filterLR_window(infile,outfiles, outfile_root,expt_id):
    logging.info("Inside filterLR_window. Computing HT of dz within a small window in time and each with a separate fw_id.")
    dz_id = pickle.load(open(infile))
    expt_id = int(expt_id)

    # get num_frames
    num_frames = Spectrum_LR.get_num_frames(expt_id)
    print num_frames
    for oo in outfiles:
        os.unlink(oo)

    for n, startF in enumerate(range(1, num_frames, 100)):
        fw_id = Spectrum_LR.task_DzHilbertTransform(dz_id,
                                                    cache=cacheValue,
                                                    rowS = params[expt_id]['rowS'],
                                                    rowE = params[expt_id]['rowE'],
                                                    colS = params[expt_id]['colS'],
                                                    colE = params[expt_id]['colE'],
                                                    startF=startF, # startF is the frame where to start the window from
                                                    period=2 # length of the window in time
                                                    )
        if (fw_id is None):
            break

        print "fw_id",fw_id

        for plotcol in [100, 1000]:
            outfile = "%s.%d.%d.fw_id" % (outfile_root, n, plotcol)
            pickle.dump(fw_id, open(outfile, 'w'))

@subdivide(filterLR_window,
           regex(r"(.*workflow/)(\d+).(\d+).(\d+).fw_id$"),
           [r'\1\2.\3.\4.fw_id_EF_VTS',
            r'\1\2.\3.\4.fw_id_VAEF'],
           r'\4',
        )
def computeEnergyFlux(infile, outfiles, plotcol):

    plotcol = int(plotcol)

    logging.info("Computing the energy flux values for every fw_id and storing them in disk as 2 separate files.")
    print infile
    fw_id = pickle.load(open(infile))

    print "& outfile:",outfiles
    print "FW_ID #",fw_id

    raw_EF,left_EF,right_EF,t,z,x,raw_VAEF,left_VAEF,right_VAEF= Spectrum_LR.compute_vertically_averaged_energy_flux(fw_id, plotcol=plotcol)

    pickle.dump((raw_EF,left_EF,right_EF,t,x,z), open(outfiles[0], 'w'))
    pickle.dump((raw_VAEF,left_VAEF,right_VAEF,t), open(outfiles[1], 'w'))


@collate(computeEnergyFlux,
         formatter(),
         ['{subpath[0][1]}/plots/{basename[0]}.VTS_WavesRawLeftRight_dn2tEF.pdf',
            '{subpath[0][1]}/plots/{basename[0]}.Plot_WavesRawLeftRight_dn2tVAEF.pdf'])
def energyFluxPlots(infiles, outfiles):
    logging.info('Inside EnergyFluxPlots. Making the energy flux plots for every fw_id created')
    logging.info(infiles)
    print infiles
    raw_EF,left_EF,right_EF,t,x,z = pickle.load(open(infiles[0]))
    raw_VAEF,left_VAEF,right_VAEF,t = pickle.load(open(infiles[1]))
    #print raw_EF.shape,t.shape,raw_VAEF.shape
    Spectrum_LR.plot_energy_flux_VTS(raw_EF,left_EF,right_EF,t,z,0.05,outfiles[0])
    plotting_functions.sharexy_overlay3plots(raw_VAEF,left_VAEF,right_VAEF,t,0.7,'Raw','Left','Right','Time','Energy flux',outfiles[1])


@collate(computeEnergyFlux,
         regex(r"(.*workflow/)(\d+).(\d+).(\d+).fw_id_VAEF$"),
         r'\1\2.\4.VAEF.csv',
         )
def mergeEnergyFlux(infiles, outfile):
    logging.info("Merge all the VAEF files in the workflow to create the plot consisting of all fw_id average energy flux values")

    # sort by part number increasing numerically
    infiles = sorted(infiles, key=lambda s: int(os.path.basename(s).split('.')[1]) )

    logging.info(infiles)
    logging.info(outfile)

    f = open(outfile, 'w')
    f.write('t, raw, left, right\n')
    for infile in infiles:
        raw_VAEF,left_VAEF,right_VAEF,t = pickle.load(open(infile))
        print len(raw_VAEF),len(left_VAEF),len(right_VAEF)
        a,b,c,d = Spectrum_LR.compute_mergeEF(raw_VAEF,left_VAEF,right_VAEF,t)

        f.write( "%f, %f, %f, %f\n" % (d, a,b,c))
    f.close()


@transform(mergeEnergyFlux,
           formatter(),
           '{subpath[0][1]}/plots/{basename[0]}.finalEF.pdf')
def plotMergeEnergyFlux( infile, outfile):
    logging.info("Plotting final merged EF plot !!")
    data = np.loadtxt(infile, skiprows=1, delimiter=',')

    t = data[:,0]
    raw = data[:,1]
    left = data[:,2]
    right = data[:,3]

    plt.clf()

    plt.plot(t, raw, 'o-', label = 'Raw')
    plt.plot(t, right, 'o-', label = 'Right')
    plt.plot(t, left, 'o-', label = 'Left')

    plt.legend()
    plt.xlabel('t (s)')
    plt.ylabel('Energy Flux')

    plt.savefig(outfile)




@transform(filterLR_window,formatter('fw_id'),
            '{subpath[0][1]}/plots/{basename[0]}.VTS_PlotWavesRawLeftRight_dn2t.pdf')
def plotWavesVTSWindow(infile,outfile):

    print infile
    fw_id = pickle.load(open(infile))
    print "&",outfile
    print len(outfile)
    ###", outfile


    print "####",fw_id
    Spectrum_LR.filtered_waves_VTS(fw_id,
            600,  # column number
            .04,      # maxmin
            plotName = outfile,
            )

@jobs_limit(1)
@transform(computeDz, suffix('.dz_id'), '.fw_id')
def filterLR(infile, outfile):
    dz_id = pickle.load(open(infile))

    fw_id = Spectrum_LR.task_DzHilbertTransform(dz_id, cacheValue,rowS=320,rowE = 860,colS=60,colE=1260)
    print "fw_id",fw_id
    pickle.dump(fw_id, open(outfile, 'w'))

@transform(filterLR_window,
           formatter('.fw_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.VTS_PlotWavesRawLeftRight_dn2t.pdf')
def plotWavesVerticalTimeSeries(infile, outfile):
    fw_id = pickle.load(open(infile))

    Spectrum_LR.filtered_waves_VTS(
            fw_id,
            600,  # column number
            .04,      # maxmin
            plotName = outfile,
            )

@subdivide(filterLR_window,
           formatter('.fw_id'),
           ['{subpath[0][1]}/plots/{basename[0]}.LR.right.pdf',
            '{subpath[0][1]}/plots/{basename[0]}.LR.left.pdf'],
           ['right', 'left']
           )
def plotLR(infile, outfiles, extra_arg):
    logging.info('plotLR called')
    nc_id = pickle.load(open(infile))

    for i in range(len(outfiles)):
        plots.plot_slice(extra_arg[i],  # var
                    nc_id, # id of nc file
                    'vts', 300,
                    maxmin = 0.02,  # min_max value
                    plotName=outfiles[i],
                   )


@transform(computeDz, suffix('.dz_id'), '.Axi_id')
def computeAxi(infile, outfile):
    dz_id = pickle.load(open(infile))
    
    Axi_id = WaveCharacteristics.compute_a_xi(
            dz_id,
            cache=cacheValue,
            )

    pickle.dump(Axi_id, open(outfile, 'w'))


@transform(computeAxi, 
           formatter(),
           '{subpath[0][1]}/movies/{basename[0]}.Axi.mp4')
def movieAxi(infile, outfile):
    Axi_id = pickle.load(open(infile))

    # make the movie
    movieplayer.movie('Axi',  # var
                      Axi_id, # id of nc file
                      4,  # min_max value
                      saveFig=True,
                      movieName= outfile
                     )


@transform(startExperiment, suffix('.expt_id'), '.exptParameters')
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

@transform(determineStratParams, 
           formatter('.stratParams'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotStratification.pdf')
def plotStratification(infile, outfile):

    logging.info("Creating stratification plot.")

    s = pickle.load(open(infile))
    plt.clf()
    stratification_plot.plot_stratification(s['strat_id'],s['z_offset'],plot_name = outfile)

@merge(getParameters, tabledir + 'tableExperimentParameters.txt')
def tableExperimentParameters(infiles, outfile):

    logging.info("Creating a table of experimental parameters.")

    if os.path.exists(outfile):
        os.unlink(outfile)

    f = open(outfile, 'w')

    for infile in infiles:
        params = pickle.load(open(infile))
        
        for key, value in params.iteritems():
            f.write(key + ":\t" + str(value) + "\n")
        f.write("\n")


@transform([computeAxi],
           formatter('.Axi_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotEnergyFlux_axi.pdf')
def plotEnergyFlux(infile, outfile):

    logging.info("This function computes energy flux from the vertical displacement amplitude rather than from dz (OLD)")

    Axi_id = pickle.load(open(infile))
    
    Energy_flux.compute_energy_flux_raw(
            Axi_id,
            350,  # rowStart
            900,  # rowEnd
            600,      # column
            plotname = outfile,
            )

@transform(computeAxi, 
           formatter('.Axi_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotAxiHorizontalTimeSeries.pdf')
def plotAxiHorizontalTimeSeries(infile, outfile):

    logging.info("Plotting Axi horizontal time series.")

    Axi_id = pickle.load(open(infile))
    
    axi_TS_row.compute_energy_flux(
            Axi_id,
            700,  # row number
            4,      # maxmin
            plotname = outfile,
            )

@transform(computeAxi, 
       formatter('.Axi_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotAxiHorizontalTimeSeries.pdf')
def plotAxiVerticalTimeSeries(infile, outfile):
    Axi_id = pickle.load(open(infile))
    
    axi_TS_col.compute_energy_flux(
            Axi_id,
            600,  # column number
            4,      # maxmin
            plotname = outfile,
            )

@transform([filterLR], suffix('.fw_id'), '.FFT_raw')
def FFT_raw(infile, outfile):
    fw_id = pickle.load(open(infile))

    #timeS,timeE,tstep,rowS,rowE,zstep,colS,colE,xstep, max_kx, max_kz,max_omega = Spectrum_Analysis.xzt_fft(fw_id,rowS=250,rowE=800)

    ncfile = '/data/filtered_waves/%d/waves.nc' % fw_id
    ncvar = 'raw_array'
    result = Spectrum_Analysis.xzt_fft(ncfile, ncvar)

    pickle.dump(result, open(outfile, 'w'))

@transform([filterLR], suffix('.fw_id'), '.FFT_left')
def FFT_left(infile, outfile):
    fw_id = pickle.load(open(infile))

    ncfile = '/data/filtered_waves/%d/waves.nc' % fw_id
    ncvar = 'left_array'
    result = Spectrum_Analysis.xzt_fft(ncfile, ncvar)

    pickle.dump(result, open(outfile, 'w'))

@transform([filterLR], suffix('.fw_id'), '.FFT_right')
def FFT_right(infile, outfile):
    fw_id = pickle.load(open(infile))

    ncfile = '/data/filtered_waves/%d/waves.nc' % fw_id
    ncvar = 'right_array'
    result = Spectrum_Analysis.xzt_fft(ncfile, ncvar)

    pickle.dump(result, open(outfile, 'w'))

@collate([FFT_raw, FFT_left, FFT_right],
         formatter(),
         [  '{subpath[0][1]}/plots/{basename[0]}.3dFFT_max_kx.pdf',
            '{subpath[0][1]}/plots/{basename[0]}.3dFFT_max_kz.pdf',
            '{subpath[0][1]}/plots/{basename[0]}.3dFFT_max_omega.pdf',
         ])
def FFT_plots(infiles, outfiles):
    logging.info('called FFT_plots')

    print infiles
    kx,kz,omega,l_max_kx, l_max_kz,l_max_omega = pickle.load(open(infiles[0]))
    kx,kz,omega,raw_max_kx, raw_max_kz,raw_max_omega = pickle.load(open(infiles[1]))
    kx,kz,omega,r_max_kx, r_max_kz,r_max_omega = pickle.load(open(infiles[2]))

    Spectrum_Analysis.plot_3Dfft_dominant_frequency(kz,omega,raw_max_kx,l_max_kx,r_max_kx,"kz","omega","K_X", outfiles[0])
    Spectrum_Analysis.plot_3Dfft_dominant_frequency(kx,omega,raw_max_kz,l_max_kz,r_max_kz,"kx","omega","K_Z", outfiles[1])
    Spectrum_Analysis.plot_3Dfft_dominant_frequency(kx,kz,raw_max_omega,l_max_omega,r_max_omega,"kx","kz","OMEGA", outfiles[2])


@transform([filterLR],
           formatter('.fw_id'), 
           '{subpath[0][1]}/plots/{basename[0]}.plotFilterdLR.pdf')
def plotFilteredLR(infile, outfile):
    fw_id = pickle.load(open(infile))
    
    Spectrum_LR.plot_data(fw_id, plotName = outfile)

if __name__ == "__main__":

    start_time = datetime.datetime.now()

    print "="*40
    print "*"*12,"WORKFLOW START","*"*12

    print "="*40

    finalTasks1 = [
                filterLR_window,
                ]

    finalTasks2 = [
                tableExperimentParameters,
                plotStratification,
                plotDz,
                #plotAxiHorizontalTimeSeries,
                #plotAxiVerticalTimeSeries,
                filterLR_window,
                #computeEnergyFlux,
                energyFluxPlots,
                mergeEnergyFlux,
                plotMergeEnergyFlux,
                #filterLR_window,
                #computeEnergyFlux,
                #plotWavesVTSWindow,
                #FFT_plots,
                #plotLR_NR,
                #plotLR_WR,
                #movieVideo,
                #movieDz,
                #movieAxi,
                #plotEnergyFlux,
                #plotFilteredLR,
                #plotWavesVerticalTimeSeries_NR,
                #plotWavesVerticalTimeSeries_WR,
                #plotWavesVerticalTimeSeries,
                ]

    forcedTasks = [
                #plotStratification,
                #energyFluxPlots,
                #filterLR_WR,
                #startExperiment,
                #determineSchlierenParameters,
                #computeDz,
                #computeAxi,
                #getParameters,
                #tableExperimentParameters,
                #FFT_left,
                #FFT_raw,
                #FFT_right,
                #plotStratification,
                #plotWavesVerticalTimeSeries,
                #plotDz,
                #plotLR,
                #plotFilteredLR,
                #plotAxiHorizontalTimeSeries,
                #plotAxiVerticalTimeSeries,
                #plotEnergyFlux,
                ]

    pipeline_printout_graph( open('workflow_%s.pdf' % datetime.datetime.now().isoformat(), 'w'),
        'pdf',
        finalTasks1,
       forcedtorun_tasks = forcedTasks,
       no_key_legend=True)


    pipeline_run(finalTasks1,
            forcedTasks,
            verbose=5,
          #multiprocess=4,
            one_second_per_job=True)

    pipeline_printout_graph( open('workflow_%s.pdf' % datetime.datetime.now().isoformat(), 'w'),
        'pdf',
        finalTasks2,
       forcedtorun_tasks = forcedTasks,
       no_key_legend=True)

    pipeline_run(finalTasks2,
            forcedTasks,
            verbose=5,
          #multiprocess=4,
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
