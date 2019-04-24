import os
import csv
from readDecentrStateFromOptG2ofiles import readDecentrStateFromOptG2oFiles
from writeG2oFile import writeDecentrStateToG2oFiles
import IPython
import pandas as pd
import numpy as np
import time
def runSyncGaussSeidal(update_state, distributed_mapper_location, group_idx, group_reindexing,  max_iters):
    outputDir = "dgs_data/{}".format(group_idx)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    # write decentr_state into g2o file
    writeDecentrStateToG2oFiles(update_state, outputDir, group_reindexing) ## recheck group indexing note from Shwarya
    # run the code in c++
    current_path = os.getcwd()
    os.system("{} --dataDir {}/{}/ --nrRobots {} --traceFile {}/{}/trace --maxIter {}".format(\
        distributed_mapper_location, current_path, outputDir, len(update_state), current_path, outputDir,\
        max_iters))

    print ("Obtained optimized .g2o files from DOPt!")
    
    # read data back from file
    update_state = readDecentrStateFromOptG2oFiles(outputDir, update_state, '_optimized')
    # get stats
    dgs_stats = dict()
    filename = "{}/trace".format(outputDir)
    trace_file = "{}_overall_error.txt".format(filename)
    if os.path.exists(trace_file):
        [dgs_stats['exchange_gs'], dgs_stats['num_iter'], dgs_stats['exchange_ddf'],\
         dgs_stats['comm_link_count']] = getDistGaussSeidelStats(outputDir)
    return update_state, dgs_stats


def getDistGaussSeidelStats(directory):
    # do the stotistics:
    filename = "{}".format(directory)
    trace_file = "{}/trace_overall_error.txt".format(filename)
    # read the trace file:
    trace = pd.read_csv(trace_file,delimiter=' ',header=None, error_bad_lines=False,quoting=csv.QUOTE_NONE,names=list(range(0,100)),sep='delimiter')
    end_index = np.where(trace.iloc[0]==-1)[0]
    rotation_iterations = end_index-1

    end_index = np.where(trace.iloc[1]==-1)[0]
    pose_iterations = end_index - 1

    # different error level
    centralized_error = trace.iloc[2,0]
    distributed_error = trace.iloc[3,0]

    # TODO: where do these numbers come?
    sizePose = 6*8
    sizeRotation = 9*8

    commLinks = 2 * trace.iloc[4,0]

    nrGaussSeidelIterations = pose_iterations + rotation_iterations
    informationExchangeGaussSeidel = pose_iterations*(commLinks*sizePose) + rotation_iterations*(commLinks*sizeRotation)

    # Compare the result with DOF-SAM Communication
    nrGNIterations = 1;  #  typically 3
    sizePoseDDFSAM = 6*8;   # Assumed to be 12 doubles, 1 double = 8 byte
    informationExchangeDDF = nrGNIterations*((commLinks*sizePoseDDFSAM) + (commLinks*sizePoseDDFSAM)**2)
    return [informationExchangeGaussSeidel, nrGaussSeidelIterations, informationExchangeDDF, commLinks]
