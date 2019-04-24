import pathlib
from pathlib import Path
import os.path
from os import path
import sys
from io import StringIO
from scipy.spatial import distance
import numpy as np
import IPython
from copy import deepcopy

def getRobotData(data, num_robots, overlap):
    nr=num_robots

    if data['n_frames'] % nr !=0:
        n_frames_padded = data['n_frames'] + nr - (data['n_frames']% nr)
    else:
        n_frames_padded = data['n_frames']

    assignments1 = np.reshape(np.arange(0,n_frames_padded,1),(nr,-1))
    assignments1[assignments1>=data['n_frames']]=0
    remapper=dict()
    for i in range(nr):
        remapper[i]=assignments1[i]
        if i > 0:
            remapper[i]=np.append(assignments1[i-1,-overlap:],remapper[i])
        if i < nr-1:
            remapper[i] = np.append(remapper[i],assignments1[i+1,:overlap])

    remapper[nr-1] = remapper[nr-1][remapper[nr-1]>0]

    robots=dict()
    ## Splitting into 10 robots
    for i in range(nr):
        robots[i]=deepcopy(data)
        assignments=remapper[i]
        robots[i]['T_W_C'] = [data['T_W_C'][j] for j in assignments]
        robots[i]['times'] = [data['times'][j] for j in assignments]
        robots[i]['netvlad'] = [data['netvlad'][:,j] for j in assignments]
        robots[i]['descs'] = [data['descs'][j] for j in assignments]
        robots[i]['wids3'] = [data['wids3'][j] for j in assignments]
        robots[i]['wids4'] = [data['wids4'][j] for j in assignments]
        robots[i]['gt_T_W_C'] = [data['gt_T_W_C'][j] for j in assignments]
        robots[i]['lms_in_frame'] = [data['lms_in_frame'][j] for j in assignments]

    for i in range(nr):
        observed_lms=[]
        n_frames=len(robots[i]['lms_in_frame'])
        for frame in range(n_frames):
            observed_lms=np.unique(np.append(observed_lms,list(robots[i]['lms_in_frame'][frame])))
        ##np.unique will already do the sorting


        lm_reindexing=np.zeros(robots[i]['p_W_lm'].size)
        lm_reindexing[np.array(observed_lms,dtype=np.uint64)]=np.arange(0,observed_lms.shape[0],1)
        robots[i]['p_W_lm'] = robots[i]['p_W_lm'][np.array(observed_lms,dtype=np.uint64),:]
        for frame in range(n_frames):
            robots[i]['lms_in_frame'][frame]=lm_reindexing[np.array(robots[i]['lms_in_frame'][frame],dtype=np.uint64)]

    return robots
