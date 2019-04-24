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
import pickle
import time
import os
from getData import getData
from getRobotData import getRobotData
from getIndivStream import getIndivStream

from getDecentrState import initDecentrState
from getDecentrState import assignCC

from stepDecentrState import *
from Plot.plotting_funcs import *
from Plot.plotDataOverTime import plotDataOverTime

from static_tools import evalAccuracy

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#Config
regen_data = 0
regen_robots = 0
regen_stream = 0

dataset_path = 'kitti'
sequence_id = '00'
data_type = 'kitti'

parpool_size = 4

num_robots = 10
overlap = 3

robcar_path = 'robotcar_netvlad_feats'
robcar_path = Path(robcar_path)
dataset_path = Path(dataset_path)

root = dataset_path / sequence_id   #kitti/00
dpath = root / 'dslam'              #kitti/00/dslam

out_file = dpath / 'full_data.mat'   #kitti/00/dslam/full_data.mat
out_file = dpath / 'full_data.txt'   #kitti/00/dslam/full_data.txt
#start = time.time()
if regen_data==1:
    print("Getting Data")
    data = getData(dataset_path, sequence_id, data_type)
    save_obj(data,'data')
if regen_data==0:
    print("Loading Data")
    data = load_obj('data')
#end = time.time()
#print(end-start)

if regen_robots==1:
    print("Getting Robot Data")
    robots = getRobotData(data, num_robots, overlap)
    save_obj(robots,'robots')
if regen_robots==0:
    print("Loading Robot Data")
    robots = load_obj('robots')

if regen_stream==1:
    print("Getting Decentralized Stream Data")
    decentr_stream = getIndivStream(robots)
    save_obj(decentr_stream,'decentr_stream')
if regen_stream==0:
    print("Loading Decentralized Stream Data")
    decentr_stream = load_obj('decentr_stream')

#Setup params dict
params = dict()
params["num_robots"] = num_robots
params["netvlad_dim"] = 128
params["clusters_per_robot"] = 1
params["use_dvpr"] = 1
params["min_dist_geover"] = 10 # min dist between geo verifications for a given pair of num_robots
params["robust_relpose_min_group_size"] = 1 # robust rel pose rejects outliers in rel pose detection
params["robust_relpose_consistency_range"] = 20
params["robust_relpose_position_tolerance"] = 4
params["opt_max_iters"] = 20
params["use_tardioli"] = 1
# for parameter studies
params["run_i"] = 1
distributed_mapper_location=os.getcwd()+'/distributed-mapper/cpp/build/runDistributedMapper'

#run sim
decentr_state = initDecentrState(num_robots,robots)
decentr_state = assignCC(robcar_path,decentr_state,params)
real_time = time.time()
netvlad_match_stats = dict()
data_increment = dict()
dgs_stats=[]
last_accur_eval = 5
opt_future = initOptHandle(num_robots)
times_list=[]
colours=[]
for i in range(num_robots):
    colours.append(generate_new_color(colours,pastel_factor = 0.9))

# evalAccuracy(decentr_state, opt_future, real_time)

for step_j in range(len(decentr_stream)):
    #if step_j > 10:
    #    evalAccuracy(decentr_state, opt_future, real_time)
    end_time = time.time()
    print ('On step #' + str(step_j))
    times_list.append(decentr_stream[step_j]["times"])
    if (end_time - real_time < decentr_stream[step_j]["times"]):
        time.sleep(decentr_stream[step_j]["times"] - end_time + real_time)    
    [decentr_state, data_increment[step_j], netvlad_match_stats[step_j], opt_future, dgs_stats_i] = stepDS(decentr_state, decentr_stream[step_j],params,distributed_mapper_location,opt_future)
    #IPython.embed()

    dgs_stats.append(dgs_stats_i)

    if step_j % params["num_robots"] == 0 and step_j!=0:
        pass
        #plotDState(decentr_state,colours)
#plotDataOverTime(data_increment, times_list)
    
print("End of Program")
