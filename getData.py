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
import json

#class structtype():
#    pass

#Get VO and NetVlad Data
def getData( dataset_path, sequence_id, data_type ):

    vo_data = {
        'n_frames': None,
        'T_W_C': None,
        'times': None,
        'p_W_lm': None,
        'n_lm': None,
        'lms_in_frame': None,
        'descs': None,
        'wids3': None,
        'wids4': None,
        'gt_T_W_C': None
    }

    dataset_path = Path(dataset_path)
    sequence_id_fname = sequence_id + '.txt'

    root = dataset_path / sequence_id   #kitti/00
    dpath = root / 'dslam'              #kitti/00/dslam

    #Files
    pose_file = dpath / 'poses_0.txt'   #kitti/00/dslam/poses_0.txt
    time_file = dpath / 'times_0.txt'
    lm_pos_file = dpath / 'lm_pos_0.txt'
    lm_obs_file = dpath / 'lm_obs_0.txt'
    desc_file = dpath / 'descs_0.txt'

    #Working with kitti dataset
    if data_type == 'kitti':
        gt_pose_file = dataset_path / 'poses' / sequence_id_fname   #kitti/poses/00.txt
        gt_times_file = root / 'times.txt'                          #kitti/00/times.txt

    assert(pose_file.is_file())
    assert(time_file.is_file())
    assert(lm_pos_file.is_file())
    assert(lm_obs_file.is_file())
    assert(desc_file.is_file())
    assert(gt_pose_file.is_file())
    assert(gt_times_file.is_file())

    T_W_C_raw = np.loadtxt(str(pose_file)) #(1684, 16)
    
    vo_data['n_frames'] = np.size(T_W_C_raw,0)

    vo_data['T_W_C'] = [ dict() for i in range(vo_data['n_frames'])]
    for i in range(vo_data['n_frames']):
        rowdata = T_W_C_raw[i,:]
        reshapedata = rowdata.reshape(4,4)
        vo_data['T_W_C'][i] = reshapedata

    vo_data['times'] = np.loadtxt(str(time_file))
    vo_data['p_W_lm'] = np.loadtxt(str(lm_pos_file))
    vo_data['n_lm']  = np.size(vo_data['p_W_lm'],0)

    vo_data['lms_in_frame'] = [ dict() for i in range(vo_data['n_frames'])]
    vo_data['decs'] = [ dict() for i in range(vo_data['n_frames'])]
    vo_data['wids3'] = [ dict() for i in range(vo_data['n_frames'])]
    vo_data['wids4'] = [ dict() for i in range(vo_data['n_frames'])]

    vo_data_descs = [ dict() for i in range(vo_data['n_frames'])]

    lm_obs = np.loadtxt(str(lm_obs_file))                #(1392027, 2)
    descs_and_wids = np.loadtxt(str(desc_file))
    descs_and_wids_col_no = np.size(descs_and_wids,1)
    descs = descs_and_wids[:,0:descs_and_wids_col_no-3] #(1392027, 31)
    wids3 = descs_and_wids[:,descs_and_wids_col_no-2] #1392027x1
    wids4 = descs_and_wids[:,descs_and_wids_col_no-1] #1392027x1

    for i in range(vo_data['n_frames']):
        vo_data['lms_in_frame'][i] = lm_obs[lm_obs[:,0] == i, 1] ## indices are correct now
        #IPython.embed()
        #vo_data['descs'][i] = descs[lm_obs[:,0]==i,:]
        vo_data_descs[i] = descs[lm_obs[:,0]==i,:]
        vo_data['wids3'][i] = wids3[lm_obs[:,0]==i]
        vo_data['wids4'][i] = wids4[lm_obs[:,0]==i]
    vo_data['descs'] = vo_data_descs

    #Get ground truth poses
    if data_type == 'kitti':
        gt_times = np.loadtxt(str(gt_times_file))
        gt_T_W_C_raw = np.loadtxt(str(gt_pose_file))

    gt_times_reshaped = gt_times.reshape(-1,1)     #gt_times 4541x1
    vo_data_times_reshaped = vo_data['times'].reshape(-1,1)     #vo_data['times'] 1684
    dist = distance.cdist(gt_times_reshaped, vo_data_times_reshaped, "sqeuclidean")
    dist_index = np.argmin(dist,0) # !! Doesn't give exact indices like Matlab

    gt_T_W_C_raw = gt_T_W_C_raw[dist_index] #1684x12
    vo_data['gt_T_W_C'] = [ dict() for i in range(vo_data['n_frames'])]

    for i in range(vo_data['n_frames']):
        vo_data['gt_T_W_C'][i] = [(np.transpose(gt_T_W_C_raw[i,:].reshape(4,3))),(0,0,0,1)]

    feat_file = dpath / 'kitti_netvlad.json' #kitti/00/dslam/netvlad_feats.bin
    netvlad_time_file = root / 'times.txt' ##kitti/00/times.loadtxt

    assert(feat_file.is_file())
    assert(netvlad_time_file.is_file())

    netvlad_dim = 4096


    ## Load NetVLAD descriptors from file
    netvlad_feats = []
    netvlad_file = open(str(feat_file),'r')
    for line in netvlad_file.readlines():
        data = json.loads(line)
        netvlad = data['descriptor']
        netvlad_feats.append(netvlad)

    if data_type == 'kitti':
        netvlad_times = np.loadtxt(str(netvlad_time_file))

    netvlad_times_reshaped = netvlad_times.reshape(-1,1)
    vo_data_times_reshaped = vo_data['times'].reshape(-1,1)     #vo_data['times'] 1684
    netvlad_dist = distance.cdist(netvlad_times_reshaped, vo_data_times_reshaped, "cityblock")
    netvlad_index = np.argmin(netvlad_dist,0) # it's perfect

    #Change this !
    net=np.asarray(netvlad_feats[netvlad_index[i]][0]).reshape(-1,1)

    for i in range(1,netvlad_index.shape[0]):
        net = np.concatenate((net, np.asarray(netvlad_feats[netvlad_index[i]][0]).reshape(-1,1)),1)

    #feat_file = dpath / 'netvlad_feats.txt'
    #net = np.genfromtxt(str(feat_file),delimiter=',')
    vo_data["netvlad"] = net # 4096 x 1684


    return vo_data
