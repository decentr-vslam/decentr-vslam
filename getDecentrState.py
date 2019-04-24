import pathlib
from pathlib import Path
import os.path
from os import path
import sys
from io import StringIO
from scipy.spatial import distance
import numpy as np
from numpy.linalg import matrix_power
import IPython
from copy import deepcopy
from sklearn.cluster import KMeans

def initDecentrState(num_robots,robots):

    #init decentr state
    decentr_state =[ dict() for i in range(num_robots)]

    for i in range(num_robots):
        decentr_state[i]['Sim_W_O'] = np.identity(4)
        decentr_state[i]['Sim_O_C'] = []
        decentr_state[i]['Sim_Cprev_C'] = []
        decentr_state[i]['original_T_O_C'] = []

        decentr_state[i]['times'] = []
        netvlad_dim = 128
        decentr_state[i]['netvlad'] = np.empty((netvlad_dim,0))

        decentr_state[i]['descs'] = []
        decentr_state[i]['wids3'] = []
        decentr_state[i]['wids4'] = []

        decentr_state[i]['lms_in_frame'] = []
        decentr_state[i]['gt_T_W_C'] = []
        decentr_state[i]['T_gt_O_ate'] = robots[i]['T_W_C'][1]

        #DVPR related members
        decentr_state[i]['dvpr_queries_netvlad'] = []
        decentr_state[i]['dvpr_queries_robot_i'] = []
        decentr_state[i]['dvpr_queries_frame_i'] = []

        #Robust RelPose
        decentr_state[i]['consistent_groups'] = [ dict() for i in range(num_robots)]

        for j in range(num_robots):
            decentr_state[i]['consistent_groups'][j]['members']=dict()
            decentr_state[i]['consistent_groups'][j]['floating']= True

        decentr_state[i]['place_matches'] = dict()
        decentr_state[i]['grouped_with'] = np.empty((1,0))
        decentr_state[i]['converged'] = True

        #Pre-loading landmarks to make more manageable
        T_O_W = matrix_power(robots[i]['T_W_C'][0] , -1)
        temp1 = T_O_W[0:3,0:3]
        temp2 = robots[i]['p_W_lm'].transpose()
        temp3 = T_O_W[0:3,3]
        temp3 = temp3.reshape(3,1)
        decentr_state[i]['p_O_lm'] = np.add((np.matmul(temp1,temp2)),temp3)
        #IPython.embed()

    return decentr_state

def assignCC(filepath,decentr_state,params):
    print("Assigning Cluster Centers")

    relevant_params = dict()
    relevant_params['netvlad_dim']= params['netvlad_dim']
    relevant_params['num_robots']= params['num_robots']
    relevant_params['clusters_per_robot']= params['clusters_per_robot']

    #location of train_feats
    filepath = Path(filepath)
    trainfeats_file = filepath / 'train_feats.txt'   #robotcar_netvlad_feats/train_feats.txt
    assert(trainfeats_file.is_file())
    train_feats = np.loadtxt(trainfeats_file) # shape (256, 27311)
    #print(train_feats.shape)
    train_feats = train_feats[0:relevant_params['netvlad_dim'],:] # shape (128, 27311)

    clusters_filename = 'clustercenters' + '_' + str(relevant_params['netvlad_dim'])\
    + '_' + str(relevant_params['num_robots']) + '_'\
    + str(relevant_params['clusters_per_robot']) +'.txt'
    clusters_file = filepath / clusters_filename

    #prevent clustering each time.
    if clusters_file.is_file():
        kmeanscc =  np.genfromtxt(clusters_file,delimiter=',')
    else:
        kmeans = KMeans(n_clusters= relevant_params['num_robots']*relevant_params['clusters_per_robot'],\
        max_iter = 1000000, random_state=0,init = 'k-means++',n_jobs = -1).fit(train_feats.transpose())
        #print(kmeans.labels_)
        #print(kmeans.cluster_centers_)
        #print(kmeans.cluster_centers_.shape) #numrobots*clustersperrobot, netvlad dim - (10,128)
        #print(kmeans.inertia_)
        #print(kmeans.n_iter_)
        kmeanscc = kmeans.cluster_centers_
        np.savetxt(clusters_file,kmeanscc)

    assert(relevant_params['num_robots']==len(decentr_state))
    for i in range(len(decentr_state)):
        decentr_state[i]['cluster_center']= kmeanscc[i,:].transpose()
    return decentr_state
