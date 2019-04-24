import numpy as np
from scipy.spatial import distance
import IPython

from simulateDV import *
from decentralized_optimization import *

def fixR(R):
    if (np.abs(np.linalg.det(R)) - 1) > 1e-5:
        assert(False)
    [U,S,Vt] = np.linalg.svd(R)
    V=Vt.T
    S=np.diag(S)
    corr = np.zeros((3,3))
    for i in range(3):
        corr += np.dot(vec(V[:,i]),vec(V[:,i]).T) /  S[i,i]
    return np.dot(R,corr)

def info(*args):
  sequentialTypes = [dict, list, tuple] 
  for var in args:
    t=type(var)
    if t== np.ndarray:  
      return type(var),var.dtype, var.shape
    elif t in sequentialTypes: 
      return type(var), len(var)
    else:
      return type(var)
      
def stepDS(decentr_state, stream_data, params, distributed_mapper_location, opt_future):
    dgs_stats = dict()

    robot_i = stream_data["robot_i"]
    if decentr_state[robot_i]["Sim_O_C"] == []:
        decentr_state[robot_i]["Sim_O_C"].append(np.eye(4))
    else:
        decentr_state[robot_i]["Sim_O_C"].append(np.dot(decentr_state[robot_i]["Sim_O_C"][-1], stream_data["Sim_Cprev_C"]))
        decentr_state[robot_i]["Sim_O_C"][-1][0:3,0:3] = fixR(decentr_state[robot_i]["Sim_O_C"][-1][0:3,0:3])
        assert((np.linalg.det(decentr_state[robot_i]["Sim_O_C"][-1][0:3,0:3]) - 1) < 1e-5)
        decentr_state[robot_i]["Sim_Cprev_C"].append(stream_data["Sim_Cprev_C"])


    decentr_state[robot_i]["original_T_O_C"].append(stream_data["original_T_O_C"])
    decentr_state[robot_i]["times"].append(stream_data["times"])
    decentr_state[robot_i]["netvlad"] = np.concatenate((decentr_state[robot_i]["netvlad"],vec(stream_data["netvlad"])[:128]),1)

    decentr_state[robot_i]["descs"].append(stream_data["descs"])
    decentr_state[robot_i]["wids3"].append(stream_data["wids3"])
    decentr_state[robot_i]["wids4"].append(stream_data["wids4"])
    decentr_state[robot_i]["lms_in_frame"].append(stream_data["lms_in_frame"])

    decentr_state[robot_i]["gt_T_W_C"].append(stream_data["gt_T_W_C"])
    [decentr_state, dvpr_increment, gv_increment, netvlad_match_stats] = DVPR(decentr_state,robot_i,params)
    opti=True
    if opti:
        [decentr_state, opt_future, dgs_stats, opt_increment] = manageAsyncGaussSeidel(decentr_state, opt_future, distributed_mapper_location, False, params["opt_max_iters"], True)
    else:
        opt_increment=np.zeros([len(decentr_state),len(decentr_state)])
    for i in range(len(dgs_stats)):
        dgs_stats[i]["end_time"] = stream_data["times"]
    data_increment = np.stack((opt_increment,dvpr_increment,gv_increment))
    return decentr_state, data_increment, netvlad_match_stats, opt_future, dgs_stats

def initOptHandle(N):
    optHandle=dict()
    optHandle["optimized_groups"]=dict()
    for i in range(N):
        optHandle["optimized_groups"][i]=dict()
        optHandle["optimized_groups"][i]["members"]=[i]
        
    optHandle["running"] = False
    return optHandle
        
    
def DVPR(decentr_state, robot_i, params):
    
    N = len(decentr_state)
    dvpr_increment = np.zeros((N,N))
    gv_increment = np.zeros((N,N))
    query_netvlad = decentr_state[robot_i]["netvlad"][:,-1] ##last element of the list
    query_info = info(query_netvlad)
    query_frame_i = len(decentr_state[robot_i]["Sim_O_C"])-1


    time_dist = 30
    candidates = (decentr_state[robot_i]["times"] < (decentr_state[robot_i]["times"][-1] - time_dist))
    if np.sum(candidates)>0:
        mask=np.tile(vec(candidates),decentr_state[robot_i]["netvlad"].shape[0]).T
        remapped_candidates=np.ma.array(decentr_state[robot_i]["netvlad"], mask=np.logical_not(mask))
        distances = distance.cdist(remapped_candidates.T,query_netvlad[np.newaxis, :],'sqeuclidean',1)
        distances = np.ma.array(distances,mask=np.logical_not(candidates))
        best_frame = np.ma.argmin(distances,0)
        if (distances[int(best_frame)] < 0.01):
            print ("Simulating Decentralized Verification")
            [decentr_state, _] = simulateDV(decentr_state, robot_i, best_frame, robot_i, query_frame_i, params)

    if params["use_dvpr"]:
        cluster_centers = np.vstack([decentr_state[i]["cluster_center"] for i in range(len(decentr_state))])
        distances = distance.cdist(cluster_centers, query_netvlad[np.newaxis, :128], 'sqeuclidean', 1)
        cluster_robot = int(np.argmin(distances, 0))

        if len(decentr_state[cluster_robot]["dvpr_queries_netvlad"])>0:
            existing_entries = np.vstack([decentr_state[cluster_robot]["dvpr_queries_netvlad"][i] for i in range(len(decentr_state[cluster_robot]["dvpr_queries_netvlad"]))])
            distances = distance.cdist(existing_entries, query_netvlad[np.newaxis,:], 'sqeuclidean', 1)
            min_dist = distances.min(0)
            min_dist_index = np.argmin(distances, 0)
            match_robot_i = decentr_state[cluster_robot]["dvpr_queries_robot_i"][int(min_dist_index)]
            match_frame_i = decentr_state[cluster_robot]["dvpr_queries_frame_i"][int(min_dist_index)]

            if (match_robot_i == robot_i):
                min_dist = np.inf
                match_robot_i = 0
                match_frame_i = 0
        else:
            min_dist = np.inf
            match_robot_i = 0
            match_frame_i = 0

        decentr_state[cluster_robot]["dvpr_queries_netvlad"].append(query_netvlad)
        decentr_state[cluster_robot]["dvpr_queries_robot_i"].append(robot_i)
        decentr_state[cluster_robot]["dvpr_queries_frame_i"].append(query_frame_i)
        if cluster_robot != robot_i:
            dvpr_increment[robot_i, cluster_robot] += query_netvlad.nbytes + 5
            dvpr_increment[cluster_robot, robot_i] += 13


    else:
        distances = np.full((N,1), np.inf)
        print("distance: ", distances)
        best_frames = np.full((N,1), np.inf)
        print("best_frames: ", best_frames)
        for queried_robot in range(N):
            if queried_robot == robot_i:
                continue
            dvpr_increment[robot_i, queried_robot] += query_netvlad.nbytes
            if len(decentr_state[queried_robot]["netvlad"]) > 0:
                dist = distance.cdist(decentr_state[queried_robot]["netvlad"], query_netvlad[:,np.newaxis].T, 'sqeuclidean', 1)
                distances[queried_robot] = dist.min(0)
                best_frames[queried_robot] = np.argmin(dist, 0)
            dvpr_increment[queried_robot, robot_i] += 8

        min_dist = min(distances)
        match_robot_i = np.argmin(distances)
        match_frame_i = best_frames[match_robot_i]

    if params["min_dist_geover"]>0:
        matched_frames = list(decentr_state[robot_i]["place_matches"].keys())

        if len(decentr_state[robot_i]["place_matches"])>0:
            same_pair=[]
            for i in range(len(matched_frames)):
                fr = matched_frames[i]
                if decentr_state[robot_i]["place_matches"][fr]["robot_i"] == match_robot_i:
                    same_pair.append(i)
            if len(same_pair)>0:
                recent = matched_frames[same_pair[-1]]
                A=decentr_state[robot_i]["Sim_O_C"][-1][:3,3]
                B=decentr_state[robot_i]["Sim_O_C"][recent][:3,3]
                if np.linalg.norm(A-B) < params["min_dist_geover"]:
                    netvlad_match_stats = np.zeros([1,4])
                    print("Discarding place recognition near previous match!")
                    return decentr_state, dvpr_increment, gv_increment, netvlad_match_stats

    if min_dist < 0.01:
        decentr_state, gv_increment = simulateDV(decentr_state, match_robot_i, match_frame_i, robot_i, query_frame_i, params)
        netvlad_match_stats = np.array([[robot_i, query_frame_i, match_robot_i, match_frame_i]])
    else:
        netvlad_match_stats = np.zeros([1,4])

    return decentr_state, dvpr_increment, gv_increment, netvlad_match_stats


        
