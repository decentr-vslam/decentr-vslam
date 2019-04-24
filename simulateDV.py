import numpy as np
import os.path
import time
import scipy.stats as st
from copy import deepcopy
from shutil import copyfile

import IPython

def vec(v):
    return v.reshape(-1,1)
    
def simulateDV(decentr_state, match_robot_i, match_frame_i, query_robot_i, query_frame_i, params):

    print ("Making verification request for relative pose between robot" + str(match_robot_i) + " and robot " + str(query_robot_i))

    N=len(decentr_state)
    gv_increment = np.zeros((N,N))

    verification_request = makeVerificationRequest(decentr_state, match_robot_i, match_frame_i, query_robot_i, query_frame_i, params)

    if params["use_tardioli"]:
        kp_description = 2
    else:
        kp_description = 32

    n_query_kp = np.count_nonzero(verification_request[:,2]==0)
    gv_increment[query_robot_i, match_robot_i] = 9 + n_query_kp * (kp_description + 12)


    temp_lock_file = 'temp_lock.txt'
    fid = open(temp_lock_file,'w')
    np.savetxt('temp_request.txt',verification_request,fmt = '%.10f', delimiter=' ' )
    fid.close()
    os.remove(temp_lock_file)

    while not os.path.isfile('temp_result.txt'):
        time.sleep(0.001)
    while os.path.isfile('temp_lock.txt'):
        time.sleep(0.001)

    verification_result = np.genfromtxt('temp_result.txt') ## find out the format of this file

    gv_increment[match_robot_i, query_robot_i] = 2*6*8

    if verification_result[4]==0:

        return decentr_state, gv_increment

    Sim_M_Q = verification_result[5:].reshape(4,4).T
    print (Sim_M_Q, "is the transformation found!")
    
    match=dict()
    match["robot_i"] = match_robot_i
    match["frame_i"] = match_frame_i
    match["Sim_M_Q"] = Sim_M_Q

    match_accepted = True

    if params["robust_relpose_min_group_size"] > 1:
        [match_accepted, decentr_state[query_robot_i]["consistent_groups"][match["robot_i"]]] = acceptMatchIC(match, query_frame_i, query_robot_i, decentr_state, params)

    if match_accepted==True:
        decentr_state[query_robot_i]["place_matches"][query_frame_i]= match
           
        if [match_frame_i not in list(decentr_state[match_robot_i]["place_matches"].keys())]:
            decentr_state[match_robot_i]["place_matches"][match_frame_i] = dict()
        decentr_state[match_robot_i]["place_matches"][match_frame_i]["robot_i"] = query_robot_i
        decentr_state[match_robot_i]["place_matches"][match_frame_i]["frame_i"] = query_frame_i
        
        decentr_state[match_robot_i]["place_matches"][match_frame_i]["Sim_M_Q"] = np.linalg.inv(Sim_M_Q)
        decentr_state[match_robot_i]["converged"]=False
        decentr_state[query_robot_i]["converged"]=False

        match_group = np.append(decentr_state[match_robot_i]["grouped_with"], match_robot_i)
        query_group = np.append(decentr_state[query_robot_i]["grouped_with"], query_robot_i)

        if not np.any(match_group==query_robot_i):
            for i in range(match_group.shape[0]):
                idx1= int(match_group[i])
                decentr_state[idx1]["grouped_with"] = np.unique(np.append(decentr_state[idx1]["grouped_with"],query_group))
                
            for i in range(query_group.shape[0]):
                idx2 = int(query_group[i])
                decentr_state[idx2]["grouped_with"] = np.unique(np.append(decentr_state[idx2]["grouped_with"],match_group))
                
                
        assert(not (np.sum(decentr_state[query_robot_i]["grouped_with"] == query_robot_i)))
        assert(not (np.sum(decentr_state[match_robot_i]["grouped_with"] == match_robot_i)))
    return decentr_state, gv_increment

def makeVerificationRequest(decentr_state,match_robot_i,match_frame_i,query_robot_i, query_frame_i, params):

    verification_request =[]
    original_T_C_O = np.linalg.inv(decentr_state[query_robot_i]['original_T_O_C'][query_frame_i])
    p_O_lm = decentr_state[query_robot_i]['p_O_lm']
    p_O_lm = p_O_lm[:,np.array(decentr_state[query_robot_i]['lms_in_frame'][query_frame_i],dtype=np.uint64)]
    p_C_lm = np.transpose(np.dot(original_T_C_O[0:3, 0:3], p_O_lm) + vec(original_T_C_O[0:3, 3]))

    if params['use_tardioli']:
        descs = decentr_state[query_robot_i]['wids4'][query_frame_i];
    else:
        descs = decentr_state[query_robot_i]['descs'][query_frame_i];

    num_kp = np.size(descs, 0); #should be size 826

    temp_repmat  = np.tile([np.array([query_robot_i,query_frame_i,0])], (num_kp, 1)) #size num_kp,3 ie 826,3
    if verification_request == []:
        verification_request = np.concatenate(
            (temp_repmat, vec(np.array(descs, dtype=np.uint64)), p_C_lm), 1)

    original_T_C_O = np.linalg.inv(decentr_state[match_robot_i]['original_T_O_C'][match_frame_i])
    p_O_lm = decentr_state[match_robot_i]['p_O_lm']

    p_O_lm = p_O_lm[:,np.array(decentr_state[match_robot_i]['lms_in_frame'][match_frame_i],dtype=np.uint64)]
    p_C_lm = np.transpose(np.dot(original_T_C_O[0:3, 0:3], p_O_lm) + vec(original_T_C_O[0:3, 3]))

    if params['use_tardioli']:
        descs = decentr_state[match_robot_i]['wids4'][match_frame_i];
    else:
        descs = decentr_state[match_robot_i]['descs'][match_frame_i];

    num_kp = np.size(descs, 0);

    temp_repmat  = np.tile([np.array([match_robot_i,match_frame_i,1])], (num_kp, 1)) #size num_kp,3 ie 826,3
    verification_request = np.concatenate((verification_request, np.concatenate((temp_repmat, vec(np.array(descs,dtype=np.uint64)),p_C_lm),1)),0)

    return verification_request


def acceptMatchIC(match, query_frame_i, query_robot_i, decentr_state, params):
    this = dict()
    this["frame_i"] = query_frame_i
    this["match"] = match

    robot_state = decentr_state[query_robot_i]
    consistent_group = robot_state["consistent_groups"][match["robot_i"]]

    bool_list=[]

    for i in range(len(consistent_group["members"])):
        bool_list.append( areFramesWithinRange(robot_state, consistent_group["members"][i]["frame_i"], query_frame_i, params["robust_relpose_consistency_range"]))


    for i in range(len(consistent_group["members"])):
        if bool_list[i]!=True:
            consistent_group["members"].pop(i)
    if np.all(np.array(bool_list)==0):
        consistent_group["floating"]=True

    cons_list=[]

    for i in range(len(consistent_group["members"])):
        print ("Consistent matches")
                
        cons_list.append( areMatchesConsistent(decentr_state, query_robot_i, consistent_group["members"][i], this, params["robust_relpose_position_tolerance"]) )

    if np.all(np.array(cons_list)):
        consistent_group["members"][len(consistent_group["members"])] = this
    elif consistent_group["floating"]:
        consistent_group["members"] = {this}
        match_accepted = False
        return match_accepted, consistent_group
    else:
        match_accepted = False
        return match_accepted, consistent_group

    if consistent_group["floating"] == False:
        match_accepted = True
    else:
        if len(consistent_group["members"]) > params["robust_relpose_min_group_size"]:
            consistent_group["floating"] = False
            match_accepted = True
        else:
            match_accepted = False

    return match_accepted, consistent_group

def areFramesWithinRange(robot_data, frame_a, frame_b, prange):
    pose_a = robot_data["Sim_O_C"][frame_a]
    pos_a = pose_a[0:3,3]
    pose_b = robot_data["Sim_O_C"][frame_b]
    pos_b = pose_b[0:3,3]

    result = np.linalg.norm(pos_a - pos_b) < prange

    ## is a boolean like it should be

    return result

def tInv(T):
    temp1=np.concatenate((T[0:3,0:3].T, -np.dot(T[0:3,0:3],T[0:3,3]).reshape(-1,1)),1)
    temp2=np.concatenate((np.zeros((1,3)),np.ones((1,1))),1)
    return np.concatenate((temp1,temp2),0)
def areMatchesConsistent(all_robot_data, query_robot, match_a, match_b, position_tolerance):

    T_M1_Q1 = match_a["match"]["Sim_M_Q"]
    T_M2_Q2 = match_b["match"]["Sim_M_Q"]

    T_O_Q1 = all_robot_data[query_robot]["Sim_O_C"][match_a["frame_i"]]
    T_O_Q2 = all_robot_data[query_robot]["Sim_O_C"][match_b["frame_i"]]

    T_Q1_Q2 = np.dot(tInv(T_O_Q1),T_O_Q2) ## check this later

    match_robot = match_a["match"]["robot_i"]

    assert(match_robot == match_b["match"]["robot_i"])

    T_O_M1 = all_robot_data[match_robot]["Sim_O_C"][match_a["match"]["frame_i"]]
    T_O_M2 = all_robot_data[match_robot]["Sim_O_C"][match_b["match"]["frame_i"]]

    T_M1_M2 = np.dot(tInv(T_O_M1) , T_O_M2)

    T_M1_Q2_1 = np.dot(T_M1_Q1, T_Q1_Q2)
    T_M1_Q2_2 = np.dot(T_M1_M2, T_M2_Q2)

    error = np.dot(tInv(T_M1_Q2_1), T_M1_Q2_2)

    result = np.linalg.norm(error[0:3,3]) < position_tolerance
    ## returns a boolenan

    return result
