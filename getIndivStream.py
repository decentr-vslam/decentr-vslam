import numpy as np
from copy import deepcopy
import IPython
def getIndivStream(robots):
    individual = dict()

    #Individual streams
    for robot_i in range(len(robots)):
        vo_data = robots[robot_i]
        individual[robot_i] = dict()
        for frame in range(len(robots[robot_i]["T_W_C"])):

            individual[robot_i][frame] = dict()

            individual[robot_i][frame]["robot_i"]=robot_i;
            if frame==0:
                individual[robot_i][frame]['Sim_Cprev_C'] = np.eye(4)
            else:
                individual[robot_i][frame]['Sim_Cprev_C'] = np.dot(np.linalg.inv(vo_data['T_W_C'][frame-1]),vo_data['T_W_C'][frame])

            individual[robot_i][frame]['original_T_O_C'] = np.dot(np.linalg.inv(vo_data['T_W_C'][0]), vo_data['T_W_C'][frame])

            individual[robot_i][frame]["times"] = vo_data["times"][frame] - vo_data["times"][0]
            individual[robot_i][frame]["netvlad"] = vo_data["netvlad"][frame]

            individual[robot_i][frame]["descs"] = vo_data["descs"][frame]
            individual[robot_i][frame]["wids3"] = vo_data["wids3"][frame]
            individual[robot_i][frame]["wids4"] = vo_data["wids4"][frame]
            individual[robot_i][frame]["lms_in_frame"] = vo_data["lms_in_frame"][frame]
            individual[robot_i][frame]["gt_T_W_C"] = vo_data["gt_T_W_C"][frame]

    spliced = []
    for robot_i in range (len(individual)):
        for frame_i in range (len(individual[robot_i])):
            spliced.append(individual[robot_i][frame_i])

    # Recheck time values?
    spliced_time = []
    for i in range (len(spliced)):
        spliced_time.append(spliced[i]["times"])

    #sort in ascending order and get indices
    sort_index = sorted(range(len(spliced_time)),key=spliced_time.__getitem__)

    decentr_stream =[]
    for i in range (len(sort_index)):
        decentr_stream.append(spliced[sort_index[i]])

    return decentr_stream
