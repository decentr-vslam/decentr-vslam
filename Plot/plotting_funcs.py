import numpy as np
import matplotlib.pyplot as plt
from simulateDV import tInv
import IPython
import random

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def plotDState(decentr_state,colours,plot_gt=True):

    N=len(decentr_state)
    robot_xz=dict()
    for i in range(N):
        Sim_W_C = []
        robot_xz[i]=[]

        for j in range(len(decentr_state[i]["Sim_O_C"])):
            temp=decentr_state[i]["Sim_O_C"]
            Sim_W_C.append(np.dot(decentr_state[i]["Sim_W_O"],temp[j]))

        for j in range(len(Sim_W_C)):
            temp=Sim_W_C[j]
            robot_xz[i].append([temp[0,3], temp[2,3]])

        robot_xz[i] = np.asarray(robot_xz[i])
    robot_xz_gt = dict()
    plt.ion()
    for i in range(N):
        gt_T_W_C = []
        robot_xz_gt[i] = []
        
        for j in range(len(decentr_state[i]["gt_T_W_C"])):
            temp=decentr_state[i]["gt_T_W_C"][j]
            temp=np.concatenate((temp[0],np.array(temp[1]).reshape(1,-1)),0)
            gt_T_W_C.append(np.dot(tInv(decentr_state[0]["T_gt_O_ate"]) , temp))


        for j in range(len(gt_T_W_C)):
            temp=gt_T_W_C[j]
            
            robot_xz_gt[i].append([temp[0,3], temp[2,3]])
        robot_xz_gt[i] = np.asarray(robot_xz_gt[i])
    if plot_gt:
        for i in range(len(robot_xz_gt)):
            if robot_xz_gt[i].size==0:
                continue
            plt.plot(robot_xz_gt[i][:,0], robot_xz_gt[i][:,1], '-', 'r',5)
            plt.draw()


    for i in range(N):
        matches = decentr_state[i]["place_matches"]
        for match_i in list(matches.keys()):
            if len(matches[match_i]) > 0:
                match = matches[match_i]
                robo1 = robot_xz[i][match_i,:]
                robo2 = robot_xz[match["robot_i"]][match["frame_i"],:]
                plt.plot([robo1[0], robo2[0]], [robo1[1], robo2[1]], '-', 'k',10)
    for i in range(len(robot_xz)): ##entire trajectory
        if robot_xz[i].size == 0:
            continue
        plt.plot(robot_xz[i][:,0], robot_xz[i][:,1],'o', c=colours[i])
    plt.pause(0.01)
    #plt.clf()
