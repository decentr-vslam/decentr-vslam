from matrix_tool import rpyxyzToT
import functools
import numpy as np 
from scipy.optimize import least_squares as lsq


def evalAccuracy(decentr_state, opt_handle, time):
    # accuracy is a dict
    accuracy = dict()
    accuracy['time'] = time
    n_robots = len(decentr_state)

    matched_group_assignment = np.array(list(range(n_robots)))

    for i in range(n_robots):
        if decentr_state[i]['grouped_with']:
            matched_group_assignment[i] = min(decentr_state[i]['grouped_with'].min(), i)
        else:
            matched_group_assignment[i] = i
    
    groups = np.unique(matched_group_assignment)

    accuracy['matched_groups'] = []

    # counting matched frames
    for i in range(len(groups)):
        _matched_groups = dict()
        group_index = groups[i]
        _matched_groups['members'] = []
        for _i, _m in enumerate(matched_group_assignment):
            if _m == group_index:
                _matched_groups['members'].append(_i)
        
        _matched_groups['num_frames'] = 0

        for member_i in _matched_groups['members']:
            _matched_groups['num_frames'] += len(decentr_state[member_i]['Sim_O_C'])
        
        accuracy['matched_groups'].append(_matched_groups)
    
    accuracy['optimized_groups'] = opt_handle['optimized_groups']

    for group_i in range(len(accuracy['optimized_groups'])):
        members = accuracy['optimized_groups'][group_i]['members']
        [accuracy['optimized_groups'][group_i]['ATE'], decentr_state] = getConnectedAte(decentr_state, members)
    
    return [accuracy, decentr_state]
    

def getConnectedAte(decentr_state, group):
    p_gt_C = np.empty([0, 3])
    for g in group:
        if decentr_state[g]['gt_T_W_C']:
            # TODO: something wrong here
            for gt_t_w_c in decentr_state[g]['gt_T_W_C']:
                p_gt_C = np.vstack([p_gt_C, gt_t_w_c[0][0:3, 3:4].T])
    
    p_gW_C = np.empty([0, 3])

    for g in group:
        if decentr_state[g]['Sim_W_O'].size != 0:
            for sim_o_c in decentr_state[g]['Sim_O_C']:
                T_gW_C = np.multiply(decentr_state[g]['Sim_W_O'], sim_o_c)
                p_gW_C = np.vstack([p_gW_C, T_gW_C[0:3, 3:4].T])
    
    if p_gW_C.size != 0:
        [decentr_state[int(group.min())]['T_gt_ate'], connected_ate] = alignTrajs(p_gt_C.T, p_gW_C.T, decentr_state[int(min(group))]['T_gt_ate'])
    
    num_frames = p_gW_C.shape[0]
    
    return [num_frames, connected_ate, decentr_state]
        

def applyT(init_T, estimate_state):
    dT = rpyxyzToT(np.vstack[np.arctan2(estimate_state[0:3])/4, estimate_state[3:6]])
    T_gt_O = np.multiply(dT, init_T)
    return T_gt_O


def alignError(init_T, p_O_C, estim_state):
    T_gt_O = applyT(init_T, estim_state)
    p_gt_C_estim = np.multiply(T_gt_O[0:3, 0:3], p_O_C) + T_gt_O[0:3, 3]
    alignerror = (p_gt_C_estim.T - p_gt_C_estim.T).reshape([-1, 1])
    return alignerror


def alignTrajs(p_gt_C, p_O_C, initial_T_gt_O):
    state = np.hstack([np.zeros(1, 3), initial_T_gt_O[0:3, 3:4].T]).T

    bound_applyT = functools.partial(applyT, initial_T_gt_O)
    bound_alignE = functools.partial(alignError, initial_T_gt_O, p_O_C)
    optim_state = lsq(bound_alignE, state)

    errs = alignError(initial_T_gt_O, p_O_C, optim_state).reshape([3, -1])
    ate = np.sqrt(np.sum(np.sqaure(errs), axis=0)).mean()
    optim_T_gt_O = applyT(initial_T_gt_O, optim_state)
    return [optim_T_gt_O, ate]




    