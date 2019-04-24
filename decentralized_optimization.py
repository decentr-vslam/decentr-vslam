import os
import numpy as np
import multiprocessing
from GaussianSeidal import runSyncGaussSeidal
from ProcessCheck import countBetweenRobotLinks, consistentlyApplyState, setConvergedWhereApplicable


def manageAsyncGaussSeidel(decentr_state, opt_handle, distributed_mapper_location, wait_finished, max_iters, renew):
    n_robots = len(decentr_state)  # each robot has a different state
    dgs_stats = list()  # each robot has a different stats
    opt_increment = np.zeros([n_robots, n_robots])  # should be a np.array

    # renew : restart cluster:
    if renew:
        # start with full group_assigment
        opt_handle['group_assignment'] = np.array([n for n in range(n_robots)])
        for i in range(n_robots):
            # TODO: what is grouped_with
            # assert all to be list
            expand_group_i = np.hstack([decentr_state[i]['grouped_with'].reshape(1,-1), np.array([i]).reshape([1, 1])])
            opt_handle['group_assignment'][i] = expand_group_i.min()
        # IPython.embed()
        opt_groups = np.unique(opt_handle['group_assignment'])
        opt_handle["futures"] = []

        for group_i in opt_groups:
            group_mask = (opt_handle['group_assignment'] == group_i)
            group_size = group_mask.sum()

            if group_size > 1:
                # TODO: what is converged
                idxs = np.where(group_mask)[0]
                members_converged=[]
                for i in range(idxs.shape[0]):
                    members_converged.append(decentr_state[idxs[i]]["converged"])
                if not (sum(members_converged) == len(members_converged)):
                    # there are some members not converged
                    # TODO: what's the use of group_reindexing?
                    # multi-process working
                    group_reindexing = np.zeros((n_robots))
                    group_reindexing[np.where(group_mask)[0]] = np.arange(0,group_size,1) ## Pythonic?
                    '''
                    p = multiprocessing.Process(runSyncGaussSeidal, args=(decentr_state, \
                        distributed_mapper_location, group_i, group_reindexing, max_iters))
                    
                    multi_pool.append(p)
                    p.start()
                    p.join()
                    '''                    
                    to_calculate=[]
                    for i in range(idxs.shape[0]):
                        to_calculate.append(decentr_state[idxs[i]])
                    print (group_mask)

                    opt_handle["futures"]=runSyncGaussSeidal(to_calculate, distributed_mapper_location, group_i, group_reindexing, max_iters)
                        
                    if ['between_robot_links' not in opt_handle.keys()]:
                        opt_handle['between_robot_links'] = dict()                   
                    opt_handle['between_robot_links'][group_i] = countBetweenRobotLinks(to_calculate, group_reindexing)
                    print("Process {} Finished!".format(group_i))
        

                    update_state=opt_handle["futures"][0]
                    dgs_stats_i=opt_handle["futures"][1]
                    dgs_stats.append(dgs_stats_i)


                    print(dgs_stats_i['comm_link_count'])
                    print(opt_handle['between_robot_links'][group_i])
                    links = np.array(opt_handle['between_robot_links'][group_i])
                    links = links + links.T
                    linksum = links.sum()
                    assert(dgs_stats_i['comm_link_count'] == linksum)
                    # we should change here too
                    group = np.where(opt_handle['group_assignment'] == group_i)[0]
                    print ("Check the meaning of groups without this line")
                    group_data_exchange = links * dgs_stats_i['exchange_gs']/linksum

                    for i in range(group.shape[0]):
                        for j in range(group.shape[0]):
                            opt_increment[group[i], group[j]] = opt_increment[group[i], group[j]] + group_data_exchange[i,j]
        
                    # align or something else
        
                    decentr_state = consistentlyApplyState(update_state, decentr_state,idxs)
                    decentr_state = setConvergedWhereApplicable(update_state, decentr_state,idxs)
        
                    opt_handle['optimized_groups'] = list()
                    # unique list value
                    # TODO: group_assignment should be one less in python
                    # IPython.embed()
                    groups_set = set(opt_handle['group_assignment'])
                    groups = list(groups_set)
        
                    # deal with empty
                    if not opt_handle['optimized_groups']:
                        for gi in range(n_robots):
                            opt_handle['optimized_groups'].append(dict())
                            opt_handle['optimized_groups'][gi]['members'] = list()

                        for group_i in range(len(groups)):
                            opt_handle['optimized_groups'][group_i]['members'] = list()
                            for ga_i, ga in enumerate(opt_handle['group_assignment']):
                                if ga == groups[group_i]:
                                    opt_handle['optimized_groups'][group_i]['members'].append(ga_i)

                        print("Have {} optimization groups".format(len(opt_handle['optimized_groups'])))

        
    
    return [decentr_state, opt_handle, dgs_stats, opt_increment]
