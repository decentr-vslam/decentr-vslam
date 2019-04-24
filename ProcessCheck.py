import numpy as np
from matrix_tool import fixT, tInv
import IPython

def countBetweenRobotLinks(decentr_state, group_reindexing):
    nr_robots = len(decentr_state)
    between_robot_links = np.zeros((nr_robots, nr_robots))

    for robot_i in range(nr_robots):
        robot_state = decentr_state[robot_i]

        for match_i in list(robot_state['place_matches'].keys()):
            place_match = robot_state['place_matches'][match_i]     

            if (not place_match) or (group_reindexing[place_match["robot_i"]] <= robot_i):
                # we only do one direction match
                continue
            
            matched_robot = group_reindexing[place_match["robot_i"]]
            # TODO: Understanding the data in grouped with
            assert(place_match["robot_i"] in decentr_state[robot_i]['grouped_with'])
            between_robot_links[robot_i,int(matched_robot)] += + 1
    
    return between_robot_links


def consistentlyApplyState(updated_state, decentr_state,idxs):
    # check data consistency
    n_robots = len(decentr_state)
    if updated_state:
        # if update_state is not empty
        for robot_i in range(len(updated_state)):
            updated = updated_state[robot_i]['Sim_O_C']
            n_update_frames = len(updated)  # the frames needed to update
            n_frames = len(decentr_state[idxs[robot_i]]['Sim_O_C'])  # total frames

            assert(n_update_frames <= n_frames)

            for frame_i in range(n_update_frames-1):
                # we don't update the last one
                decentr_state[idxs[robot_i]]['Sim_O_C'][frame_i] = updated[frame_i]  # update frame
            
            T_update = np.matmul(fixT(updated[n_update_frames-1]), tInv(decentr_state[idxs[robot_i]]['Sim_O_C'][n_update_frames-1]))

            for frame_i in range(n_update_frames-1, n_frames):
                decentr_state[idxs[robot_i]]['Sim_O_C'][frame_i] = fixT(np.matmul(T_update, decentr_state[idxs[robot_i]]['Sim_O_C'][frame_i]))
    
    return decentr_state


def setConvergedWhereApplicable(updated_state, decentr_state,idxs):
    n_robots = len(updated_state)
    if updated_state:
        # if update_state is not empty 
        for robot_i in range(n_robots):
            matched_frames = [bool(x) for x in decentr_state[idxs[robot_i]]['place_matches']]
            last_matched_frame = []
            for mi, mf in enumerate(matched_frames):
                if mf and (not last_matched_frame):
                    last_matched_frame.append(mi)
                elif mf:
                    last_matched_frame[0] = mi

            if not last_matched_frame:
                # no match here
                continue
            
            last_updated_frame = len(updated_state[robot_i]['Sim_O_C'])

            if last_updated_frame >= last_matched_frame[0]:
                decentr_state[idxs[robot_i]]['converged'] = True

    return decentr_state
