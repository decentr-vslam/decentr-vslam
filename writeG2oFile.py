import numpy as np
from matrix_tool import tInv
import IPython

def gtsamFrameID(robot_i, frame_i):
    identity = (robot_i+97)*(2**56)+frame_i
    return identity


def rot2quat(R):
    """ROT2QUAT - Transform Rotation matrix into normalized quaternion.

    Usage: q = rot2quat(R)

    Input:
    R - 3-by-3 Rotation matrix

    Output:
    q - 4-by-1 quaternion, with form [w x y z], where w is the scalar term.
    """
    # By taking certain sums and differences of the elements
    # of R we can obtain all products of pairs a_i a_j with
    # i not equal to j. We then get the squares a_i^2 from
    # the diagonal of R.
    a2_a3 = (R[0, 1] + R[1, 0]) / 4
    a1_a4 = (R[1, 0] - R[0, 1]) / 4
    a1_a3 = (R[0, 2] - R[2, 0]) / 4
    a2_a4 = (R[0, 2] + R[2, 0]) / 4
    a3_a4 = (R[1, 2] + R[2, 1]) / 4
    a1_a2 = (R[2, 1] - R[1, 2]) / 4

    D = np.array([[+1, +1, +1, +1],
                  [+1, +1, -1, -1],
                  [+1, -1, +1, -1],
                  [+1, -1, -1, +1]]) * 0.25

    aa = np.dot(D, np.r_[np.sqrt(np.sum(R ** 2) / 3), np.diag(R)])

    # form 4 x 4 outer product a \otimes a:
    a_a = np.array([[aa[0], a1_a2, a1_a3, a1_a4],
                    [a1_a2, aa[1], a2_a3, a2_a4],
                    [a1_a3, a2_a3, aa[2], a3_a4],
                    [a1_a4, a2_a4, a3_a4, aa[3]]])

    # use rank-1 approximation to recover a, up to sign.
    U, S, V = np.linalg.svd(a_a)
    q = U[:, 0]
    # q = np.dot(_math.sqrt(S[0]), U[:, 0]) # Use this if you want unnormalized quaternions
    return q


def writeG2oPose(file_id, robot_idx, frame_idx, T_W_C):
    frame_id = gtsamFrameID(robot_idx, frame_idx)
    x=T_W_C[0,3]
    y=T_W_C[1,3]
    z=T_W_C[2,3]
    R=T_W_C[:3,:3]
    assert (sum(sum(R.imag))==0)
    R=R/(np.linalg.det(R)+1e-5)
    q=rot2quat(R)
    assert (np.linalg.norm(q)>1e-3)
    q=q/np.linalg.norm(q)
    assert (np.linalg.norm(q)>1e-3)
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    if (sum(q.imag) != 0):
        assert (False)
    file_id.write('VERTEX_SE3:QUAT %d %f %f %f %f %f %f %f\n'%(frame_id, x, y, z, qx, qy, qz, qw))


def writeG2oConstraint(file_id, from_id, to_id, T_from_to, covariance):
    dt = T_from_to[0:3, 3]
    dx = dt[0]
    dy = dt[1]
    dz = dt[2]
    
    dR = T_from_to[0:3, 0:3]
    dq = rot2quat(dR)
    norm_q = np.linalg.norm(dq)

    if norm_q > 1e-3:
        dq /= norm_q
    else:
        print("Norm close to zero for unit quaternion (2)")
        assert(False)
    
    dqw = dq[0]
    dqx = dq[1]
    dqy = dq[2]
    dqz = dq[3]
    
    I = covariance

    file_id.write('EDGE_SE3:QUAT %d %d   %f %f %f   %.7f %.7f %.7f %.7f   %f %f %f %f %f %f   %f %f %f %f %f   %f %f %f %f   %f %f %f   %f %f   %f\n' \
        %(from_id, to_id, dx, dy, dz, dqx, dqy, dqz, dqw, \
        I[0,0], I[0,1], I[0,2], I[0,3], I[0,4], I[0,5], \
        I[1,1], I[1,2], I[1,3], I[1,4], I[1,5], \
        I[2,2], I[2,3], I[2,4], I[2,5], \
        I[3,3], I[3,4], I[3,5], \
        I[4,4], I[4,5], \
        I[5,5]))

def writeDecentrStateToG2oFiles(decentr_state, outputDir, group_reindexing):
    nr_robots = len(decentr_state)
    file_ids = []

    for robot_i in range(nr_robots):
        file_ids.append(open("{}/{}.g2o".format(outputDir, robot_i), "w"))
    
    for robot_i in range(nr_robots):
        robot_state = decentr_state[robot_i]
        nr_poses = len(robot_state['Sim_O_C'])
        file_id = file_ids[robot_i]

        # add poses from Sim_O_C to initial
        for pose_i in range(nr_poses):
            T_O_C = robot_state['Sim_O_C'][pose_i]
            writeG2oPose(file_id, robot_i, pose_i, T_O_C)
        
        # generate odometry graph from pose_i-1 to pose_i
        for pose_i in range(nr_poses-1):
            relative_pose = robot_state['Sim_Cprev_C'][pose_i]
            writeG2oConstraint(file_id, gtsamFrameID(robot_i, pose_i), gtsamFrameID(robot_i, pose_i + 1), relative_pose, np.identity(6))
        
        # add inter-robot-edges to graphs:
        for match_i in list(robot_state["place_matches"].keys()):
            place_match = robot_state['place_matches'][match_i]

            if not place_match or (group_reindexing[place_match['robot_i']] < robot_i):
                continue
            
            matched_robot = group_reindexing[place_match['robot_i']]
            assert((place_match['robot_i'] in decentr_state[robot_i]['grouped_with']) or (robot_i == group_reindexing[place_match['robot_i']]))

            if (matched_robot < 0 ) or (matched_robot >= nr_robots):
                print("Matched robot out of boundary")
                assert(False)
            
            # add edges to both the graphs
            query_id = gtsamFrameID(robot_i, match_i)
            match_id = gtsamFrameID(matched_robot, place_match['frame_i'])

            T_Q_M = tInv(place_match['Sim_M_Q'])

            writeG2oConstraint(file_id, query_id, match_id, T_Q_M, np.identity(6))
            writeG2oConstraint(file_ids[int(matched_robot)], query_id, match_id, T_Q_M, np.identity(6))
    
    # close all file ids:
    for f in file_ids:
        f.close()
