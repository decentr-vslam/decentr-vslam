import re
import numpy as np
import IPython
def gtsamFrameIdToIndices(identity):
    assert (type(identity) is int)
    frame_i = np.mod(identity, 2**56)
    robot_i = (identity - frame_i + 1) / 2**56 - 97 + 1
    return robot_i, frame_i

def vec(v):
    return v.reshape(-1,1)

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


def quat2rot(q):
    """QUAT2ROT - Transform quaternion into rotation matrix

    Usage: R = quat2rot(q)

    Input:
    q - 4-by-1 quaternion, with form [w x y z], where w is the scalar term.

    Output:
    R - 3-by-3 Rotation matrix
    """

    q = q / np.linalg.norm(q)

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    xy = 2 * x * y
    xz = 2 * x * z
    yz = 2 * y * z
    wx = 2 * w * x
    wy = 2 * w * y
    wz = 2 * w * z

    R = np.array([[w2 + x2 - y2 - z2, xy - wz, xz + wy],
                  [xy + wz, w2 - x2 + y2 - z2, yz - wx],
                  [xz - wy, yz + wx, w2 - x2 - y2 + z2]])
    return R


def readDecentrStateFromOptG2oFiles(g2o_dir, decentr_state, suffix):
    nr_robots = len(decentr_state)
    for robot_i in range(nr_robots):
        file_id = open(g2o_dir+'/' + str(robot_i) + suffix + '.g2o', 'r')
        while True:
            line = file_id.readline()[:-1]

            if line.split(' ')!='VERTEX_SE3:QUAT':
                break
            head,frame_id,x,y,z,qx,qy,qz,qw = line.split(' ')
            if head[:]!=head[:]:
                break
            frame_id = int(frame_id)
            x=float(x)
            y=float(y)
            z=float(z)
            qx=float(qx)
            qy=float(qy)
            qz=float(qz)

            qw=float(qw)
            [robot_i_val, frame_i] = gtsamFrameIdToIndices(frame_id)
            assert (robot_i_val == robot_i)

            Sim_W_C = np.eye(4)
            Sim_W_C[:3,3] = np.array([x,y,z])
            q = np.array([qw,qx,qy,qz])
            if (abs(np.linalg.norm(q)-1)>1e-3):
                print('Error: Quaternion has not unit norm')
            else:
                q = q/np.linalg.norm(q)

            Sim_W_C[:3,:3] = fixR(quat2rot(q))

            decentr_state[robot_i]["Sim_O_C"][frame_i] = Sim_W_C
    
    return decentr_state




