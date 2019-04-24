# matrix tools for matrix:
import math
import numpy as np


def fixR(R):
    # R is basically an array & not matrix
    if (np.abs(np.linalg.det(R) - 1.0) > 1e-5):
        print(R)
        print(np.linalg.det(R))
        assert(False)
    
    [_, S, V] = np.linalg.svd(R)
    corr = np.matmul(V[:, 0:1], V[:, 0:1].T)/S[0] + np.matmul(V[:, 1:2], V[:, 1:2].T)/S[1] \
        + np.matmul(V[:, 2:3], V[:, 2:3].T)/S[2]

    return np.matmul(R, corr)


def fixT(T):
    assert(T.shape == (4, 4))
    T[0:3, 0:3] = fixR(T[0:3, 0:3]) 
    # TODO: do we need to check [0, 0, 0, 1]
    return T


def tInv(T):
    invT = np.hstack([T[0:3, 0:3].T, -np.matmul(T[0:3, 0:3].T, T[0:3, 3:4])])
    invT = np.vstack([invT, np.array([0, 0, 0, 1])])
    return invT


def rpyxyzToT(rpyxyz):
    T = np.identity(4)
    if type(rpyxyz) == list:
        assert(len(rpyxyz) == 6)
    elif type(rpyxyz) == np.ndarray:
        assert(rpyxyz.shape[0] == 6)

    roll = rpyxyz[0]
    pitch = rpyxyz[1]
    yaw = rpyxyz[3]
    
    R_roll = np.array([[ 1, 0, 0],
                       [ 0, math.cos(roll), -math.sin(roll)],
                       [ 0, math.sin(roll), math.cos(roll)]])

    R_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
    
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                      [math.sin(yaw), math.cos(yaw), 0],
                      [0, 0, 1]])
    
    R = np.matmul(R_roll, np.matmul(R_pitch, R_yaw))

    T[0:3, 0:3] = R
    T[0:3, 3] = rpyxyz[3:]
    return T

if __name__ == "__main__":
    rpyxyz = [0.3, 0.3, 0.3, 1, 2, 3]
    T = rpyxyzToT(rpyxyz)
    print(T)
    