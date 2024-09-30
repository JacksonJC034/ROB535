import numpy as np

def ACC_Controller(t, x, param):
    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    # cost function and constraints for the QP
    P = np.zeros((2,2))
    q = np.zeros([2, 1])
    A = np.zeros([5, 2])
    b = np.zeros([5])
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # set the parameters
    lam = 10.0 # Tracking constraint
    alpha = 0.1 # Safety constraint
    w = 1000000.0 # Slack

    # construct the cost function
    P[0, 0] = 1
    P[1, 1] = w
    # q = ...
    
    # construct the constraints
    A[0, 0] = (x[1] - vd) / m
    A[0, 1] = -1
    b[0] = -lam * ((x[1] - vd)** 2) / 2
    
    A[1, 0] = (1 / m) * (1.8 + (x[1] - v0) / (Cdg))
    B = x[0] - (1 / 2) * (v0 - x[1])** 2 / Cdg - 1.8 * x[1]
    b[1] = alpha * B + (v0 - x[1])
    
    A[2, 0] = -1 / m
    b[2] = Cdg
    
    A[3, 0] = 1 / m
    b[3] = Cag
    
    A[4, 1] = -1
    b[4] = 0

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    return A, b, P, q