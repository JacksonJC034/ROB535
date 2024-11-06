import numpy as np
import cvxpy as cp

def calc_Jacobian(x, u, param):

    L_f = param["L_f"]
    L_r = param["L_r"]
    dt   = param["h"]

    psi = x[2]
    v   = x[3]
    delta = u[1]
    a   = u[0]

    # Jacobian of the system dynamics
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    beta = np.arctan((L_r / (L_f + L_r)) * np.arctan(delta))
    A[0, 0] = 1
    A[0, 2] = -dt * v * np.sin(psi + beta)
    A[0, 3] = dt * np.cos(psi + beta)
    A[1, 1] = 1
    A[1, 2] = dt * v * np.cos(psi + beta)
    A[1, 3] = dt * np.sin(psi + beta)
    A[2, 2] = 1
    A[2, 3] = (dt * np.arctan(delta)) / (((L_r**2 * np.arctan(delta)**2) / (L_f + L_r)**2 + 1)**(0.5) * (L_r + L_f))
    A[3, 3] = 1
    
    B[0, 1] = -dt * (L_r * v * np.sin(psi + beta)) / ((delta**2 + 1) * ((L_f + L_r)**2) / (L_f + L_r)** 2 + 1) * (L_r + L_f)
    B[1, 1] = dt * (L_r * v * np.cos(psi + beta)) / ((delta**2 + 1) * ((L_f + L_r)**2) / (L_f + L_r)** 2 + 1) * (L_r + L_f)
    B[2, 1] = (dt * v) / ((delta ** 2 + 1) * ((L_r ** 2 * np.arctan(delta) ** 2) / (L_f + L_r) ** 2 + 1) ** (3/2) * (L_f + L_r))
    B[3, 0] = dt

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return [A, B]

def LQR_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]

    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl  # input constraints

    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # define the parameters
    Q = np.eye(4)  * 10
    R = np.eye(2)  * 1
    Pt = np.eye(4) * 5

    # define the cost function
    P_state = np.kron(np.eye(len_state - 1), Q)
    P_state_combine = np.block([
        [P_state, np.zeros((n_x - dim_state, dim_state))],
        [np.zeros((dim_state, n_x - dim_state)), Pt]
    ])
    P_control = np.kron(np.eye(len_ctrl), R)
    P = np.block([
        [P_state_combine, np.zeros((n_x, n_u))],
        [np.zeros((n_u, n_x)), P_control]
    ])
    q = np.zeros(n_var)
    
    # define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    
    for k in range(len_ctrl):
        A_k, B_k = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        A_cur = np.zeros((dim_state, n_var))
        A_cur[:, k * dim_state : (k + 1) * dim_state] = -A_k
        A_cur[:, (k + 1) * dim_state : (k + 2) * dim_state] = np.eye(dim_state)
        A_cur[:, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = -B_k

        A[k * dim_state : (k + 1) * dim_state, :] = A_cur
    
    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
                      [A @ x == b,
                       x[0:dim_state] == x0 - x_bar[0, :]])
    prob.solve(verbose=False, max_iter = 10000)


    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act

def CMPC_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl # input constraints

    a_limit = param["a_lim"]
    delta_limit = param["delta_lim"]
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # define the parameters
    Q = np.eye(4)  * 0.5
    R = np.eye(2)  * 0.1
    Pt = np.eye(4) * 200
    
    # define the cost function
    P_state = np.kron(np.eye(len_state - 1), Q)
    P_state_combine = np.block([
        [P_state, np.zeros((n_x - dim_state, dim_state))],
        [np.zeros((dim_state, n_x - dim_state)), Pt]
    ])
    P_control = np.kron(np.eye(len_ctrl), R)
    P = np.block([
        [P_state_combine, np.zeros((n_x, n_u))],
        [np.zeros((n_u, n_x)), P_control]
    ])
    q = np.zeros(n_var)
    
    # define the constraints
    A = np.zeros((n_eq, n_var))
    b = np.zeros(n_eq)
    G = np.zeros((n_ieq, n_var))
    ub = np.zeros(n_ieq)
    lb = np.zeros(n_ieq)
    
    for k in range(len_ctrl):
        A_k, B_k = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        A_cur = np.zeros((dim_state, n_var))
        A_cur[:, k * dim_state : (k + 1) * dim_state] = -A_k
        A_cur[:, (k + 1) * dim_state : (k + 2) * dim_state] = np.eye(dim_state)
        A_cur[:, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = -B_k

        A[k * dim_state : (k + 1) * dim_state, :] = A_cur
        
        G[k * dim_ctrl : (k + 1) * dim_ctrl, n_x + k * dim_ctrl : n_x + (k + 1) * dim_ctrl] = np.eye(dim_ctrl)
        ub[k * dim_ctrl : (k + 1) * dim_ctrl] = np.array([a_limit[1], delta_limit[1]]) - u_bar[k, :]
        lb[k * dim_ctrl : (k + 1) * dim_ctrl] = np.array([a_limit[0], delta_limit[0]]) - u_bar[k, :]

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    cons = [A @ x == b,
            G @ x <= ub,
            G @ x >= lb,
            x[0:dim_state] == x0 - x_bar[0, :]]
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                      cons)
    prob.solve(verbose=False, max_iter = 10000)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act