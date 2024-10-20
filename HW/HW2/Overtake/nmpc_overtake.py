import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def nmpc_controller():
    # Declare simulation constants
    T = # TODO: planning horizon
    N = # TODO: Number of control intervals
    h = T / N

    # system dimensions
    Dim_state = # TODO
    Dim_ctrl  = # TODO

    # additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1)) # initial condition, # the state should be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1)) # leader car's velocity w.r.t ego car
    v_des = ca.MX.sym('v_des')
    delta_last = ca.MX.sym('delta_last')
    params = ca.vertcat(x_init, v_leader, v_des, delta_last)
    
    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1))
    u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    L_f = 1.0 # Car parameters, do not change
    L_r = 1.0 # Car parameters, do not change

    beta = # TODO 

    xdot = # TODO

    # Discrete time dynmamics model
    Func_dynmaics_dt = # TODO
    
    # Declare model variables, note the dimension
    x = # TODO
    u = # TODO

    # Keep in the same lane and take over it while maintaing a high speed
    P = # TODO
    L = # TODO

    Func_cost_terminal = ca.Function('P', [x_model, params], [P])
    Func_cost_running = ca.Function('Q', [x_model, u_model, params], [L])

    # state and control constraints
    state_ub = # TODO 
    state_lb = # TODO 
    ctrl_ub  = # TODO 
    ctrl_lb  = # TODO 
    
    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    ub_var = np.concatenate((ub_u.reshape((# TODO, 1)), ub_x.reshape((# TODO, 1))))
    lb_var = np.concatenate((lb_u.reshape((# TODO, 1)), lb_x.reshape((# TODO, 1))))

    # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    cons_dynamics = []
    ub_dynamics = np.zeros((# TODO, 1))
    lb_dynamics = np.zeros((# TODO, 1))
    for k in range(N):
        Fx = Func_dynmaics_dt(x[:, k], u[:, k], params)
        # TODO


    # state constraints: G(x) <= 0
    cons_state = []
    for k in range(N):
        #### collision avoidance:
        cons_state.append(# TODO)

        #### Maximum lateral acceleration ####
        dx = (x[:, k+1] - x[:, k]) / h
        ay = # TODO: Compute the lateral acc using the hints
        
        gmu = (0.5 * 0.6 * 9.81)
        cons_state.append(# TODO)
        cons_state.append(# TODO)

        #### lane keeping ####
        cons_state.append(# TODO)
        cons_state.append(# TODO)

        #### steering rate ####
        if k >= 1:
            d_delta = # TODO
            cons_state.append(# TODO)
            cons_state.append(# TODO)
        else:
            d_delta = # TODO, for the first input, given d_last from param
            cons_state.append(# TODO)
            cons_state.append(# TODO)

    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e5

    # cost function: # NOTE: You can also hard code everything here
    J = Func_cost_terminal(x[:, -1], params)
    for k in range(N):
        J = J + Func_cost_running(x[:, k], u[:, k], params)

    # initial condition as parameters
    cons_init = [x[:, 0] - x_init]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))
    
    # Define variables for NLP solver
    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)))
    cons_NLP = cons_dynamics + cons_state + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_state_cons, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_state_cons, ub_init_cons))

    # Create an NLP solver
    prob = {"x": vars_NLP, "p":params, "f": J, "g":cons_NLP}
    
    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], params.shape[0], lb_var, ub_var, lb_cons, ub_cons
