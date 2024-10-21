import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def nmpc_controller():
    # Declare simulation constants
    T = 30 # planning horizon
    N = 0.1 # Number of control intervals
    h = T / N

    # system dimensions
    Dim_state = 4 # [x, y, ψ, v]
    Dim_ctrl  = 2 # [δ, a]

    # additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1)) # initial condition, # the state should be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1)) # leader car's velocity w.r.t ego car
    v_des = ca.MX.sym('v_des')
    delta_last = ca.MX.sym('delta_last')
    params = ca.vertcat(x_init, v_leader, v_des, delta_last)
    
    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1))
    u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    L_f = 1.0  # Distance from CG to front axle
    L_r = 1.0  # Distance from CG to rear axle

    beta = beta = ca.arctan((L_r / (L_r + L_f)) * ca.tan(u_model[0])) 

    xdot = ca.vertcat(
        x_model[3] * ca.cos(x_model[2] + beta) - v_leader[0],
        x_model[3] * ca.sin(x_model[2] + beta),
        (x_model[3] / L_r) * ca.sin(beta),
        u_model[1]
    )

    # Discrete time dynmamics model
    Func_dynamics_dt = ca.Function('f_dt', [x_model, u_model, params], [x_model + xdot * h])
    
    # Declare model variables, note the dimension
    x = ca.MX.sym('x', Dim_state, N + 1)  # State trajectory
    u = ca.MX.sym('u', Dim_ctrl, N)       # Control trajectory

    # Cost function design
    # Weights for the cost function
    w_v = 1.0       # Weight for velocity tracking (running cost)
    w_delta = 0.1   # Weight for steering angle regularization
    w_a = 0.1       # Weight for acceleration regularization
    w_vN = 10.0     # Weight for velocity tracking (terminal cost)
    w_yN = 100.0    # Weight for lane position in terminal cost
    
    # Keep in the same lane and take over it while maintaing a high speed
    # Terminal cost
    P = w_vN * ca.power(x_model[3] - v_des, 2) + w_yN * ca.power(x_model[1], 2)
    
    # Running cost
    L = w_v * ca.power(x_model[3] - v_des, 2) + w_delta * ca.power(u_model[0], 2) + w_a * ca.power(u_model[1], 2)

    Func_cost_terminal = ca.Function('P', [x_model, params], [P])
    Func_cost_running = ca.Function('Q', [x_model, u_model, params], [L])

    # state and control constraints
    state_ub = np.array([1e5, 3.0, 1e5, 50.0])   # Upper bounds for [x, y, ψ, v] 
    state_lb = np.array([-1e5, -1.0, -1e5, 0.0]) # Lower bounds for [x, y, ψ, v] 
    ctrl_ub = np.array([0.6, 4.0])               # Upper bounds for [δ, a] 
    ctrl_lb = np.array([-0.6, -10.0])            # Lower bounds for [δ, a] 
    
    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), ub_x.reshape((Dim_state * (N + 1), 1))))
    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), lb_x.reshape((Dim_state * (N + 1), 1))))

    # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    cons_dynamics = []
    ub_dynamics = np.zeros((Dim_state * N, 1))
    lb_dynamics = np.zeros((Dim_state * N, 1))
    for k in range(N):
        Fx = Func_dynamics_dt(x[:, k], u[:, k], params)
        cons_dynamics += [x[:, k + 1] - Fx]


    # state constraints: G(x) <= 0
    cons_state = []
    r_x = 30.0  # Collision avoidance ellipse parameters
    r_y = 2.0

    mu = 0.6    # Coefficient of friction
    g = 9.81    # Gravity acceleration
    gmu = 0.5 * mu * g  # Maximum allowable lateral acceleration

    y_L = 3.0   # Left lane boundary
    y_R = -1.0  # Right lane boundary

    dot_delta_max = 0.6  # Maximum steering rate (rad/s)
    for k in range(N):
        #### collision avoidance:
        cons_state.append(1 - (x[0, k] / r_x)**2 - (x[1, k] / r_y)**2)

        #### Maximum lateral acceleration ####
        dx = (x[:, k+1] - x[:, k]) / h
        ay = x[3, k] * (x[2, k + 1] - x[2, k]) / h # Compute the lateral acc using the hints
        
        cons_state.append(ay - gmu)
        cons_state.append(-ay - gmu)

        #### lane keeping ####
        cons_state.append(x[1, k] - y_L)
        cons_state.append(-x[1, k] - y_R)

        #### steering rate ####
        if k >= 1:
            d_delta = u[0, k] - u[0, k - 1]
            cons_state.append(d_delta - dot_delta_max * h)     # Δδ <= dot_delta_max * h
            cons_state.append(-d_delta - dot_delta_max * h)
        else:
            d_delta = u[0, k] - delta_last
            cons_state.append(d_delta - dot_delta_max * h)
            cons_state.append(-d_delta - dot_delta_max * h)

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
