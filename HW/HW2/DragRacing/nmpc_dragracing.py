from sim import *
from utils import *

def nmpc_controller(kappa_table = None):
    T = 10 # TODO: planning horizon
    N = 50 # TODO: number of control intervals
    h = T / N
    ###################### Modeling Start ######################
    # system dimensions
    Dim_state = 6
    Dim_ctrl  = 2
    Dim_aux   = 4 # (F_yf, F_yr, F_muf, F_mur)

    xm = ca.MX.sym('xm', (Dim_state, 1))
    um = ca.MX.sym('um', (Dim_ctrl, 1))
    zm = ca.MX.sym('zm', (Dim_aux, 1))

    ## rename the control inputs
    Fx = um[0]
    delta = um[1]

    ## Air drag and Rolling Resistance
    Fd = param["Frr"] + param["Cd"] * xm[0]**2
    Fd = Fd * ca.tanh(- xm[0] * 100)
    Fb = 0.0
    ## 
    
    # hint: find useful functions in utils.py
    af, ar = get_slip_angle(xm[0], xm[1], xm[2], delta, param) # TODO: refer to the car simulator to get the slip angle
    Fzf, Fzr = normal_load(Fx, param) # TODO: refer to the car simulator to get the normal load distribution
    Fxf, Fxr = chi_fr(Fx) # TODO: refer to the car simulator to get the tire force distribution

    Fyf = tire_model_ctrl(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"]) # TODO: use the modified tire force model tire_model_ctrl()
    Fyr = tire_model_ctrl(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"]) # TODO: use the modified tire force mode ltire_model_ctrl()

    dUx = (Fxf * ca.cos(delta) - zm[0] * ca.sin(delta) + Fxr + Fd) / param["m"] + xm[2] * xm[1]
    dUy = (zm[0] * ca.cos(delta) + Fxf * ca.sin(delta) + zm[1] + Fb) / param["m"] - xm[2] * xm[0] # TODO: refer to car simulator, replace Fyf and Fyr with auxiliary variable
    dr = (param["L_f"] * (zm[0] * ca.cos(delta) + Fxf * ca.sin(delta)) - param["L_r"] * zm[1]) / param["Izz"] # TODO: refer to car simulator, replace Fyf and Fyr with auxiliary variable
    
    dx = ca.cos(xm[5]) * xm[0] - ca.sin(xm[5]) * xm[1]
    dy = ca.sin(xm[5]) * xm[0] + ca.cos(xm[5]) * xm[1]
    dyaw = xm[2]
    xdot = ca.vertcat(dUx, dUy, dr, dx, dy, dyaw)

    xkp1 = xdot * h + xm
    Fun_dynmaics_dt = ca.Function('f_dt', [xm, um, zm], [xkp1])

    # enforce constraints for auxiliary variable z[0] = Fyf
    alg = ca.vertcat(
        zm[0] - tire_model_ctrl(af, Fzf, Fxf, param["C_alpha_f"], param["mu_f"]),
        zm[1] - tire_model_ctrl(ar, Fzr, Fxr, param["C_alpha_r"], param["mu_r"])
    )
    Fun_alg = ca.Function('alg', [xm, um, zm], [alg])
    
    ###################### MPC variables ######################
    x = ca.MX.sym('x', (Dim_state, N + 1))
    u = ca.MX.sym('u', (Dim_ctrl, N))
    z = ca.MX.sym('z', (Dim_aux, N))
    p = ca.MX.sym('p', (Dim_state, 1))

    ###################### MPC constraints start ######################
    ## MPC equality constraints ##
    cons_dynamics = []
    for k in range(N):
        xkp1 = Fun_dynmaics_dt(x[:, k], u[:, k], z[:, k])
        Fy2  = Fun_alg(x[:, k], u[:, k], z[:, k])
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k+1] - xkp1[j])
        for j in range(2):
            cons_dynamics.append(Fy2[j])
    
    ## MPC inequality constraints ##
    # G(x) <= 0
    cons_ineq = []

    ## state / inputs limits:
    for k in range(N):
        # Minimal longitudinal speed: Ux >= 2
        cons_ineq.append(2 - x[0, k])  # 2 - Ux <= 0

        # Engine power limits: Fx <= Peng / Ux
        cons_ineq.append(u[0, k] - param["Peng"] / x[0, k])  # Fx - Peng / Ux <= 0

        # Collision avoidance
        cons_ineq.append(
            1 - ((x[3, k] - 500) / 10)**2 - (x[4, k] / 10)**2
        )  # 1 - ((x - 500)/10)^2 - (y/10)^2 <= 0

    ## friction cone constraints
    for k in range(N):
        Fx_k, delta_k = u[0, k], u[1, k]
        af_k, ar_k = get_slip_angle(x[0, k], x[1, k], x[2, k], delta_k, param)
        Fzf_k, Fzr_k = normal_load(Fx_k, param)
        Fxf_k, Fxr_k = chi_fr(Fx_k) 

        Fyf_k = z[0, k] # use tire_model_ctrl or auxiliary variable z[0, k] z[1, k]
        Fyr_k = z[1, k] # use tire_model_ctrl or auxiliary variable z[0, k] z[1, k]

        # Front tire friction cone
        cons_ineq.append(
            Fyf_k**2 + Fxf_k**2 - (param["mu_f"] * Fzf_k)**2 - z[2, k]**2
        )
        # Rear tire friction cone
        cons_ineq.append(
            Fyr_k**2 + Fxr_k**2 - (param["mu_r"] * Fzr_k)**2 - z[3, k]**2
        )        

    ###################### MPC cost start ######################
    ## cost function design, you can use a desired velocity v_des for stage cost
    J = 0.0
    J += -x[3, N]  # TODO: terminal cost
    
    # Stage cost weights
    w_y = 100.0    # Weight for staying close to y = 0
    w_phi = 100.0  # Weight for stabilizing yaw angle
    W_alpha = 1000.0  # Weight for slip angle penalty
    W_zmu = 1000.0    # Weight for friction cone slack variables
    
    ## road tracking 
    for k in range(N):
        # stage cost
        J += w_y * x[4, k]**2 # Stay close to y = 0
        J += w_phi * x[5, k]**2 # Stabilize yaw angle to 0
 
    ## excessive slip angle / friction
    for k in range(N):
        Fx_k = u[0, k]; delta_k = u[1, k]
        af_k, ar_k = get_slip_angle(x[0, k], x[1, k], x[2, k], delta_k, param)
        Fzf, Fzr = normal_load(Fx_k, param)
        Fxf, Fxr = chi_fr(Fx_k)

        xi = 0.85
        F_offset = 2000 ## a slacked ReLU function using only sqrt()
        Fyf_max_sq = (param["mu_f"] * Fzf)**2 - (0.999 * Fxf)**2
        Fyf_max_sq = (ca.sqrt( Fyf_max_sq**2 + F_offset) + Fyf_max_sq) / 2
        Fyf_max = ca.sqrt(Fyf_max_sq)

        ## modified front slide sliping angle
        alpha_mod_f = ca.arctan(3 * Fyf_max / param["C_alpha_f"] * xi)

        Fyr_max_sq = (param["mu_f"] * Fzf)**2 - (0.999 * Fxf)**2
        Fyr_max_sq = (ca.sqrt( Fyr_max_sq**2 + F_offset) + Fyr_max_sq) / 2
        Fyr_max = ca.sqrt(Fyr_max_sq)

        ## modified rear slide sliping angle
        alpha_mod_r = ca.arctan(3 * Fyr_max / param["C_alpha_r"] * xi)

        ## limit friction penalty
        J = J + W_alpha * ca.if_else( # Avoid front tire saturation
            ca.fabs(af_k) >= alpha_mod_f,
            (ca.fabs(af_k) - alpha_mod_f)**2,
            0
        )
        J = J + W_alpha * ca.if_else( # Avoid rear tire saturation
            ca.fabs(ar_k) >= alpha_mod_r,
            (ca.fabs(ar_k) - alpha_mod_r)**2,
            0
        )
        J = J + W_zmu * (z[2, k]**2 + z[3, k]**2) # Penalize slack variable for friction cone limits

    # initial condition as parameters
    cons_init = [x[:, 0] - p]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))

    state_ub = np.array([ 1e2,  1e2,  1e2,  1e8,  1e8,  1e8])
    state_lb = np.array([-1e2, -1e2, -1e2, -1e8, -1e8, -1e8])
    ctrl_ub  = np.array([1e5, param["delta_max"]]) # (traction force, param["delta_max"])
    ctrl_lb  = np.array([0.0, -param["delta_max"]]) # (-traction force, -param["delta_max"])
    aux_ub   = np.array([ 1e5,  1e5,  1e5,  1e5])
    aux_lb   = np.array([-1e5, -1e5, -1e5, -1e5])

    lb_dynamics = np.zeros((len(cons_dynamics), 1))
    ub_dynamics = np.zeros((len(cons_dynamics), 1))

    lb_ineq = np.zeros((len(cons_ineq), 1)) - 1e9
    ub_ineq = np.zeros((len(cons_ineq), 1))

    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)
    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)
    ub_z = np.matlib.repmat(aux_ub, N, 1)
    lb_z = np.matlib.repmat(aux_lb, N, 1)

    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), 
                             lb_x.reshape((Dim_state * (N+1), 1)),
                             lb_z.reshape((Dim_aux * N, 1))
                             ))

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), 
                             ub_x.reshape((Dim_state * (N+1), 1)),
                             ub_z.reshape((Dim_aux * N, 1))
                             ))

    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)), z.reshape((Dim_aux * N, 1)))
    cons_NLP = cons_dynamics + cons_ineq + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_ineq, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_ineq, ub_init_cons))

    prob = {"x": vars_NLP, "p":p, "f": J, "g":cons_NLP}

    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], p.shape[0], lb_var, ub_var, lb_cons, ub_cons