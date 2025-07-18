import casadi as ca
import numpy as np

def step_lower_mpc(Qin_est, Qout_meas, h_meas, w_meas, E_meas, P_meas, h_ref, trigger, 
                        N, zs, models_coefficients):
    """
    Height Reference MPC implemented as a Python function.
    """
    # Solver options
    p_opts = {
        "expand": True,
        "print_time": 0,
        "verbose": False,
        "error_on_fail": False,
    }
    s_opts = {
        'max_iter': 2000,
        "print_level": 0,
        "warm_start_init_point": "yes"
    }

    opti = ca.Opti()
    opti.solver("ipopt", p_opts, s_opts)

    # Parameters
    A = 40
    B1_pressure = np.array(models_coefficients["a_pressure"])
    B2_pressure = np.array(models_coefficients["b_pressure"])
    C_pressure = np.array(models_coefficients["c_pressure"])
    A_power = np.array(models_coefficients["A_power"])
    B_power = np.array(models_coefficients["B_power"])
    C_power = np.array(models_coefficients["C_power"])

    # Create parameters and set their values
    Qout_meas_param = opti.parameter(zs)
    h_meas_param = opti.parameter(zs)
    P_meas_param = opti.parameter(zs)
    Qin_est_param = opti.parameter(N+zs)
    w_meas_param = opti.parameter(3, zs)
    h_ref_param = opti.parameter(1)
    E_meas_param = opti.parameter(3, zs)
    trigger_param = opti.parameter(3)

    # Set parameter values
    opti.set_value(Qout_meas_param, Qout_meas)
    opti.set_value(h_meas_param, h_meas)
    opti.set_value(P_meas_param, P_meas)
    opti.set_value(Qin_est_param, Qin_est)
    opti.set_value(w_meas_param, w_meas)
    opti.set_value(h_ref_param, h_ref)
    opti.set_value(E_meas_param, E_meas)
    opti.set_value(trigger_param, trigger)

    # Variables
    Qout = opti.variable(N+zs)
    E = opti.variable(3, N+zs)
    P = opti.variable(N+zs)
    w = opti.variable(3, N+zs)
    h = opti.variable(N+zs)
    s_h = opti.variable(N+zs)
    s_P = opti.variable(N+zs)

    # Objective function
    objective = 0
    for t in range(zs, N+zs):
        objective += (
            0.1 * (h[t] - h_ref_param) ** 2 +
            0.5*((w[:, t] - w[:, t-1]).T @ (w[:, t] - w[:, t-1])) +
            1 * (
                ca.if_else(trigger_param[0] > 0, 0, w[0, t]) +
                ca.if_else(trigger_param[1] > 0, 0, w[1, t]) +
                ca.if_else(trigger_param[2] > 0, 0, w[2, t])
            ) +
            10 * s_P[t] + 
            10 * s_h[t]
        )

    opti.minimize(objective)

    # ARX model constraints
    for t in range(zs, N+zs):
        opti.subject_to(Qout[t] == ca.if_else(
            w[0,t-1] <= 600,
            3.2216 + 0.08378681 * w[0,t-1],
            3.22 + 0.083 * 600 + 0.8371 * (w[0,t-1] - 600)
        ) + ca.if_else(
            w[1,t-1] <= 600,
            3.2216 + 0.08378681 * w[1,t-1],
            3.22 + 0.083 * 600 + 0.8371 * (w[1,t-1] - 600)
        ) + ca.if_else(
            w[2,t-1] <= 600,
            3.2216 + 0.08378681 * w[2,t-1],
            3.22 + 0.083 * 600 + 0.8371 * (w[2,t-1] - 600)
        ))

        opti.subject_to(E[:,t] == A_power @ ca.vcat([E[:,t-1],E[:,t-2]]) + B_power @ w[:,t-1] + C_power)
        opti.subject_to(P[t] == B1_pressure @ Qout[t-1] + B2_pressure @ Qout[t-1]**2 + C_pressure)
        opti.subject_to(h[t] == h[t-1] + (100 * (Qin_est_param[t-1] - Qout[t-1])) / (A * 40))

    # Additional constraints
    for t in range(zs, N+zs):
        opti.subject_to(w[:,t] >= 0)
        opti.subject_to(w[:,t] <= 1500)
        opti.subject_to(h[t] <= (200 + s_h[t]))
        opti.subject_to(h[t] >= (70 - s_h[t]))
        opti.subject_to(P[t] <= 1)
        opti.subject_to(P[t] >= 0)
        opti.subject_to(s_h[t] >= 0)
        opti.subject_to(s_P[t] >= 0)

    # Initial conditions
    opti.subject_to(Qout[0:zs] == Qout_meas_param)
    opti.subject_to(h[0:zs] == h_meas_param)
    opti.subject_to(w[:,0:zs] == w_meas_param)
    opti.subject_to(P[0:zs] == P_meas_param)
    opti.subject_to(E[:,0:zs] == E_meas_param)

    try:
        sol = opti.solve()
        return (
            sol.value(w),
            sol.value(Qout),
            sol.value(h),
            sol.value(E),
            sol.value(P)
        )
    except Exception as e:
        print("Solver failed to find a solution.")
        print(f"Error: {e}")
        
        # Debugging information
        print("\nSolver Debug Information:")
        print("w:", opti.debug.value(w))
        print("Qout:", opti.debug.value(Qout))
        print("h:", opti.debug.value(h))
        print("E:", opti.debug.value(E))
        print("P:", opti.debug.value(P))
        print("Objective value:", opti.debug.value(opti.f))
