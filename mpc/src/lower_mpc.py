import casadi as ca
import numpy as np
import polars as pl 

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

    a1_qout = np.array([0.98, 0.98, 0.98])
    b1_qout = np.array([0.00575, 0.00575, 0.00575])

    a1_power = np.array([0.62416, 0.62416, 0.62416])
    b1_power = np.array([0.01719, 0.01719, 0.01719])

    A_qout = np.diag(a1_qout) 
    B_qout = np.diag(b1_qout)
    C_qout = np.zeros(3)

    A_power = np.diag(a1_power) 
    B_power = np.diag(b1_power)
    C_power = np.zeros(3)

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
    Qout_pump = opti.variable(3, N+zs)
    E = opti.variable(3, N+zs)
    P = opti.variable(N+zs)
    w = opti.variable(3, N+zs)
    h = opti.variable(N+zs)
    s_h = opti.variable(N+zs)
    s_P = opti.variable(N+zs)

    # Objective function
    objective = 0   

    w1 = 20
    w2 = 5
    w3 = 50
    w4 = 10
    w5 = 1e3 
    for t in range(zs, N+zs):
        objective += (
            w1 * (h[t] - h_ref_param) ** 2 
            + w2 * ((w[:, t] - w[:, t-1]).T @ (w[:, t] - w[:, t-1]))
            + w3 * (ca.if_else(trigger_param[0] > 0, 0, w[0, t]) +
                ca.if_else(trigger_param[1] > 0, 0, w[1, t]) +
                ca.if_else(trigger_param[2] > 0, 0, w[2, t])) +
            w4 * s_P[t] + 
            w5 * s_h[t]
        )

    opti.minimize(objective)


    
    for t in range(zs, N+zs):
        opti.subject_to(Qout_pump[:, t] == A_qout @ Qout_pump[:, t-1] + B_qout @ w[:, t-1] + C_qout)
        opti.subject_to(Qout[t] == ca.sum1(Qout_pump[:,t]))

        #opti.subject_to(E[:,t] == A_power @ ca.vcat([E[:,t-1],E[:,t-2]]) + B_power @ w[:,t-1] + C_power)
        opti.subject_to(E[:,t] == A_power @ E[:,t-1] + B_power @ w[:,t-1] + C_power)

        opti.subject_to(P[t] == B1_pressure @ Qout[t-1] + B2_pressure @ Qout[t-1]**2 + C_pressure)
        opti.subject_to(h[t] == h[t-1] + (100 * (Qin_est_param[t-1] - Qout[t-1])) / (A * 40))

    # Additional constraints
    for t in range(zs, N+zs):
        opti.subject_to(w[:,t] >= 0)
        opti.subject_to(E[:,t] >= 0)
        opti.subject_to(w[:,t] <= 1500)
        opti.subject_to(h[t] <= (200 + s_h[t]))
        opti.subject_to(h[t] >= (70 - s_h[t]))
        opti.subject_to(s_h[t] >= 0)
        opti.subject_to(P[t] <= 1)
        opti.subject_to(P[t] >= 0)
        opti.subject_to(Qout[t] >= 0)
        opti.subject_to(Qout[t] <= 800)
        opti.subject_to(s_P[t] >= 0)

    # Initial conditions
    opti.subject_to(Qout[0:zs] == Qout_meas_param)
    opti.subject_to(h[0:zs] == h_meas_param)
    opti.subject_to(w[:,0:zs] == w_meas_param)
    opti.subject_to(P[0:zs] == P_meas_param)
    opti.subject_to(E[:,0:zs] == E_meas_param)

    try:
        sol = opti.solve()
        w_val = sol.value(w)        
        E_val = sol.value(E)          
        Qout_val = sol.value(Qout)    
        pressure = sol.value(P)    
        height_sys = sol.value(h)    

        result = {
            'u1': (w_val[0, :].copy()).astype(int),    
            'u2':(w_val[1, :].copy()).astype(int),
            'u3': (w_val[2, :].copy()).astype(int),
            'qout': np.round(Qout_val.copy(), 1),    
            'p1_power': np.round(E_val[0, :].copy(), 1),
            'p3_power': np.round(E_val[1, :].copy(), 1),
            'p4_power': np.round(E_val[2, :].copy(), 1),
            'pressure_sys': np.round(pressure.copy(), 3),
            'height_sys': np.round(height_sys.copy(), 1),
        }
        return {k: v[zs] for k, v in result.items()}

        
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


class LMPCDataBuffer:
    def __init__(self):
        # Change or extend keys as needed:
        self.data = {
            'time_utc': [],
            'qout': [],
            'qin': [],
            'u1': [],
            'u2': [],
            'u3': [],
            'p1_power': [],
            'p3_power': [],
            'p4_power': [],
            'height_sys': [],
            'height_ref': [],
            'pressure_sys': [],
            'perf_time_lower': [],
            'perf_time_upper': [],
        }

    def initialize(self):
        self.data = {
                    "time_utc": [
                        "2023-10-11 00:00:00+00:00",
                        "2023-10-11 00:01:00+00:00",
                        "2023-10-11 00:02:00+00:00"
                    ],
                    "u1": [0.0, 0.0, 0.0],
                    "u2": [0.0, 0.0, 0.0],
                    "u3": [756.9333333333333, 750.0, 750.0],
                    "p1_power": [147.38333333333333, 132.16666666666666, 136.31666666666666],
                    "p3_power": [208.37219522124886, 208.12735204132585, 207.8796690938776],
                    "p4_power": [150.2, 150.0, 149.75],
                    "qout": [0.5392500013113022, 0.5376500050226848, 0.5399333328008652],
                    "qin": [0.0, 0.0, 0.0],
                    "height_sys": [0.0, 0.0, 0.0],
                    "height_ref": [29.57499993642171, 29.346666685740153, 29.686666679382324],
                    "pressure_sys": [0.5, 0.5, 0.5],      
                    "opt_time_lmpc": [np.nan, np.nan, np.nan],   
                }

    def update(self, entry_dict):
        for key in self.data:
            val = entry_dict.get(key, None)
            if isinstance(val, np.generic):
                val = val.item()
            self.data[key].append(val)

    def to_dataframe(self, save=False, file_path=None):
        n = 3
        for k, v in self.data.items():
            try:
                self.data[k] = [float(x) if x is not None else None for x in v]
            except Exception:
                pass
        self.data = {k: v[n:] for k, v in self.data.items()}
        df = pl.DataFrame(self.data)
        if save:
            df.write_parquet(file_path)
        return df

        