import casadi as ca 
import numpy as np
import polars as pl 
from typing import Optional, Dict, List, Any
from typing import Optional
np.set_printoptions(suppress=True, precision=1)


def step_upper_level(horizon, prices_values, co2_progn_values, inflow_values, h_init, energy_init, Qout_init, resid):
    opti = ca.Opti()

    np.random.seed(42) 

    p_opts = {
        "expand":True,
        "print_time": 0,
        "verbose": False,
        "error_on_fail": False,
    }
    s_opts = {
        'max_iter':1000,
        "print_level":0, 
        "warm_start_init_point": "yes"}

    opti.solver("ipopt",p_opts,s_opts)



    num_scenarios = resid.shape[0]//24
    #print(num_scenarios)
    usable_length = num_scenarios * 24
    ### Parameters
    da_prices = opti.parameter(horizon)
    co2_progn = opti.parameter(horizon)
    Qin_forecast = opti.parameter(horizon)
    energy = opti.variable( horizon)
    Qout = opti.variable(horizon)
    height = opti.variable(num_scenarios, horizon)

    opti.set_value(da_prices, prices_values)
    opti.set_value(co2_progn, co2_progn_values)
    opti.set_value(Qin_forecast, inflow_values)

    opti.subject_to(height[:, 0] == h_init)
    opti.subject_to(Qout[0] == Qout_init)
    opti.subject_to(energy[0] == energy_init)

    horizon = 24
    error_samples = np.random.normal(30, 2, 1)
    e_qin = opti.parameter(horizon)
    opti.set_value(e_qin, error_samples)

    objective = 0

    #resids = np.random.normal(30, 2, size=(num_scenarios, horizon))
    bootstrap_samples = np.random.choice(resid, size=usable_length, replace=True)
    resids_bootstrap = resid[:usable_length].reshape((num_scenarios, 24))

    cvar_alpha = 0.95
    epsilon = 0.001

    objective = 0  #
    eta = opti.variable()              
    s = opti.variable(num_scenarios, horizon)  # 2D: scenarios x time

    for t in range(1, horizon):

        w1 = 1000
        w2 = 10

        objective += w1*(da_prices[t] * energy[t]) 
        objective += w1*(co2_progn[t] * energy[t]) 

        opti.subject_to(Qout[t] ==  4.64*energy[t] + 0.5336* Qout[t-1])
        opti.subject_to(energy[t] >= 0)
        opti.subject_to(energy[t] <= 300)
        opti.subject_to(Qout[t] >= 0)
        opti.subject_to(Qout[t] <= 1800)


        for scenario in range(num_scenarios):
            objective += 1e-3*s[scenario, t-1] 
            actual_inflow = Qin_forecast[t] 
            delta_h = (100/40)*(Qout[t] - actual_inflow)
            opti.subject_to(height[scenario, t] == height[scenario, t-1] + delta_h)
            opti.subject_to(eta <= 10)  
            opti.subject_to(height[scenario, t] >= 50)  
            opti.subject_to(s[scenario, t-1] <= 200 - height[scenario, t])

    # Now, the proper CVaR constraint:
    opti.subject_to(eta + (1/(1-cvar_alpha)) * (1/num_scenarios) * ca.sum1(s) <= epsilon)

    opti.minimize(objective / num_scenarios)

    try:
        sol = opti.solve()

        return {
                "qout": round(sol.value(Qout[1]), 1),
                "height_ref": round(np.max(sol.value(height[:,1])), 1),
                "energy_ref": round(sol.value(energy[1]), 1),
                "co2_progn": round(sol.value(co2_progn[1]), 4),
                "da_price": round(sol.value(da_prices[1]), 4),
                "qin": round(sol.value(Qin_forecast[1]), 1),
                "objective": round(float(sol.value(opti.f)), 1),
            }

    except Exception as e:
        print("Solver failed to find a solution.")
        print(f"Error: {e}")
        
        print("\nSolver Debug Information:")
        print("Qout:", np.round(opti.debug.value(Qout),1))
        print("h:", np.round(opti.debug.value(height),1))
        print("E:", np.round(opti.debug.value(energy),1))
        print("Objective value:", opti.debug.value(opti.f))
    


class UMPCDataBuffer:
    def __init__(self):
        self.data = {
            'time_utc': [],
            'qout': [],
            'qin': [],
            'qin_q10': [], 
            'qin_q50': [], 
            'qin_q90': [], 
            'height_ref': [],
            'energy_ref': [],
            'co2_progn': [],
            'da_price': [],
            'objective': [],
            'opt_time_umpc': []
        }

    def initialize(self, entry_dict: Dict[str, Any]) -> None:
        for key in self.data:
            self.data[key].append(entry_dict.get(key, None))

    def update(self, entry_dict: Dict[str, Any]) -> None:
        for key in self.data:
            val = entry_dict.get(key, None)
            if isinstance(val, np.generic):
                val = val.item()
            self.data[key].append(val)

    def to_dataframe(self, save: bool = False, file_path: Optional[str] = None, skip_ini: bool = False) -> pl.DataFrame:
        if skip_ini:
            self.data = {k: v[1:] for k, v in self.data.items()}
        df = pl.DataFrame(self.data)
        if save:
            if file_path is None:
                raise ValueError("file_path must be provided when save=True")
            df.write_parquet(file_path)
        return df
        