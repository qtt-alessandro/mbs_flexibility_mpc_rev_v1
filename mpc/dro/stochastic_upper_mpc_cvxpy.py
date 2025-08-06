import cvxpy as cp
from cvxpy import constraints
import numpy as np
import polars as pl 
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any
from scipy import stats
np.set_printoptions(suppress=True, precision=1)


def step_upper_level(horizon, prices_values, co2_progn_values, inflow_values, h_ini, energy_init, Qout_init, resid):
    
    num_scenarios = resid.shape[0]//24
    usable_length = num_scenarios * 24
    resids_bootstrap = np.clip(resid[:usable_length], -100, 100)
    bootstrap_samples = np.random.choice(resids_bootstrap, size=usable_length, replace=True).reshape((num_scenarios, 24))
    resid = np.random.normal(0, 30, size=(num_scenarios, horizon))
    #resid = np.clip(resid, -5, 10)

    energy = cp.Variable(horizon)
    qout = cp.Variable(horizon)
    height = cp.Variable((num_scenarios, horizon))
    tau = cp.Variable(horizon)
    z = cp.Variable((num_scenarios, horizon))
    z = cp.Variable((num_scenarios, horizon))
    s_upper = cp.Variable(horizon, nonneg=True)
    s_lower = cp.Variable(horizon, nonneg=True)

    tau_lower = cp.Variable(horizon)
    z_lower = cp.Variable((num_scenarios, horizon))

    cvar_alpha = 0.25

    constraints = [
        qout >= 0, 
        qout <= 4800, 
        qout[0] == Qout_init, 
        height[:, 0] == h_ini, 
        energy[0] == energy_init,
        energy >= 0, 
        energy <= 300, 
        qout[1:] == 10*energy[1:] #+ 0.5336*qout[:-1]
]
    obj = 0

    for t in range(1, horizon):
        stage_cost = (energy[t] * prices_values[t] + 
                     energy[t] * co2_progn_values[t] + 
                     1e-3 * (energy[t] - energy[t-1])**2)
        obj += stage_cost


        for s in range(num_scenarios):
            constraints.append(height[s, t] == height[s, t-1] + (1/4)*(inflow_values[t] + bootstrap_samples[s, t] - qout[t]))
            constraints.append(z[s, t] >= height[s, t] - 250 - tau[t])
            constraints.append(z_lower[s, t] >= 20 - height[s, t] - tau_lower[t])
            constraints.append(z_lower[s, t] >= 0)
            constraints.append(z[s, t] >= 0)

        #constraints.append(
        constraints.append(tau[t] + (1/cvar_alpha) * (1/num_scenarios) * cp.sum(z[:, t]) <= s_upper[t])
        #constraints.append(
        constraints.append(tau_lower[t] + (1/cvar_alpha) * (1/num_scenarios) * cp.sum(z_lower[:, t]) <= s_lower[t])
    slack_penalty = 1000 * cp.sum(s_upper[1:] + s_lower[1:])  # High penalty weight
    obj += slack_penalty

    # Modify the existing cvar_penalty line to include both upper and lower
    #cvar_penalty = 1e4 * cp.sum([
    #    tau[t] + (1/cvar_alpha) * (1/num_scenarios) * cp.sum(z[:, t]) + 
    #    tau_lower[t] + (1/cvar_alpha) * (1/num_scenarios) * cp.sum(z_lower[:, t]) 
    #    for t in range(1, horizon)])
    #obj += cvar_penalty 

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(cp.MOSEK, verbose=False, warm_start=True)
    if prob.status == 'infeasible':
        print("Problem is infeasible!")
    # Debug constraints
        for i, constr in enumerate(constraints):
            if not constr.value():
                print(f"Constraint {i} violated: {constr}")
        return None 

    return {
        "qout": qout.value[1],
        "height_ref": height.value[:, 1],
        "energy_ref": energy.value[1],
        "co2_progn": co2_progn_values[1],
        "da_price": prices_values[1],
        "qin": inflow_values[1],
        "objective": obj.value,
    }


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
        