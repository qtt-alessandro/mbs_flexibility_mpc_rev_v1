import cvxpy as cp
from cvxpy import constraints
import numpy as np
import polars as pl 
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any
from scipy import stats
np.set_printoptions(suppress=True, precision=1)


def step_upper_level(horizon, prices_values, co2_progn_values, inflow_values, h_ini, energy_init, Qout_init, resid, num_scenarios,  cvar_alpha=None):
    
    # Ensure numpy float arrays
    prices_values = np.asarray(prices_values, dtype=float)
    co2_progn_values = np.asarray(co2_progn_values, dtype=float)
    inflow_values = np.asarray(inflow_values, dtype=float)
    resid = np.asarray(resid, dtype=float)  # shape (num_scenarios, horizon)

    assert resid.shape == (num_scenarios, horizon)

    energy = cp.Variable(horizon)
    qout   = cp.Variable(horizon)
    height = cp.Variable((num_scenarios, horizon))

    # CVaR variables (upper and lower sides)
    tau_u  = cp.Variable(horizon)
    tau_l  = cp.Variable(horizon)
    z_u    = cp.Variable((num_scenarios, horizon))
    z_l    = cp.Variable((num_scenarios, horizon))

    # Height relaxation (nonnegative)
    epsilon_u = cp.Variable(horizon, nonneg=True)
    epsilon_l = cp.Variable(horizon, nonneg=True)

    cons = [
        qout >= 0, qout <= 4800,
        qout[0] == float(Qout_init),
        height[:, 0] == float(h_ini),
        energy[0] == float(energy_init),
        energy >= 0, energy <= 3000,
        qout[1:] == 10*energy[1:]
    ]

    fac = 1.0/((1.0 - float(cvar_alpha)) * float(num_scenarios))

    obj = 0
    for t in range(1, horizon):
        # Stage cost (adapt weight as needed)
        obj += energy[t]*prices_values[t] + energy[t]*co2_progn_values[t] + 1e1*(energy[t] - energy[t-1])**2

        for s in range(num_scenarios):
            cons += [
                height[s, t] == height[s, t-1] + 0.5*(inflow_values[t] + resid[s, t] - qout[t]),
                z_u[s, t] >= height[s, t] - 250 - tau_u[t],
                z_u[s, t] >= 0,
                z_l[s, t] >= 50 - height[s, t] - tau_l[t],
                z_l[s, t] >= 0
            ]

        # Hard CVaR constraints (no epsilon)
        cons += [
            tau_u[t] + fac * cp.sum(z_u[:, t]) <= 10,
            tau_l[t] + fac * cp.sum(z_l[:, t]) <= 10,
        ]

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.MOSEK, verbose=False, warm_start=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("Problem status:", prob.status)
        return None


    return {
        "qout": qout.value[1].item(),
        "height_ref_scen": height.value[:, 1].copy(),
        "height_ref": float(np.mean(height.value[:, 1])),
        "energy_ref": energy.value[1].item(),
        "co2_progn": co2_progn_values[1].item(),
        "da_price": prices_values[1].item(),
        "qin": inflow_values[1].item(),
        "objective": float(prob.value),
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
            'height_ref_scen': [],
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
        