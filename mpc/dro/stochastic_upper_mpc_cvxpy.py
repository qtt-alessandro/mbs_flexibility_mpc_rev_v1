import cvxpy as cp
from cvxpy import constraints
import numpy as np
import polars as pl 
np.set_printoptions(suppress=True, precision=1)


def step_upper_level(horizon, prices_values, co2_progn_values, inflow_values, h_init, energy_init, Qout_init, resid):
    
    num_scenarios = resid.shape[0]//24
    usable_length = num_scenarios * 24
    ### Parameters
    da_prices = prices_values
    co2_progn = co2_progn_values

    energy = cp.Variable(horizon)
    qout = cp.Variable(horizon)
    height = cp.Variable(horizon)
    residuals = np.random.normal(30, 2, size=(horizon))
    
    bootstrap_samples = np.random.choice(resid, size=usable_length, replace=True)
    resids_bootstrap = resid[:usable_length].reshape((num_scenarios, 24))

    print(resids_bootstrap[0, :].shape)
    num_scenarios = resids_bootstrap.shape[0]
    horizon = resids_bootstrap.shape[1]  # Should be 24

    beta = cp.Variable(horizon)                      # one beta per time step
    z = cp.Variable((num_scenarios, horizon)) 

    alpha = 0.9
    beta = cp.Variable()  # auxiliary variable for VaR
    z = cp.Variable(horizon) 

    constraints = [
        qout >= 0, 
        qout <= 1800, 
        qout[0] == Qout_init, 
        height[0] == h_init, 
        energy[0] == energy_init,
        height >= 70,  
        #height <= 200, 
        energy >= 0, 
        energy <= 300 
    ]

    objective_expr = 0

    constraints.append(qout[1:] == 4.64*energy[1:] + 0.5336*qout[:-1])
    constraints.append(height[1:] == height[:-1] + 2.5*(inflow_values[1:] + residuals[1:] - qout[1:]))

    constraints += [
    z >= height + residuals - beta,
    z >= 0,
    beta + (1/(len(residuals)*(1-alpha))) * cp.sum(z) <= 200
]


    for t in range(1, horizon):
        objective_expr += 100*energy[t]*prices_values[t] + 100*energy[t]*prices_values[t]

    objective = cp.Minimize(objective_expr)
    prob = cp.Problem(objective, constraints)
    prob.solve(cp.MOSEK)

    return{'outflow': qout.value,'height': height.value, "energy":energy.value}
    

class UMPCDataBuffer():

    def __init__(self):
        self.data = {
            'time_utc':[],
            'qout': [],
            'qin': [],
            'qin_q10' : [], 
            'qin_q50' : [], 
            'qin_q90' : [], 
            'height_ref': [],
            'energy_ref': [],
            'co2_progn': [],
            'da_price': [],
            'objective': [],
            'opt_time_umpc': []
        }

    def initialize(self, entry_dict):
        for key in self.data:
            self.data[key].append(entry_dict.get(key, None))

    def update(self, entry_dict):
        for key in self.data:
            val = entry_dict.get(key, None)
            if isinstance(val, np.generic):
                val = val.item()
            self.data[key].append(val)

    def to_dataframe(self, save=False, file_path=None, skip_ini=False):
        if skip_ini:
            self.data = {k: v[1:] for k, v in self.data.items()}
        df = pl.DataFrame(self.data)
        if save:
            df.write_parquet(file_path)
        return df