import casadi as ca 
import numpy as np
import polars as pl 

def step_upper_level(horizon, prices_values, co2_progn_values, inflow_values, h_init, energy_init, Qout_init):

    opti = ca.Opti()

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

    ### Parameters
    da_prices = opti.parameter(horizon)
    co2_progn = opti.parameter(horizon)
    Qin_forecast = opti.parameter(horizon)
    energy = opti.variable(horizon)
    Qout = opti.variable(horizon)
    height = opti.variable(horizon)
    s_h = opti.variable(horizon)

    # set values is used only for paramters
    opti.set_value(da_prices, prices_values)
    opti.set_value(co2_progn, co2_progn_values)
    opti.set_value(Qin_forecast, inflow_values)

    opti.subject_to(height[0] == h_init)
    opti.subject_to(Qout[0] == Qout_init)
    opti.subject_to(energy[0] == energy_init)
    



    objective = 0
    for t in range(1, horizon):
       
        w1 = 10
        w2 = 1e5

        objective +=  w1*(da_prices[t] * energy[t]) + w1*(co2_progn[t] * energy[t]) + w2*s_h[t] + 1e-3*(energy[t] - energy[t-1])**2

        #opti.subject_to(Qout[t] ==  2.107*energy[t] + 0.8032* Qout[t-1])
        opti.subject_to(Qout[t] ==  4.64*energy[t] + 0.5336* Qout[t-1])
        opti.subject_to(height[t] == height[t-1] + (100/40)*(Qin_forecast[t] - Qout[t]))
        opti.subject_to(energy[t] >= 0)
        opti.subject_to(energy[t] <= 300)
        opti.subject_to(Qout[t] >= 0) 
        opti.subject_to(Qout[t] <= 1800) 
        opti.subject_to(height[t] <= 200 + s_h[t])
        opti.subject_to(height[t] >= 70  - s_h[t])
        opti.subject_to(s_h[t] >= 0)
        

    opti.minimize(objective)

    try:
        sol = opti.solve()

        return {
                "qout": round(sol.value(Qout[1]), 1),
                "height_ref": round(sol.value(height[1]), 1),
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
        print("Qout:", opti.debug.value(Qout))
        print("h:", opti.debug.value(height))
        print("E:", opti.debug.value(energy))
        print("Objective value:", opti.debug.value(opti.f))
    

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