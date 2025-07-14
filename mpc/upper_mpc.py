import casadi as ca 
import numpy as np
import polars as pl 

def step_upper_level(horizon, prices_values, co2_progn_values, inflow_values, h_init, energy_init, Qout_init):
    
    breakpoints = [0, 30, 80]
    slopes = [4.82246804, 14.86423641]
    intercepts = [-18.62012472, -319.87317564]
    
    breakpoints = [0, 11, 30, 80]
    slopes = [1.23892075, 6.27193941, 14.6711783]
    intercepts = [0, -55.6090746, -307.586241]
    

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
    #energy_cost = opti.variable(horizon+1)
    #co2_cost = opti.variable(horizon+1)

    # Initial conditions
    opti.set_initial(height[0], h_init)
    opti.set_initial(energy[0], energy_init)
    opti.set_initial(Qout[0], Qout_init)
    opti.set_initial(s_h,0)
    # set values is used only for paramters
    opti.set_value(da_prices, prices_values)
    opti.set_value(co2_progn, co2_progn_values)
    opti.set_value(Qin_forecast, inflow_values)
    #opti.set_initial(energy_cost, 0)
    #opti.set_initial(co2_cost, 0)


    objective = 0
    for t in range(horizon):
       
        w1 = 10
        objective +=  (w1*(da_prices[t] * energy[t])) #(Qout[t] - Qin_forecast[t] )**2 #

        opti.subject_to(Qout[t] ==  2.686*energy[t] + 0.6614* Qout[t-1])
        opti.subject_to(height[t] == height[t-1] + (100/40)*(Qin_forecast[t] - Qout[t]))
        opti.subject_to(energy[t] >= 0)
        opti.subject_to(energy[t] <= 1500)
        opti.subject_to(Qout >= 0) 
        opti.subject_to(height[t] <= 200)
        opti.subject_to(height[t] >= 70)
        #opti.subject_to(s_h[t] >= 0)
        

    opti.minimize(objective)

    try:
        sol = opti.solve()

        return {
            "Qout": sol.value(Qout[0]),
            "height": sol.value(height[0]),
            "cum_energy": sol.value(energy[0]),
            "co2_progn": sol.value(co2_progn[0]),
            "da_price": sol.value(da_prices[0]),
            #"energy_cost": float(ca.sum1(sol.value(energy_cost))),   # if/when you use it
            #"co2_cost": float(ca.sum1(sol.value(co2_cost))),         # if/when you use it
            "objective": float(sol.value(opti.f)),
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
            'Qout': [],
            'height': [],
            'co2_progn': [],
            'cum_energy': [],
            'da_price': [],
            'objective': [],
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

    def to_dataframe(self):
        return pl.DataFrame(self.data)