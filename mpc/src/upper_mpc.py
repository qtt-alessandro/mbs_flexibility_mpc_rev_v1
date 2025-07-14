import casadi as ca 
import numpy as np

def step_upper_level(horizon, prices_values,co2_values,  inflow_values, h_init, energy_init, Qout_init, trigger_values, w1, w2):
    
    breakpoints = [0, 30, 80]
    slopes = [4.82246804, 14.86423641]
    intercepts = [-18.62012472, -319.87317564]
    
    breakpoints = [0, 11, 30, 80]
    slopes = [1.23892075, 6.27193941, 14.6711783]
    intercepts = [0, -55.6090746, -307.586241]
    
    
    # Solver options
    p_opts = {
        "expand":True,
        "print_time": 0,
        "verbose": False,
        "error_on_fail": False,
    }
    s_opts = {
        'max_iter':1000,
        "print_level":0, 
        "warm_start_init_point": "yes"        }

    opti = ca.Opti()

    opti.solver("ipopt",p_opts,s_opts)

    ### Parameters
    prices = opti.parameter(horizon)
    co2_prices = opti.parameter(horizon)
    Qin_forecast = opti.parameter(horizon)
    trigger = opti.parameter(3)
    #inflows_values = opti.parameter(horizon)
    # MPC Variables 
    energy = opti.variable(3, horizon)
    Qout = opti.variable(horizon)
    height = opti.variable(horizon)
    s_h = opti.variable(horizon)
    energy_cost = opti.variable(horizon+1)
    co2_cost = opti.variable(horizon+1)

    # Initial conditions
    opti.set_initial(height[0], h_init)
    opti.set_initial(energy[:, 0], energy_init)
    opti.set_initial(Qout[0], Qout_init)
    opti.set_initial(s_h,0)
    # set values is used only for paramters
    opti.set_value(trigger, trigger_values)
    opti.set_value(prices, prices_values)
    opti.set_value(co2_prices, co2_values)
    opti.set_value(Qin_forecast, inflow_values)
    opti.set_initial(energy_cost, 0)
    opti.set_initial(co2_cost, 0)


    objective = 0
    for t in range(horizon):
       
        objective += (  w1*(prices[t] * (energy[0, t] + energy[1, t] + energy[2, t]))
                        + w2*(co2_prices[t] * (energy[0, t] + energy[1, t] + energy[2, t]))
                        + 1e3*s_h[t]
                        + 0.2* ((energy[:, t] - energy[:, t-1]).T @ (energy[:, t] - energy[:, t-1]))
                        + 1*(ca.if_else(trigger[0] > 0, 0, energy[0, t])
                                + ca.if_else(trigger[1] > 0, 0, energy[1, t])
                                + ca.if_else(trigger[2] > 0, 0, energy[2, t])))

        opti.subject_to(energy_cost[t+1] == w1*(prices[t] * (energy[0, t] + energy[1, t] + energy[2, t])))
        opti.subject_to(co2_cost[t+1] == w2*(co2_prices[t] * (energy[0, t] + energy[1, t] + energy[2, t])))


        # Energy consumption vs outflow Model
        opti.subject_to(Qout[t] ==  ca.if_else(
                                energy[0,t] < breakpoints[1],
                                slopes[0] * energy[0,t] + intercepts[0],
                                ca.if_else(
                                    energy[0,t] < breakpoints[2],
                                    slopes[1] * energy[0,t] + intercepts[1],
                                    slopes[2] * energy[0,t] + intercepts[2]
                                )
                            ) + ca.if_else(
                                energy[1,t] < breakpoints[1],
                                slopes[0] * energy[1,t] + intercepts[0],
                                ca.if_else(
                                    energy[1,t] < breakpoints[2],
                                    slopes[1] * energy[1,t] + intercepts[1],
                                    slopes[2] * energy[1,t] + intercepts[2]
                                )
                            )+ ca.if_else(
                                energy[2,t] < breakpoints[1],
                                slopes[0] * energy[2,t] + intercepts[0],
                                ca.if_else(
                                    energy[2,t-1] < breakpoints[2],
                                    slopes[1] * energy[2,t] + intercepts[1],
                                    slopes[2] * energy[2,t] + intercepts[2]
                                )))
        


        opti.subject_to(height[t] == height[t-1] + (100/40)*(Qin_forecast[t] - Qout[t]))
        opti.subject_to(energy[:, t] >= 0)
        opti.subject_to(energy[:, t] <= 80)
        opti.subject_to(Qout >= 0) 
        opti.subject_to(height[t] <= (200 + s_h[t]))
        opti.subject_to(height[t] >= (70 - s_h[t]))
        opti.subject_to(s_h[t] >= 0)
        

    opti.minimize(objective)

    try:
        sol = opti.solve()

        return (
            sol.value(Qout),
            sol.value(height),
            sol.value(energy), 
            float(ca.sum1(sol.value(energy_cost))),
            float(ca.sum1(sol.value(co2_cost))), 
            float(sol.value(opti.f))
        )
    
    except Exception as e:
        print("Solver failed to find a solution.")
        print(f"Error: {e}")
        
        print("\nSolver Debug Information:")
        print("Qout:", opti.debug.value(Qout))
        print("h:", opti.debug.value(height))
        print("E:", opti.debug.value(energy))
        print("Objective value:", opti.debug.value(opti.f))
    