import numpy as np 
import cvxpy as cp 
import polars as pl 

class LMPCDataBuffer:

    def __init__(self):
        self.data = {
            'p1_power': [],
            'p1_qout': [],
            'u1': [],
            'p3_qout': [], 
            'p3_power': [],
            'u3': [], 
            'p4_power': [],
            'p4_qout': [], 
            'u4': [], 
            'h': []
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
        for k, v in self.data.items():
            self.data[k] = [float(x) for x in v]
        return pl.DataFrame(self.data)
    
 
class LowerMPC:
    def __init__(self, sysid_params, N, Q, R):
        for key, value in sysid_params.items():
            setattr(self, key, value)
        self.N = N
        self.Q = Q
        self.R = R
 
 
    def step(self, ini_dict, inflow, reference):
        N = self.N
        p1_power_ini = ini_dict.get('p1_power')[-1]
        p1_qout_ini = ini_dict.get('p1_qout')[-1]
        p3_qout_ini = ini_dict.get('p3_qout')[-1]
        p3_power_ini = ini_dict.get('p3_power')[-1]
        p4_power_ini = ini_dict.get('p4_power')[-1]
        p4_qout_ini = ini_dict.get('p4_qout')[-1]
        u1_ini = ini_dict.get('u1')[-1]
        u4_ini = ini_dict.get('u4')[-1]
        u3_ini = ini_dict.get('u3')[-1]
        h_ini = ini_dict.get('h')[-1]
        
 
        # Optimization variables
        u1 = cp.Variable(N+1)
        u3 = cp.Variable(N+1)
        u4 = cp.Variable(N+1)
        power_p1 = cp.Variable(N+1)
        power_p3 = cp.Variable(N+1)
        power_p4 = cp.Variable(N+1)
        qout_p1 = cp.Variable(N+1)
        qout_p3 = cp.Variable(N+1)
        qout_p4 = cp.Variable(N+1)
        h = cp.Variable(N+1)
        
        # Initial state and input constraints
        constraints = [
            power_p1[0] == p1_power_ini,
            qout_p1[0] == p1_qout_ini,
            qout_p3[0] == p3_qout_ini,
            power_p3[0] == p3_power_ini,
            power_p4[0] == p4_power_ini,
            qout_p4[0] == p4_qout_ini,
            u1[0] == u1_ini,
            u4[0] == u4_ini,
            u3[0] == u3_ini,
            h[0] == h_ini,
            #h[N] == reference
        ]
 
        upper_bound_lag = 1
        
        
        for k in range(0, N):
            constraints += [
                # Power ARX
                power_p1[k+1] == self.a_p1_power * power_p1[k] + self.b_p1_power * u1[k+1],
                power_p3[k+1] == self.a_p3_power * power_p3[k] + self.b_p3_power * u3[k+1],
                power_p4[k+1] == self.a_p4_power * power_p4[k] + self.b_p4_power * u4[k+1],
 
                # Qout ARX
                qout_p1[k+1] == self.a_p1_qout * qout_p1[k] + self.b_p1_qout * u1[k+1],
                qout_p3[k+1] == self.a_p3_qout * qout_p3[k] + self.b_p3_qout * u3[k+1],  
                qout_p4[k+1] == self.a_p4_qout * qout_p4[k] + self.b_p4_qout * u4[k+1],
 
                
                h[k+1] == h[k] + 100*(inflow[k] - qout_p1[k+1])/(40*240),
 
 
                u1 >= 0, u1 <= 1500,
                u3 >= 0, u3 <= 1500,
                u4 >= 0, u4 <= 1500,
                power_p1 >= 0, power_p1 <= 45,
                power_p3 >= 0, power_p3 <= 45,
                power_p4 >= 0, power_p4 <= 45,
                qout_p1 >= 0, qout_p1 <= 800,
                qout_p3 >= 0, qout_p3 <= 800,
                qout_p4 >= 0, qout_p4 <= 800,
                h >=80, h <= 220,
                
            ]
            
        w1 = 1e3
        w2 = 20
        w3 = 100
        
        ref_traj = reference*np.ones(N+1)

        c1 = cp.sum_squares(h - ref_traj)
        c2 = (cp.sum_squares(cp.diff(u1)) + cp.sum_squares(cp.diff(u3)) + cp.sum_squares(cp.diff(u4)))
        c3 = (cp.sum(power_p1) + cp.sum(power_p3) + cp.sum(power_p4))

        
        cost = w1*c1 + w2*c2 + w3*c3  
 
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.MOSEK, warm_start=True, verbose=False)
 
        N_DECIMALS = 1
        #print("prob status:", prob.status)
 
        result = {
            'u1': np.round(u1.value.copy(), N_DECIMALS)[1] if u1.value is not None else None,
            'u3': np.round(u3.value.copy(), N_DECIMALS)[1] if u3.value is not None else None,
            'u4': np.round(u4.value.copy(), N_DECIMALS)[1] if u4.value is not None else None,
            'p1_power': np.round(power_p1.value.copy(), N_DECIMALS)[1] if power_p1.value is not None else None,
            'p3_power': np.round(power_p3.value.copy(), N_DECIMALS)[1] if power_p1.value is not None else None,   # << this should be power_p3!
            'p4_power': np.round(power_p4.value.copy(), N_DECIMALS)[1] if power_p4.value is not None else None,
            'p1_qout': np.round(qout_p1.value.copy(), N_DECIMALS)[1] if qout_p1.value is not None else None,
            'p3_qout': np.round(qout_p3.value.copy(), N_DECIMALS)[1] if qout_p3.value is not None else None,
            'p4_qout': np.round(qout_p4.value.copy(), N_DECIMALS)[1] if qout_p4.value is not None else None,
            'h': np.round(h.value.copy(), N_DECIMALS)[1] if h.value is not None else None,
        }
        return result
    
    