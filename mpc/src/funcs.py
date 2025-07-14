import numpy as np
import pandas as pd
import logging


class LowerMPCVariablesManager:
    def __init__(self):
        self.h_hist = []
        self.w_hist = []
        self.Qout_hist = []
        self.E_hist = []
        self.P_hist = []
        self.Qin_hist = []
        self.time = np.array([])
        self.perf_time_hist_lower = np.array([])
        self.perf_time_hist_upper = np.array([])
    
    def initialize(self, df, zs):
        # Extract initial conditions from the DataFrame
        self.zs = zs
        self.time = df.index[:zs]
        self.h_hist = df["h"][:zs].values
        self.w_hist = np.vstack([
            df["w1"][:zs].values,
            df["w2"][:zs].values,
            df["w3"][:zs].values
        ])
        
        self.Qout_hist = df["Qout"][:zs].values
        
        self.E_hist = np.vstack([
            df["E1"][:zs].values,
            df["E2"][:zs].values,
            df["E3"][:zs].values
        ])
        
        self.P_hist = df["P"][:zs].values
        self.Qin_hist = df["Qin"][:zs].values
        self.perf_time_hist_lower = np.zeros(zs)
        self.perf_time_hist_upper = np.zeros(zs)

    def update(self, time, sol_w, sol_Qout, sol_h, sol_E, sol_P, inflow_kf_current, perf_time_lower, perf_time_upper):
        """
        Update histories with new data from MPC solution.
        Assumes sol_w and sol_E are 3xN arrays where we want the first prediction step.
        """
        self.time = np.append(self.time, time)
        
        sol_w_step = np.array(sol_w)[:, self.zs].reshape(3, 1)
        sol_w_filtered = np.where(sol_w_step < 1, 0, sol_w_step)
        self.w_hist = np.hstack((self.w_hist, sol_w_filtered))

        sol_E_step = np.array(sol_E)[:, self.zs].reshape(3, 1)
        sol_E_filtered = np.where(sol_E_step < 1, 0, sol_E_step)
        self.E_hist = np.hstack((self.E_hist, sol_E_filtered))

        # Handle scalar values (Qout, h, P)
        # Take the first prediction step (zs index)
        self.Qout_hist = np.append(self.Qout_hist, np.array(sol_Qout)[self.zs])
        self.h_hist = np.append(self.h_hist, np.array(sol_h)[self.zs])
        self.P_hist = np.append(self.P_hist, np.array(sol_P)[self.zs])
        
        # Update other histories
        self.Qin_hist = np.append(self.Qin_hist, inflow_kf_current)
        self.perf_time_hist_lower = np.append(self.perf_time_hist_lower, perf_time_lower)
        self.perf_time_hist_upper = np.append(self.perf_time_hist_upper, perf_time_upper)
        
    def to_dataframe(self):
        """Create a DataFrame from the stored histories"""
        sol_dict = {
            "time": self.time,
            "qout": self.Qout_hist,
            "qin": self.Qin_hist, 
            "w1": self.w_hist[0], 
            "w2": self.w_hist[1],
            "w3": self.w_hist[2],
            "e1": self.E_hist[0],
            "e2": self.E_hist[1],
            "e3": self.E_hist[2], 
            "h": self.h_hist,
            "p": self.P_hist,
            "perf_time_lower": self.perf_time_hist_lower,
            "perf_time_upper": self.perf_time_hist_upper
        }
        return pd.DataFrame(sol_dict).set_index("time").iloc[3:].round(3)
        

class UpperMPCVariablesManager():
    def __init__(self):
        self.time_hist = np.array([])  
        self.Qout_hist = np.array([])     
        self.E_hist = np.empty((3, 0))
        self.href_hist = np.array([])
        self.price_hist = np.array([])
        self.co2_hist = np.array([])
        self.inflow_hist = np.array([])
        self.perf_time_hist = np.array([])
    

    def update(self, time, Qout_hopt, href_hopt, energy_htop, price, inflow, co2_emission):
        self.time_hist = np.hstack([self.time_hist, time])
        self.Qout_hist = np.hstack([self.Qout_hist, Qout_hopt])
        self.E_hist = np.hstack([self.E_hist, energy_htop[:,0].reshape(3,1)])
        self.href_hist = np.hstack([self.href_hist, href_hopt])
        self.price_hist = np.hstack([self.price_hist, price])
        self.inflow_hist = np.hstack([self.inflow_hist, inflow])
        self.co2_hist = np.hstack([self.co2_hist, inflow])
        
        
    def to_dataframe(self):
        sol_dict = {"time": self.time_hist,
                    "qin": self.inflow_hist, 
                    "qout": self.Qout_hist,
                    "e1": self.E_hist[0],
                    "e2": self.E_hist[1],
                    "e3": self.E_hist[2],
                    "href": self.href_hist,
                    "price": self.price_hist, 
                    "co2": self.co2_hist }
        
        logging.info("Successfully created the results DataFrame for Higher MPC.")
        return pd.DataFrame(sol_dict).set_index("time").round(3)


def circular_right_shift(sequence):
    return [sequence[-1]] + sequence[:-1]