# %%
import numpy as np
import cvxpy as cp
import json 
import json 
import pandas as pd 
import polars as pl 
import os 
from lower_mpc_old_arx import step_lower_mpc, LMPCDataBuffer
from upper_mpc import step_upper_level, UMPCDataBuffer
from tqdm import tqdm 
import matplotlib.pyplot as plt 


# %%
root_folder = "/home/alqua/papers/mbs_aggregation_paper_code/mpc/"
params_path = "sys_id/results/"

# %%
data_path_wsl = "/home/alqua/data/pump_station_data/"
upper_mpc_data = pl.read_parquet(os.path.join(data_path_wsl, "sim_data/sim_data_1h_full.par"))
upper_mpc_data_pan = pl.read_parquet(os.path.join(data_path_wsl, "sim_data/sim_data_1h_full.par"))

# %%
start_date = pd.to_datetime("2024-02-15 00:00:00+00:00")
end_date = pd.to_datetime("2024-02-25 08:00:00+00:00")

upper_mpc_data = upper_mpc_data.filter((pl.col("time") >= start_date) & 
                                       (pl.col("time") <= end_date)
                                       ).with_columns(pl.col("CO2Emission")/1000, pl.col("price")/1000)

upper_mpc_data = upper_mpc_data.with_columns(pl.col("time").dt.day().alias("day_index"))
day_sliced_df = upper_mpc_data.filter(pl.col("day_index") == 15)


# %%
with open('mpc/lower_mpc_coefficients.json', 'r') as json_file:
    models_coefficients = json.load(json_file)

# %%
lmpc_data = LMPCDataBuffer()
umpc_data = UMPCDataBuffer()


# %%

init_entry = {
    "time": [0, 1, 2],
    "u1": [100, 110, 120],
    "u2": [0, 0, 0],
    "u3": [0, 0, 0],
    "p1_power": [20, 20, 20],
    "p3_power": [0, 0, 0],
    "p4_power": [0,0,0],
    "qout": [100, 100, 100],
    "qin": [400, 400, 400],
    "height_sys": [100, 101, 102],
    "height_ref": [100, 101, 102],
    "pressure_sys": [0.1, 0.15, 0.13],
    "perf_time_lower": [0.2, 0.21, 0.22],
    "perf_time_upper": [0.25, 0.26, 0.28]}


lmpc_data.initialize(init_entry)

# %%


# %%
slice_df = upper_mpc_data.select(pl.col(["time", "inflow_kf", "CO2Emission","price"]))

k = 0 
zs = 3 
N = 60
horizon = 24
step_size = 1
max_start = 3 #len(slice_df) - horizon + 1

for start_index in tqdm(range(0, max_start, step_size)):
    prices_values = slice_df["price"].slice(start_index, horizon)
    co2_values = slice_df["CO2Emission"].slice(start_index, horizon)
    inflow_values = slice_df["inflow_kf"].slice(start_index, horizon)

    umpc_opt_results = step_upper_level(
        horizon=horizon,
        prices_values=prices_values,
        co2_progn_values=co2_values,
        inflow_values=inflow_values,
        h_init=lmpc_data.data["height_sys"][-1],
        energy_init=lmpc_data.data["p1_power"][-1] + lmpc_data.data["p3_power"][-1] + lmpc_data.data["p4_power"][-1],
        Qout_init=lmpc_data.data["qout"][-1])

    umpc_data.update(umpc_opt_results)
    #print(start_index, start_index+horizon)

    inflow_kf = np.ones(600)*slice_df["inflow_kf"][start_index]
    
    for k in range(0,60):

        u_stack = np.vstack([lmpc_data.data["u1"][-zs:],
                            lmpc_data.data["u2"][-zs:], 
                            lmpc_data.data["u3"][-zs:]])

        power_stack = np.vstack([lmpc_data.data["p1_power"][-zs:],
                            lmpc_data.data["p3_power"][-zs:], 
                            lmpc_data.data["p4_power"][-zs:]])


        res_dict  = step_lower_mpc(Qin_est = inflow_kf[k+zs:k+zs+N+zs],
                                Qout_meas = lmpc_data.data["qout"][-zs:],
                                h_meas = lmpc_data.data["height_sys"][-zs:],
                                w_meas = u_stack,
                                E_meas = power_stack,
                                P_meas = lmpc_data.data["pressure_sys"][-zs:],
                                h_ref = umpc_opt_results["height"],
                                trigger = [1,0,0],
                                N = 60,
                                zs = zs,
                                models_coefficients = models_coefficients)

        res_dict['time'] = 0 
        res_dict['height_ref'] = umpc_opt_results["height"]
        #res_dict['time'] = 0 
        
        lmpc_data.update(res_dict)



# %%
opt_lmpc = lmpc_data.to_dataframe()

# %%
plt.plot(opt_lmpc["height_ref"])
plt.plot(opt_lmpc["height_sys"])

# %%



